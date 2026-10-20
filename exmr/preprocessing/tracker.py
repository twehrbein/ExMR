"""Person detection and tracking (YOLO11 + ByteTrack); keeps the most prominent person."""

import queue
import threading
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from exmr.geometry.smoothing import moving_average_smooth
from exmr.preprocessing.video import video_lwh


class Tracker:
    """Tracks all people in a video and returns the boxes of the most prominent one.

    Args:
        checkpoint: YOLO11 detection weights (yolo11m.pt is fast, yolo11x.pt more accurate)
    """

    def __init__(self, checkpoint):
        self.yolo = YOLO(checkpoint)
        # decode the video in a background thread while the detector runs
        self.yolo.add_callback("on_predict_start", _start_video_prefetch)
        self.yolo.add_callback("on_predict_end", _close_video_prefetch)

    def track(self, video_path):
        """Per frame, a list of {"id": track id, "bbx_xyxy": (4,) array} detections."""
        results = self.yolo.track(
            video_path,
            device="cuda",
            conf=0.5,
            classes=0,  # person
            verbose=False,
            stream=True,
            batch=8,
            half=True,
            persist=False,
            # ByteTrack instead of the default BoT-SORT, whose global motion compensation runs
            # on the CPU and dominates this stage, while we only need the main person's boxes
            tracker="bytetrack.yaml",
        )
        # skip Ultralytics' warm-up: its 640x640 dummy image has a shape that the video's frames
        # never have, and cuDNN 9 spends ~1 s setting up each new input shape
        self.yolo.predictor.done_warmup = True
        track_history = []
        try:
            for result in tqdm(results, total=video_lwh(video_path)[0], desc="Tracking"):
                frame = []
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                    bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                    frame = [{"id": i, "bbx_xyxy": box} for i, box in zip(track_ids, bbx_xyxy)]
                track_history.append(frame)
        finally:
            try:
                results.close()
            finally:
                _close_video_prefetch(self.yolo.predictor)
        return track_history

    def get_main_track(self, video_path):
        """(L, 4) bounding boxes [x1, y1, x2, y2] of the most prominent person, i.e. the track
        with the largest summed box area; frames where it is missing are interpolated."""
        L, w, h = video_lwh(video_path)
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        for frame_id, frame in enumerate(self.track(video_path)):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        if not id_to_frame_ids:
            raise RuntimeError(f"No person detected in {video_path}")

        def area_sum(track_id):
            boxes = np.array(id_to_bbx_xyxys[track_id])
            bbx_wh = boxes[:, 2:] - boxes[:, :2]
            return (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()

        track_id = max(id_to_frame_ids, key=area_sum)  # first one wins ties, like a stable sort
        frame_ids = torch.tensor(id_to_frame_ids[track_id])
        bbx_xyxy = torch.tensor(np.array(id_to_bbx_xyxys[track_id]))  # (N, 4)

        # fill missing frames by linear interpolation, then smooth
        present = torch.zeros(L, dtype=torch.bool)
        present[frame_ids] = True
        if len(frame_ids) < L:
            full = torch.zeros((L, 4), dtype=bbx_xyxy.dtype)
            full[present] = bbx_xyxy
            bbx_xyxy = _interpolate_missing(full, present)
        bbx_xyxy = moving_average_smooth(bbx_xyxy, window_size=5, dim=0)
        return moving_average_smooth(bbx_xyxy, window_size=5, dim=0)


def _interpolate_missing(data, present):
    """Linear interpolation over each run of missing frames; runs at the start/end copy the
    nearest present frame."""
    data = data.clone()
    for missing in _runs_of_true(~present):
        if missing[0] - 1 < 0:
            data[missing] = data[missing[-1] + 1].clone()
        elif missing[-1] + 1 >= len(data):
            data[missing] = data[missing[0] - 1].clone()
        else:
            prev = data[missing[0] - 1]
            next = data[missing[-1] + 1]
            t = torch.linspace(0, 1, len(missing) + 2)[1:-1][:, None]
            data[missing] = t * (next - prev)[None] + prev[None]
    return data


def _runs_of_true(mask):
    """(L,) bool -> list of index tensors, one per run of consecutive True values."""
    padded = torch.cat([torch.tensor([False]), mask, torch.tensor([False])])
    diffs = torch.diff(padded.int())
    starts = (diffs == 1).nonzero().flatten()
    ends = (diffs == -1).nonzero().flatten()
    return [torch.arange(s, e) for s, e in zip(starts, ends)]


class _PrefetchVideo:
    """Wraps ultralytics' video loader to decode in a background thread (at most two batches
    ahead). Only the worker thread touches the OpenCV capture; batches are consumed in order."""

    def __init__(self, loader):
        self.loader = loader
        self._pending = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = None

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._pending.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _decode(self):
        try:
            for batch in self.loader:
                if self._stop.is_set():
                    break
                self._put((True, batch))
        except Exception as error:
            self._put((False, error))
        finally:
            try:
                if self.loader.cap is not None:
                    self.loader.cap.release()
            finally:
                self._put((True, None))

    def __iter__(self):
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._decode, daemon=True)
            self._thread.start()
        return self

    def __next__(self):
        if self._stop.is_set():
            raise StopIteration
        if self._thread is None:
            iter(self)
        ok, item = self._pending.get()
        if not ok:
            self.close()
            raise item
        if item is None:
            self.close()
            raise StopIteration
        return item

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Video decoder did not stop within five seconds")
        elif self.loader.cap is not None:
            self.loader.cap.release()
        while True:  # release queued frames even if inference exited early
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break


def _start_video_prefetch(predictor):
    loader = predictor.dataset
    if loader.nf != 1 or loader.mode != "video":
        raise ValueError("Tracker decode prefetch expects exactly one video file")
    predictor.dataset = _PrefetchVideo(loader)


def _close_video_prefetch(predictor):
    if isinstance(getattr(predictor, "dataset", None), _PrefetchVideo):
        predictor.dataset.close()
