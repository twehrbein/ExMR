"""run_video: video -> preprocessing -> ExMR -> results (+ rendered videos).

Every preprocessing stage is cached in <output_dir>/cache together with the settings it was
computed with (including those of the stages it depends on), so rerunning with other settings
recomputes exactly the affected stages.
"""

import gc
import importlib.util
import json
import logging
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from exmr import __version__
from exmr.assets import RTMW_SHA256, Assets
from exmr.export import save_amass
from exmr.geometry.camera import bbx_xys_from_xyxy, cam_angvel_from_rotations, estimate_K
from exmr.preprocessing import video as V
from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD, detect_keypoints
from exmr.preprocessing.keypoints.mappings import COCO23, COCO63_LEFT_HAND, COCO63_RIGHT_HAND

log = logging.getLogger("exmr")


@dataclass
class Options:
    pose_backend: str = "sapiens2"  # 2D keypoints: "sapiens2" or "rtmw"
    sapiens2_resolution: str = "448x336"  # or "576x432", "1024x768" (slower, similar accuracy)
    detector: str = "yolo11m"  # person detector: "yolo11m" or "yolo11x" (slower)
    static_cam: bool = False  # the camera is known to be static: no visual odometry
    vo: str = "simplevo"  # camera motion if not static_cam: "simplevo" or "dpvo" (optional)
    fps30: bool = False  # resample to 30 fps first (usually not needed; faster for e.g. 60 fps)
    snap_to_ground: bool = False  # flat ground, nothing to stand on: no vertical drift
    smoothing: str = "global"  # smooth the motion over time: "global" (world frame), "both", "none"
    render: bool = True
    keep_frames: bool = False  # keep the extracted JPEG frames

    def __post_init__(self):
        for name, choices in [
            ("pose_backend", ["sapiens2", "rtmw"]),
            ("sapiens2_resolution", ["1024x768", "576x432", "448x336"]),
            ("detector", ["yolo11m", "yolo11x"]),
            ("vo", ["simplevo", "dpvo"]),
            ("smoothing", ["global", "both", "none"]),
        ]:
            if getattr(self, name) not in choices:
                raise ValueError(f"Options.{name} must be one of {choices}")


def run_video(video_path, output_dir, options=None, assets=None):
    """Reconstruct the main person of a video. Returns the results dict saved to
    <output_dir>/results.pt (see docs/OUTPUTS.md)."""
    options = options or Options()
    assets = assets or Assets()
    use_dpvo = options.vo == "dpvo" and not options.static_cam
    assets.check(options.pose_backend, options.detector, use_dpvo)
    if use_dpvo and importlib.util.find_spec("dpvo") is None:
        raise RuntimeError(
            "DPVO is not installed: see the optional DPVO section of docs/INSTALL.md, or use "
            "SimpleVO (--vo simplevo)"
        )
    video_path, output_dir = Path(video_path), Path(output_dir)
    name = video_path.stem
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    run_info = {"video": str(video_path), "options": asdict(options), "exmr": __version__}
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")

    if options.fps30:
        video_path = _resample_to_30fps(video_path, output_dir)
    frame_dir = output_dir / "frames"
    num_frames = _extract_frames(video_path, frame_dir)
    if V.video_lwh(video_path)[0] != num_frames:
        # the container's frame count disagrees with the decoded frames (variable frame rate or
        # damaged container): rebuild a constant-frame-rate video from the frames, so that every
        # reader sees the same frames
        log.warning("Frame count mismatch, rebuilding a constant-frame-rate video")
        cfr_path = output_dir / f"{name}_cfr.mp4"
        if not cfr_path.exists():
            V.encode_frames(frame_dir, cfr_path, fps=V.video_fps(video_path))
        video_path = cfr_path
    width, height = V.frame_size(frame_dir)
    fps = V.video_fps(video_path)
    log.info(f"{video_path}: {num_frames} frames, {width}x{height}, {fps:.2f} fps")

    data = _preprocess(video_path, frame_dir, num_frames, width, height, options, assets, cache)

    log.info("Running ExMR")
    results = _run_model(data, options, fps, assets)
    results["fps"] = fps
    _save_results(results, output_dir)
    save_amass(results["smplx_global"], fps, output_dir / f"{name}_amass.npz", assets.smplx)

    if options.render:
        log.info("Rendering")
        _render(results, frame_dir, output_dir, name, fps, assets)
    if not options.keep_frames:
        shutil.rmtree(frame_dir)
    return results


def _preprocess(video_path, frame_dir, L, width, height, options, assets, cache):
    detector = assets.yolo(options.detector).name
    tracker_settings = {"video": str(video_path), "frames": L, "detector": detector}
    backend = options.pose_backend
    kp_settings = {**tracker_settings, "pose_backend": backend}
    if backend == "sapiens2":
        kp_settings["resolution"] = options.sapiens2_resolution
    vo_cache = cache / "camera_rotations.pt"
    vo_settings = {"video": str(video_path), "frames": L, "odometry": options.vo}

    # SimpleVO needs only the video and the CPU: it runs in a separate process while the GPU
    # stages below run
    simplevo = None
    use_simplevo = options.vo == "simplevo" and not options.static_cam
    if use_simplevo and not _is_cached(vo_cache, vo_settings):
        log.info("Starting visual odometry in the background")
        simplevo = _in_background(_simplevo_rotations, video_path)

    def track():
        from exmr.preprocessing.tracker import Tracker

        return Tracker(assets.yolo(options.detector)).get_main_track(video_path).float()

    bbx_xyxy = _cached(cache / "tracks.pt", tracker_settings, track)  # (L, 4)
    bbx_xys = bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
    K = estimate_K(width, height)[None].repeat(L, 1, 1)  # (L, 3, 3)

    threshold = VISIBILITY_THRESHOLD[backend]

    def keypoints():
        if backend == "sapiens2":
            return detect_keypoints(
                backend, assets.sapiens2, frame_dir, bbx_xyxy,
                resolution=options.sapiens2_resolution,
            )  # fmt: skip
        return detect_keypoints(backend, assets.rtmw, frame_dir, bbx_xyxy, sha256=RTMW_SHA256)

    kp2d = _cached(cache / "keypoints.pt", kp_settings, keypoints)  # (L, 63, 3)

    def body_features():
        from exmr.preprocessing.body_features import SamBodyFeatures

        return SamBodyFeatures(assets.sam_3d_body)(frame_dir, bbx_xyxy, K)

    body_feat = _cached(cache / "body_features.pt", tracker_settings, body_features)

    def hand_features():
        from exmr.preprocessing.hand_features import HamerHandFeatures, hand_boxes_from_keypoints

        boxes = torch.stack(
            [
                hand_boxes_from_keypoints(kp2d[:, COCO63_LEFT_HAND], threshold),
                hand_boxes_from_keypoints(kp2d[:, COCO63_RIGHT_HAND], threshold),
            ],
            dim=1,
        )  # (L, 2, 4)
        return HamerHandFeatures(assets.hamer)(frame_dir, boxes)

    hand_feat = _cached(cache / "hand_features.pt", kp_settings, hand_features)

    if options.static_cam:
        R_w2c = torch.eye(3).repeat(L, 1, 1)
    else:

        def camera_rotations():
            from exmr.preprocessing.visual_odometry import DPVOOdometry

            if options.vo == "dpvo":
                return DPVOOdometry(assets.dpvo)(video_path, K[0])
            return torch.from_numpy(simplevo.result())  # waits for SimpleVO to finish

        R_w2c = _cached(vo_cache, vo_settings, camera_rotations)

    # body + foot keypoints; RTMW's raw scores become a binary visibility for the network
    kp23 = kp2d[:, COCO23].clone()
    if backend == "rtmw":
        kp23[..., 2] = (kp23[..., 2] > threshold).to(kp23.dtype)
    return {
        "kp2d": kp23,
        "bbx_xys": bbx_xys,
        "bbx_xyxy": bbx_xyxy,
        "K_fullimg": K,
        "cam_angvel": cam_angvel_from_rotations(R_w2c),
        "body_feat": body_feat,
        "hand_feat": hand_feat,
    }


def _simplevo_rotations(video_path):
    """SimpleVO's (L, 3, 3) camera rotations, as a numpy array: torch tensors would be returned
    from the separate process through shared memory."""
    from exmr.preprocessing.visual_odometry import SimpleVO

    return SimpleVO(progress=False)(video_path).numpy()


def _run_model(data, options, fps, assets):
    from exmr.model.exmr import ExMR

    model = ExMR.from_pretrained(assets.exmr, assets.smplx)
    inputs = {
        k: data[k][None].cuda()
        for k in ("kp2d", "bbx_xys", "K_fullimg", "cam_angvel", "body_feat", "hand_feat")
    }
    out = model(
        **inputs,
        static_cam=options.static_cam,
        snap_to_ground=options.snap_to_ground,
        smooth_incam=options.smoothing == "both",
        smooth_global=options.smoothing != "none",
        fps=fps,
    )
    del model
    _free_gpu_memory()
    return {
        "smplx_incam": {k: v[0].cpu() for k, v in out["smplx_incam"].items()},
        "smplx_global": {k: v[0].cpu() for k, v in out["smplx_global"].items()},
        "static_conf_logits": out["static_conf_logits"][0].cpu(),
        "K_fullimg": data["K_fullimg"],
        "bbx_xyxy": data["bbx_xyxy"],
    }


def _save_results(results, output_dir):
    _atomic_save(results, output_dir / "results.pt")
    flat = {}
    for key, value in results.items():
        if isinstance(value, dict):
            flat |= {f"{key}_{k}": v.numpy() for k, v in value.items()}
        else:
            flat[key] = np.asarray(value)
    np.savez(output_dir / "results.npz", **flat)


def _render(results, frame_dir, output_dir, name, fps, assets):
    from exmr.body_models.smplx_model import SmplxModel
    from exmr.vis.render import render_videos

    smplx = SmplxModel(assets.smplx).cuda()
    with torch.no_grad():
        verts_incam = smplx(**{k: v.cuda() for k, v in results["smplx_incam"].items()})
        verts_global = smplx(**{k: v.cuda() for k, v in results["smplx_global"].items()})
    render_videos(
        frame_dir, verts_incam, verts_global, smplx.faces, smplx.J_regressor.cuda(),
        results["K_fullimg"][0], fps,
        [output_dir / f"{name}_{view}.mp4" for view in ("incam", "global", "incam_global")],
    )  # fmt: skip


def _in_background(fn, *args):
    """The future result of fn(*args), computed in a new process. The process is started with
    "spawn": a forked process would inherit this process's state (e.g. CUDA, OpenCV's threads)."""
    executor = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    future = executor.submit(fn, *args)
    executor.shutdown(wait=False)  # the process exits once fn has returned
    return future


def _is_cached(path, settings):
    """Whether path holds a value cached with these settings (see _cached)."""
    if not path.exists():
        return False
    return torch.load(path, map_location="cpu", weights_only=True)["settings"] == settings


def _cached(path, settings, compute):
    """compute(), cached in path together with settings; recomputed when the settings differ."""
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["settings"] == settings:
            log.info(f"Using cached {path.name}")
            return cached["value"]
    log.info(f"Computing {path.stem}")
    value = compute()
    _atomic_save({"settings": settings, "value": value}, path)
    _free_gpu_memory()
    return value


def _atomic_save(obj, path):
    """Save to a temporary file first, so that an interrupted run leaves no partial file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _extract_frames(video_path, frame_dir):
    marker = frame_dir / ".extracted"
    if marker.exists() and marker.read_text().split("\n")[0] == str(video_path):
        return int(marker.read_text().split("\n")[1])
    log.info(f"Extracting frames to {frame_dir}")
    num_frames = V.extract_frames(video_path, frame_dir)
    marker.write_text(f"{video_path}\n{num_frames}")
    return num_frames


def _resample_to_30fps(video_path, output_dir):
    fps = V.video_fps(video_path)
    if abs(fps - 30) < 0.5:
        return video_path
    out = output_dir / f"{video_path.stem}_30fps.mp4"
    if not out.exists():
        log.info(f"Resampling {fps:.2f} fps to 30 fps")
        V.resample_fps(video_path, out, fps=30)
    return out


def _free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
