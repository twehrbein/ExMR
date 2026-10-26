"""RTMW-x whole-body 2D keypoints (ONNX Runtime, FP16 model, no flip test) -> COCO-63.

The crop (top_down_affine and helpers) and SimCC decoding (get_simcc_maximum) are adapted from
RTMLib (https://github.com/Tau-J/rtmlib), licensed under the Apache License 2.0, see
licenses/RTMLib_LICENSE.txt.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from exmr.preprocessing.keypoints.mappings import COCO133_TO_COCO63

INPUT_SIZE = (288, 384)  # width, height
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
BOX_PADDING = 1.25


class RTMWKeypoints:
    """2D keypoints of one person per frame, given its bounding boxes.

    Args:
        checkpoint: RTMW-x 384x288 ONNX model with FP32 inputs/outputs
        sha256: expected checksum of the checkpoint (None to skip the check)
    """

    def __init__(self, checkpoint, sha256=None, batch_size=32, num_workers=8, device_id=0):
        if sha256 is not None and _sha256(checkpoint) != sha256:
            raise ValueError(f"Unexpected RTMW checkpoint (SHA256 mismatch): {checkpoint}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.session = ort.InferenceSession(
            str(checkpoint),
            providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime CUDA provider unavailable for RTMW")

    def warm_up(self):
        """Run one batch of zeros: ONNX Runtime and cuDNN set the network up for the batch shape
        on its first use (~7 s), which can thereby happen while e.g. the tracker runs."""
        inp = self.session.get_inputs()[0]
        w, h = INPUT_SIZE
        self.session.run(None, {inp.name: np.zeros((self.batch_size, 3, h, w), dtype=np.float32)})

    def __call__(self, frame_dir, bbx_xyxy):
        """frame_dir with %06d.jpg frames, bbx_xyxy (L, 4) -> (L, 63, 3) [x, y, raw score]"""
        boxes = bbx_xyxy.cpu().numpy()
        loader = torch.utils.data.DataLoader(
            _FrameDataset(frame_dir, boxes),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            **({"prefetch_factor": 4} if self.num_workers else {}),
        )
        inp = self.session.get_inputs()[0]
        out_names = [v.name for v in self.session.get_outputs()]
        result = np.empty((len(boxes), 63, 3), dtype=np.float32)
        cursor = 0
        with tqdm(total=len(boxes), desc="RTMW-x keypoints", unit="frame") as pbar:
            for batch in loader:
                images = batch["input"].numpy()
                n = len(images)
                # pad the last batch to the full batch size: ONNX Runtime + cuDNN 9 set up the
                # convolutions anew (~7 s) for every new input shape
                images = np.pad(images, [(0, self.batch_size - n)] + [(0, 0)] * 3, mode="edge")
                simcc_x, simcc_y = self.session.run(out_names, {inp.name: images})
                simcc_x, simcc_y = simcc_x[:n], simcc_y[:n]
                center, scale = batch["center"].numpy(), batch["scale"].numpy()
                xy, scores = _decode(simcc_x, simcc_y, center, scale)
                kp = np.concatenate((xy, scores[..., None]), axis=-1)
                result[cursor : cursor + len(kp)] = kp[:, COCO133_TO_COCO63]
                cursor += len(kp)
                pbar.update(len(kp))
        if cursor != len(boxes) or not np.isfinite(result).all():
            raise RuntimeError("Incomplete or non-finite RTMW output")
        return torch.from_numpy(result)


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, boxes):
        self.paths = [str(Path(frame_dir) / f"{i:06d}.jpg") for i in range(len(boxes))]
        self.boxes = boxes
        self.mean = np.array(MEAN)  # float64 normalization, as in the evaluated setup
        self.std = np.array(STD)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        if image is None:
            raise ValueError(f"Cannot decode {self.paths[idx]}")
        center, scale = _bbox_xyxy2cs(self.boxes[idx], padding=BOX_PADDING)
        crop, scale = _top_down_affine(INPUT_SIZE, scale, center, image)
        crop = (crop - self.mean) / self.std
        chw = np.ascontiguousarray(crop.transpose(2, 0, 1), dtype=np.float32)
        return {
            "input": torch.from_numpy(chw),
            "center": torch.from_numpy(np.asarray(center, dtype=np.float32)),
            "scale": torch.from_numpy(np.asarray(scale, dtype=np.float32)),
        }


def _decode(simcc_x, simcc_y, centers, scales):
    """SimCC outputs -> keypoints in image coordinates and raw scores."""
    locs, scores = _get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs / 2.0  # SimCC split ratio
    input_size = np.asarray(INPUT_SIZE, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    keypoints = keypoints / input_size * scales[:, None, :]
    keypoints = keypoints + centers[:, None, :] - scales[:, None, :] / 2
    return keypoints, scores.astype(np.float32)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----- adapted from RTMLib (rtmlib/tools/pose_estimation/{pre,post}_processings.py) ----- #


def _bbox_xyxy2cs(bbox, padding=1.0):
    """(4,) box [x1, y1, x2, y2] -> center (2,) and padded scale (2,)."""
    x1, y1, x2, y2 = np.hsplit(bbox[None, :], [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    return center[0], scale[0]


def _get_3rd_point(a, b):
    """b + (a - b) rotated by 90 degrees anticlockwise."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _get_warp_matrix(center, scale, output_size):
    """2x3 affine matrix that maps the box (center, scale) onto an output_size image."""
    src_w = scale[0]
    dst_w, dst_h = output_size

    src_dir = np.array([0.0, src_w * -0.5])  # no rotation
    dst_dir = np.array([0.0, dst_w * -0.5])

    shift = np.array((0.0, 0.0))
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _top_down_affine(input_size, bbox_scale, bbox_center, img):
    """Crop the box from img to input_size (box grown to the input aspect ratio)."""
    w, h = input_size
    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        b_w > b_h * aspect_ratio,
        np.hstack([b_w, b_w / aspect_ratio]),
        np.hstack([b_h * aspect_ratio, b_h]),
    )
    warp_mat = _get_warp_matrix(bbox_center, bbox_scale, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    return img, bbox_scale


def _get_simcc_maximum(simcc_x, simcc_y):
    """SimCC (N, K, W) x/y -> locations (N, K, 2) and scores (N, K) of the maxima."""
    N, K, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    locs = np.stack((np.argmax(simcc_x, axis=1), np.argmax(simcc_y, axis=1)), axis=-1)
    locs = locs.astype(np.float32)
    vals = 0.5 * (np.amax(simcc_x, axis=1) + np.amax(simcc_y, axis=1))
    locs[vals <= 0.0] = -1
    return locs.reshape(N, K, 2), vals.reshape(N, K)
