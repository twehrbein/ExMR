"""HaMeR image features: the MANO decoder token of each hand, one pair per frame.

The hand crop (_crop_hand) is adapted from HaMeR's hamer/datasets/utils.py
(https://github.com/geopavlakos/hamer, MIT License, Copyright (c) 2023 UC Regents, Georgios
Pavlakos), see THIRD_PARTY_NOTICES.md.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from hamer.configs import get_config
from hamer.models import HAMER
from torch.utils.data import DataLoader
from tqdm import tqdm


def hand_boxes_from_keypoints(kp2d, threshold, min_kpts=4):
    """Per-frame boxes around the confident keypoints of one hand.

    Args:
        kp2d: (L, J, 3) hand keypoints [x, y, score]
        threshold: keypoints with score > threshold are used
        min_kpts: frames with fewer confident keypoints get an all-zero box (= no hand)
    Returns:
        (L, 4) boxes [x1, y1, x2, y2]
    """
    x, y = kp2d[..., 0], kp2d[..., 1]
    mask = kp2d[..., 2] > threshold
    inf = x.new_tensor(float("inf"))
    boxes = torch.stack(
        [
            torch.where(mask, x, inf).amin(dim=1),
            torch.where(mask, y, inf).amin(dim=1),
            torch.where(mask, x, -inf).amax(dim=1),
            torch.where(mask, y, -inf).amax(dim=1),
        ],
        dim=1,
    )
    valid = mask.sum(dim=1) >= min_kpts
    return torch.where(valid.unsqueeze(1), boxes, torch.zeros_like(boxes))


class HamerHandFeatures:
    """Per-frame HaMeR tokens (1024-d) of both hands, given their bounding boxes.

    The token (the MANO decoder output consumed by HaMeR's regression heads) is read with a
    forward hook. Hands are cropped like in HaMeR's demo; the left hand is mirrored, as HaMeR
    only models right hands. Frames without a hand box get an all-black crop.

    Args:
        data_dir: HaMeR's data directory (from its hamer_demo_data download): contains
            hamer_ckpts/checkpoints/hamer.ckpt, hamer_ckpts/model_config.yaml,
            data/mano_mean_params.npz, and the MANO model data/mano/MANO_RIGHT.pkl
        fp16: run the network under float16 autocast (~1.9x faster, tokens nearly identical)
    """

    TOKEN_DIM = 1024

    def __init__(self, data_dir, fp16=True, batch_size=16, num_workers=8, device="cuda"):
        data_dir = Path(data_dir)
        cfg = get_config(str(data_dir / "hamer_ckpts" / "model_config.yaml"))
        cfg.defrost()
        cfg.MANO.MODEL_PATH = str(data_dir / "data" / "mano")
        cfg.MANO.MEAN_PARAMS = str(data_dir / "data" / "mano_mean_params.npz")
        if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.freeze()
        self.cfg = cfg
        # like hamer.models.load_hamer, without the renderer; built on the GPU: initializing the
        # weights on the CPU takes ~2 s, and the checkpoint replaces them anyway
        with torch.device(device):
            self.model = HAMER.load_from_checkpoint(
                str(data_dir / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"),
                strict=False,
                cfg=cfg,
                init_renderer=False,
            )
        self.model.eval().to(device)
        self.fp16, self.batch_size, self.num_workers, self.device = (
            fp16, batch_size, num_workers, device
        )  # fmt: skip
        self._tokens = None
        self.model.mano_head.transformer.register_forward_hook(self._store_tokens)

    def _store_tokens(self, _module, _inputs, output):
        self._tokens = output.squeeze(1)

    @torch.no_grad()
    def __call__(self, frame_dir, hand_bbx_xyxy):
        """frame_dir with %06d.jpg frames, hand_bbx_xyxy (L, 2, 4) [left, right] boxes
        -> (L, 2, 1024) tokens"""
        dataset = _FrameDataset(Path(frame_dir), hand_bbx_xyxy, self.cfg)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        tokens = torch.zeros((len(dataset), 2, self.TOKEN_DIM), dtype=torch.float32)
        i = 0
        with tqdm(total=len(dataset), desc="HaMeR hand features", unit="frame") as pbar:
            for crops in loader:
                bs = crops.shape[0]
                tokens[i : i + bs] = (
                    self._forward(crops.reshape(-1, 3, 256, 256))
                    .reshape(bs, 2, self.TOKEN_DIM)
                    .cpu()
                )
                i += bs
                pbar.update(bs)
        if not torch.isfinite(tokens).all():
            raise RuntimeError(
                "Non-finite HaMeR hand features, probably a float16 overflow: disable FP16 "
                "accumulation in run_video (exmr/pipeline.py) and please report this"
            )
        return tokens

    def _forward(self, crops):
        """(B, 3, 256, 256) normalized crops -> (B, 1024) tokens"""
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.fp16):
            # HaMeR's ViT backbone sees the central 256x192 of the square crop
            feats = self.model.backbone(crops[:, :, :, 32:-32].to(self.device))
            self.model.mano_head(feats)  # the hook stores the tokens
        return self._tokens.float()


class _FrameDataset(torch.utils.data.Dataset):
    """Normalized (2, 3, S, S) [left, right] hand crops per frame."""

    def __init__(self, frame_dir, hand_bbx_xyxy, cfg):
        self.frame_dir, self.hand_bbx_xyxy = frame_dir, hand_bbx_xyxy
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
        self.bbox_shape = cfg.MODEL.BBOX_SHAPE

    def __len__(self):
        return len(self.hand_bbx_xyxy)

    def __getitem__(self, idx):
        path = self.frame_dir / f"{idx:06d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read frame {path}")

        boxes = self.hand_bbx_xyxy[idx].numpy()
        center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scale = 2.0 * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0  # box enlarged 2x, in 200 px

        crops = []
        for i in range(2):  # left, right
            if scale[i, 0] == 0 or scale[i, 1] == 0:  # no hand in this frame
                crops.append(np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32))
                continue
            bbox_size = _expand_to_aspect_ratio(scale[i] * 200, self.bbox_shape).max()
            patch = _crop_hand(
                img.copy(), center[i, 0], center[i, 1], bbox_size, self.img_size, flip=i == 0
            )
            patch = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # BGR HWC -> RGB CHW
            for c in range(min(img.shape[2], 3)):
                patch[c] = (patch[c] - self.mean[c]) / self.std[c]
            crops.append(patch[None])
        return np.concatenate(crops, axis=0)


def _expand_to_aspect_ratio(size_wh, target_aspect_ratio):
    """Grow a (w, h) box size to the target (w, h) aspect ratio."""
    w, h = size_wh
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)])
    return np.array([max(h * w_t / h_t, w), h])


def _crop_hand(img, c_x, c_y, bbox_size, patch_size, flip):
    """Square crop of size bbox_size around (c_x, c_y), resized to patch_size (optionally
    mirrored), with black borders."""
    if flip:
        img = img[:, ::-1, :]
        c_x = img.shape[1] - c_x - 1
    src_center = np.array([c_x, c_y], dtype=np.float64)
    src_down = np.array([0, bbox_size * 0.5], dtype=np.float32)
    src_right = np.array([bbox_size * 0.5, 0], dtype=np.float32)
    dst_center = np.array([patch_size * 0.5, patch_size * 0.5], dtype=np.float32)
    dst_down = np.array([0, patch_size * 0.5], dtype=np.float32)
    dst_right = np.array([patch_size * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_down
    src[2, :] = src_center + src_right
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_down
    dst[2, :] = dst_center + dst_right
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return cv2.warpAffine(
        img,
        trans,
        (int(patch_size), int(patch_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
