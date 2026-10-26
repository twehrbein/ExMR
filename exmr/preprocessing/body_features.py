"""SAM 3D Body image features: the body decoder's pose token, one per frame."""

import contextlib
import io
import logging
import re
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

# SAM 3D Body is not pip-installable; it is a git submodule of this repository
SAM_3D_BODY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sam-3d-body"

# Parameters that SAM 3D Body's checkpoint lacks by design: the MHR body model's buffers (loaded
# from mhr_model.pt), the hand PCA copies (set in SAM3DBody.__init__) and DINOv3's mask token
# (zero-initialized; unused at inference)
_EXPECTED_MISSING_KEY = re.compile(
    r"backbone\.encoder\.mask_token|head_pose(_hand)?\.(hand_pose_comps_ori|mhr\..+)"
)
_MISMATCH = "The model and loaded state dict do not match exactly"
_MISSING = "missing keys in source state_dict: "


def _drop_expected_mismatch(record):
    """Logging filter: drop the checkpoint-mismatch warning if it lists only expected keys."""
    msg = record.getMessage()
    if not msg.startswith(_MISMATCH) or "unexpected key" in msg or _MISSING not in msg:
        return True
    keys = msg.split(_MISSING, 1)[1].strip().split(", ")
    return not all(_EXPECTED_MISSING_KEY.fullmatch(k) for k in keys)


def _drop_ignored_kwargs(record):
    """Logging filter: DINOv3 ignores the config's drop_path, which only matters in training."""
    return not record.getMessage().startswith("Ignored kwargs: {'drop_path'")


@contextlib.contextmanager
def _quiet_model_loading():
    """Hide SAM 3D Body's expected messages while loading the model (real problems still show)."""
    filters = [
        (logging.getLogger("sam_3d_body.utils.checkpoint"), _drop_expected_mismatch),
        (logging.getLogger("dinov3"), _drop_ignored_kwargs),
    ]
    for logger, f in filters:
        logger.addFilter(f)
    try:
        # load_sam_3d_body prints "Loading SAM 3D Body model..."
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for logger, f in filters:
            logger.removeFilter(f)


def _import_sam_3d_body():
    try:
        import sam_3d_body  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(SAM_3D_BODY_ROOT))
    with warnings.catch_warnings():
        # MHR head: optional Momentum dependency, unused by the PyTorch MHR model
        warnings.filterwarnings("ignore", message="Momentum is (not )?enabled")
        import sam_3d_body
        from sam_3d_body.data import transforms
        from sam_3d_body.data.utils.io import load_image
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from sam_3d_body.utils import recursive_to

    return sam_3d_body, transforms, load_image, prepare_batch, recursive_to


class SamBodyFeatures:
    """Per-frame SAM 3D Body pose tokens (1024-d) of one person, given its bounding boxes.

    The token is read from the body decoder with a forward hook (it is not part of the model's
    output). The hand-detection tokens stay enabled as in the original model config, since
    they influence the body token through self-attention.

    Args:
        checkpoint_dir: directory with model.ckpt, model_config.yaml and assets/mhr_model.pt of
            facebook/sam-3d-body-dinov3
        fp16: run the backbone in float16 instead of the checkpoint's bfloat16 (SAM 3D Body's own
            FP16_TYPE option). Faster only with FP16 accumulation
            (torch.backends.cuda.matmul.allow_fp16_accumulation); tokens change at the 1e-3 level
    """

    TOKEN_DIM = 1024

    def __init__(self, checkpoint_dir, fp16=False, batch_size=32, num_workers=8, device="cuda"):
        sam_3d_body, self._transforms, self._load_image, self._prepare_batch, self._to = (
            _import_sam_3d_body()
        )
        checkpoint_dir = Path(checkpoint_dir)
        # built on the GPU: initializing the weights on the CPU takes ~5 s, and the checkpoint
        # replaces them anyway
        with _quiet_model_loading(), torch.device(device):
            model, model_cfg = sam_3d_body.load_sam_3d_body(
                str(checkpoint_dir / "model.ckpt"),
                device=device,
                mhr_path=str(checkpoint_dir / "assets" / "mhr_model.pt"),
            )
        assert model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS
        self.model = model.eval()
        if fp16:
            for t in [*model.backbone.parameters(), *model.backbone.buffers()]:
                if t.dtype == torch.bfloat16:
                    t.data = t.data.half()
            model.backbone_dtype = torch.float16
        self.input_size = model_cfg.MODEL.IMAGE_SIZE
        self.batch_size, self.num_workers, self.device = batch_size, num_workers, device

        # the decoder returns (tokens, pose_output); token 0 is the pose token
        self._tokens = None
        self.model.decoder.register_forward_hook(self._store_tokens)

    def _store_tokens(self, _module, _inputs, output):
        self._tokens = output[0].detach()

    @torch.no_grad()
    def __call__(self, frame_dir, bbx_xyxy, K_fullimg):
        """frame_dir with %06d.jpg frames, bbx_xyxy (L, 4), K_fullimg (L, 3, 3) -> (L, 1024)"""
        t = self._transforms
        transform = t.Compose(
            [
                t.GetBBoxCenterScale(),
                t.TopdownAffine(input_size=self.input_size, use_udp=False),
                t.VisionTransformWrapper(ToTensor()),
            ]
        )
        dataset = _FrameDataset(
            Path(frame_dir), bbx_xyxy, K_fullimg, transform, self._load_image, self._prepare_batch
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        tokens = torch.zeros((len(dataset), self.TOKEN_DIM), dtype=torch.float32)
        i = 0
        with tqdm(total=len(dataset), desc="SAM 3D Body features", unit="frame") as pbar:
            for batch in loader:
                bs = batch["img"].shape[0]
                batch = self._to(batch, self.device)
                self.model._initialize_batch(batch)
                self.model.forward_step(batch, decoder_type="body")  # the hook stores the tokens
                tokens[i : i + bs] = self._tokens[:, 0].cpu()
                i += bs
                pbar.update(bs)
        if not torch.isfinite(tokens).all():
            raise RuntimeError(
                "Non-finite SAM 3D Body features, probably a float16 overflow: disable FP16 "
                "accumulation in run_video (exmr/pipeline.py) and please report this"
            )
        return tokens


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, bbx_xyxy, K_fullimg, transform, load_image, prepare_batch):
        assert len(bbx_xyxy) == len(K_fullimg)
        self.frame_dir, self.bbx_xyxy, self.K_fullimg = frame_dir, bbx_xyxy, K_fullimg
        self.transform, self.load_image, self.prepare_batch = transform, load_image, prepare_batch

    def __len__(self):
        return len(self.bbx_xyxy)

    def __getitem__(self, idx):
        img = self.load_image(self.frame_dir / f"{idx:06d}.jpg", backend="cv2", image_format="rgb")
        batch = self.prepare_batch(
            img,
            self.transform,
            self.bbx_xyxy[idx].reshape(1, 4),
            masks=None,
            masks_score=None,
            cam_int=self.K_fullimg[idx],
        )
        del batch["img_ori"]
        # prepare_batch makes a batch of persons per image; there is exactly one
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                batch[k] = v[0]
        return batch
