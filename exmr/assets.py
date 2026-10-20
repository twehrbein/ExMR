"""Locations of the model files: one checkpoint directory with a fixed layout (docs/INSTALL.md)."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = Path(os.environ.get("EXMR_CHECKPOINT_DIR", "checkpoints"))

# SHA256 of the RTMW-x FP16 ONNX model ExMR was evaluated with
RTMW_SHA256 = "957ddb5ed34576593a918b3e514003ab9cc85d037309b9a23d9e0209530e98ed"


@dataclass
class Assets:
    root: Path = DEFAULT_CHECKPOINT_DIR

    @property
    def exmr(self):
        return self.root / "exmr" / "exmr.safetensors"

    @property
    def smplx(self):
        return self.root / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"

    def yolo(self, detector):
        return self.root / "yolo" / f"{detector}.pt"

    @property
    def sapiens2(self):
        return self.root / "sapiens2" / "sapiens2_1b_pose.safetensors"

    @property
    def rtmw(self):
        return self.root / "rtmw" / "rtmw-x_384x288_fp16.onnx"

    @property
    def sam_3d_body(self):
        return self.root / "sam-3d-body-dinov3"

    @property
    def hamer(self):
        return self.root / "hamer"

    @property
    def dpvo(self):
        return self.root / "dpvo" / "dpvo.pth"

    def check(self, pose_backend, detector, use_dpvo):
        """Raise a single error listing every missing file needed for these settings."""
        required = {
            "ExMR weights": self.exmr,
            "SMPL-X body model": self.smplx,
            "YOLO11 person detector": self.yolo(detector),
            "SAM 3D Body checkpoint": self.sam_3d_body / "model.ckpt",
            "SAM 3D Body MHR model": self.sam_3d_body / "assets" / "mhr_model.pt",
            "HaMeR checkpoint": self.hamer / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
            "MANO right hand model": self.hamer / "data" / "mano" / "MANO_RIGHT.pkl",
        }
        if pose_backend == "sapiens2":
            required["Sapiens2 1B pose weights"] = self.sapiens2
        elif pose_backend == "rtmw":
            required["RTMW-x ONNX model"] = self.rtmw
        else:
            raise ValueError(f"Unknown pose backend: {pose_backend}")
        if use_dpvo:
            required["DPVO weights"] = self.dpvo
        missing = [f"  - {name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model files (see docs/INSTALL.md for how to get them):\n"
                + "\n".join(missing)
            )
