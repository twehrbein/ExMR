"""SMPL-X mesh for ExMR's parameters (axis-angle hand poses, flat hand mean, neutral face)."""

from pathlib import Path

import torch
import torch.nn as nn


class SmplxModel(nn.Module):
    """Vertices (10475) and joints of SMPL-X for ExMR's output parameters.

    Args:
        smplx_npz: the official SMPLX_NEUTRAL.npz (not distributed with ExMR, see docs/INSTALL.md)
    """

    def __init__(self, smplx_npz):
        super().__init__()
        import smplx

        if not Path(smplx_npz).exists():
            raise FileNotFoundError(
                f"SMPL-X model not found at {smplx_npz}; see docs/INSTALL.md for how to get it"
            )
        self.model = smplx.SMPLX(
            str(smplx_npz),
            gender="neutral",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
            **{
                f"create_{name}": False
                for name in (
                    "body_pose",
                    "betas",
                    "global_orient",
                    "transl",
                    "left_hand_pose",
                    "right_hand_pose",
                    "expression",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                )
            },  # fmt: skip
        )
        # the model data comes from smplx_npz: keep it out of state_dict(), so that a module
        # holding a SmplxModel (ExMR) loads its checkpoint without it
        for module in self.model.modules():
            for name, buffer in list(module.named_buffers(recurse=False)):
                module.register_buffer(name, buffer, persistent=False)
        self.faces = torch.from_numpy(self.model.faces.astype("int64"))  # (20908, 3)

    @property
    def J_regressor(self):
        """(55, 10475) joint regressor; the first 22 joints match SMPL's."""
        return self.model.J_regressor

    def forward(self, body_pose, betas, global_orient, transl, left_hand_pose, right_hand_pose):
        """SMPL-X parameters, each (L, D) -> vertices (L, 10475, 3)"""
        L = body_pose.shape[0]
        zeros = body_pose.new_zeros((L, 3))
        return self.model(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=zeros,
            leye_pose=zeros,
            reye_pose=zeros,
            expression=body_pose.new_zeros((L, self.model.num_expression_coeffs)),
        ).vertices
