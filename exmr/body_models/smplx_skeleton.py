"""The part of SMPL-X that ExMR's kinematics need: the 22 body joints as a function of shape."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

NUM_BODY_JOINTS = 22  # pelvis + 21 body joints (SMPL-X order), no jaw/eyes/fingers


class SmplxSkeleton(nn.Module):
    """Rest-pose body joint positions of SMPL-X for given shape coefficients.

    Built from the official SMPLX_NEUTRAL.npz (not distributed with ExMR, see docs/INSTALL.md).
    """

    def __init__(self, smplx_npz, num_betas=10):
        super().__init__()
        smplx_npz = Path(smplx_npz)
        if not smplx_npz.exists():
            raise FileNotFoundError(
                f"SMPL-X model not found at {smplx_npz}; see docs/INSTALL.md for how to get it"
            )
        data = np.load(smplx_npz, allow_pickle=True)

        shapedirs = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas])).float()
        v_template = torch.tensor(np.asarray(data["v_template"])).float()
        J_regressor = torch.tensor(np.asarray(data["J_regressor"])).float()
        parents = torch.tensor(np.asarray(data["kintree_table"][0])).long()
        parents[0] = -1

        # joint positions are linear in betas: J(betas) = J_template + J_shapedirs @ betas.
        # Kept for all 55 SMPL-X joints and sliced after the product, as in the original
        # implementation: a smaller GEMM may round differently.
        J_template = J_regressor @ v_template  # (55, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", J_regressor, shapedirs)  # (55, 3, B)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)
        self.register_buffer("parents", parents[:NUM_BODY_JOINTS], False)

    def forward(self, betas):
        """betas (..., num_betas) -> rest-pose body joints (..., 22, 3)"""
        joints = self.J_template + torch.einsum("...k, jck -> ...jc", betas, self.J_shapedirs)
        return joints[..., :NUM_BODY_JOINTS, :]
