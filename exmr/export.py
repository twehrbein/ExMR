"""Export of the world-frame motion as an AMASS SMPL-X file, e.g. for the SMPL-X Blender add-on
(see docs/OUTPUTS.md)."""

import numpy as np
import torch

from exmr.body_models.smplx_skeleton import SmplxSkeleton
from exmr.geometry.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

# ExMR's world frame is y-up, AMASS's z-up: +90 degrees about x
Y_UP_TO_Z_UP = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)


def save_amass(smplx_params, fps, path, smplx_npz):
    """Save world-frame SMPL-X parameters (results["smplx_global"], each (L, D), y up) to path as
    an AMASS SMPL-X .npz (z up, float64).

    Args:
        fps: frame rate of the motion; stored rounded to an integer
        smplx_npz: the SMPLX_NEUTRAL.npz of the parameters, for the rest-pose pelvis position
    """
    params = {k: v.detach().cpu().double() for k, v in smplx_params.items()}
    R = Y_UP_TO_Z_UP
    root_orient = matrix_to_axis_angle(R @ axis_angle_to_matrix(params["global_orient"]))
    # SMPL-X rotates the body about its rest-pose pelvis J0, not about the origin: the rotated
    # world moves the pelvis from J0 + transl to R (J0 + transl)
    betas = params["betas"][0]  # the same in all frames
    pelvis = SmplxSkeleton(smplx_npz)(betas.float())[0].double()
    trans = (pelvis + params["transl"]) @ R.T - pelvis

    zeros = torch.zeros(len(trans), 3, dtype=torch.float64)
    parts = {
        "root_orient": root_orient,  # (L, 3)
        "pose_body": params["body_pose"],  # (L, 63)
        "pose_jaw": zeros,  # (L, 3), not estimated
        "pose_eye": torch.cat([zeros, zeros], dim=-1),  # (L, 6), not estimated
        "pose_hand": torch.cat([params["left_hand_pose"], params["right_hand_pose"]], dim=-1),
    }
    np.savez(
        path,
        poses=torch.cat(list(parts.values()), dim=-1).numpy(),  # (L, 165): the 55 SMPL-X joints
        **{k: v.numpy() for k, v in parts.items()},
        trans=trans.numpy(),
        betas=betas.numpy(),
        num_betas=len(betas),
        gender="neutral",
        surface_model_type="smplx",
        # rounded: the SMPL-X Blender add-on cuts the rate to an integer (29.97 -> 29) and rejects
        # rates below its target framerate (default 30)
        mocap_frame_rate=float(round(fps)),
    )
