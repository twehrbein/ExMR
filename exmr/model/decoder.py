"""Normalized network outputs -> SMPL-X parameters, and forward kinematics of the body joints."""

import torch
import torch.nn as nn

from exmr.body_models.smplx_skeleton import SmplxSkeleton
from exmr.geometry import kinematics
from exmr.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from exmr.model.body_net import (
    BETAS,
    BODY_POSE,
    GLOBAL_ORIENT,
    GLOBAL_ORIENT_GV,
    HAND_POSE,
    LOCAL_TRANSL_VEL,
    OUTPUT_DIM,
)


def r6d_to_aa(r6d):
    """(..., N*6) 6D rotations -> (..., N*3) axis-angle"""
    rotmat = rotation_6d_to_matrix(r6d.unflatten(-1, (-1, 6)))
    return matrix_to_axis_angle(rotmat).flatten(-2)


class Decoder(nn.Module):
    """Un-normalizes the body network output and runs SMPL-X forward kinematics (22 joints)."""

    def __init__(self, smplx_npz):
        super().__init__()
        # output normalization statistics (from training data), loaded with the checkpoint
        self.register_buffer("mean", torch.zeros(OUTPUT_DIM))
        self.register_buffer("std", torch.ones(OUTPUT_DIM))
        self.skeleton = SmplxSkeleton(smplx_npz)
        self.parents = self.skeleton.parents.tolist()

    def denormalize(self, x_norm, channels=slice(None)):
        return x_norm * self.std[channels] + self.mean[channels]

    def decode(self, x_norm):
        """(B, L, 331) normalized output -> dict of SMPL-X parameters (axis-angle)."""
        x = self.denormalize(x_norm)
        return {
            "body_pose": r6d_to_aa(x[..., BODY_POSE]),  # (B, L, 63)
            "betas": x[..., BETAS],  # (B, L, 10)
            "global_orient": r6d_to_aa(x[..., GLOBAL_ORIENT]),  # (B, L, 3), camera frame
            "global_orient_gv": r6d_to_aa(x[..., GLOBAL_ORIENT_GV]),  # (B, L, 3)
            "local_transl_vel": x[..., LOCAL_TRANSL_VEL],  # (B, L, 3)
            "left_hand_pose": r6d_to_aa(x[..., HAND_POSE][..., :90]),  # (B, L, 45)
            "right_hand_pose": r6d_to_aa(x[..., HAND_POSE][..., 90:]),  # (B, L, 45)
        }

    def fk_rot(self, body_pose, global_orient):
        """Global rotations of the 22 body joints, (B, L, 22, 3, 3).

        Cheaper than fk() when no positions are needed; equals the rotation block of fk().
        """
        B, L = body_pose.shape[:2]
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        # pre-normalized like get_TRS does in fk(), so both give the same rotations
        rotmat = kinematics.normalized_matrix(axis_angle_to_matrix(aa))
        return kinematics.forward_kinematics(rotmat, self.parents)

    def fk(self, body_pose, betas, global_orient, transl, get_intermediate=False, **_):
        """Global positions of the 22 body joints, (B, L, 22, 3).

        With get_intermediate, also returns the local and global (B, L, 22, 4, 4) transforms.
        Extra SMPL-X parameters (hand poses) are accepted and ignored.
        """
        B, L = body_pose.shape[:2]
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.skeleton(betas)  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.skeleton.parents]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        local_skeleton[..., 0, :] += transl

        local_mat = kinematics.get_TRS(rotmat, local_skeleton)  # (B, L, 22, 4, 4)
        global_mat = kinematics.forward_kinematics(local_mat, self.parents)
        joints = kinematics.get_position(global_mat)  # (B, L, 22, 3)
        if get_intermediate:
            return joints, local_mat, global_mat
        return joints
