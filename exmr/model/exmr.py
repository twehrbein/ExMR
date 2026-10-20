"""ExMR: initial whole-body estimate, then FootMR and HandMR refine the extremities."""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from exmr.geometry.camera import cliff_bbox_info, full_cam_transl, normalize_kp2d
from exmr.geometry.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from exmr.model import global_traj
from exmr.model.body_net import HAND_POSE, BodyNet
from exmr.model.decoder import Decoder
from exmr.model.refiners import FootRefiner, HandRefiner

# indices into the 22 SMPL-X body joints
KNEES, ANKLES = slice(4, 6), slice(7, 9)
SHOULDERS, ELBOWS, WRISTS = slice(16, 18), slice(18, 20), slice(20, 22)
# the same joints in the 63-dim body_pose (axis-angle of joints 1..21)
BODY_POSE_ANKLES, BODY_POSE_ELBOWS, BODY_POSE_WRISTS = slice(18, 24), slice(51, 57), slice(57, 63)
FOOT_KP2D = slice(15, 23)  # ankles + 6 foot keypoints of the COCO-23 layout


def _rot6d(rotmat):
    """(B, L, N, 3, 3) -> (B, L, N*6)"""
    return matrix_to_rotation_6d(rotmat).flatten(-2)


class ExMR(nn.Module):
    """Reconstructs SMPL-X body, hand and foot motion of one person from per-frame evidence.

    Inputs come from exmr.preprocessing: 2D keypoints, bounding boxes, camera intrinsics and
    angular velocity, SAM 3D Body and HaMeR image features. Outputs are SMPL-X parameters in the
    camera frame and in a gravity-aligned world frame.
    """

    def __init__(self, smplx_npz):
        super().__init__()
        self.body_net = BodyNet()
        self.foot_refiner = FootRefiner()
        self.hand_refiner = HandRefiner()
        self.decoder = Decoder(smplx_npz)
        # camera angular velocity normalization (6D), loaded with the checkpoint
        self.register_buffer("cam_angvel_mean", torch.zeros(6))
        self.register_buffer("cam_angvel_std", torch.ones(6))

    @classmethod
    def from_pretrained(cls, checkpoint, smplx_npz, device="cuda"):
        """Load the weights from a released checkpoint (.safetensors)."""
        model = cls(smplx_npz)
        state_dict = load_file(str(checkpoint))
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    @torch.inference_mode()
    def forward(
        self,
        kp2d,
        bbx_xys,
        K_fullimg,
        cam_angvel,
        body_feat,
        hand_feat,
        length=None,
        static_cam=False,
        postprocess=True,
    ):
        """
        Args:
            kp2d: (B, L, 23, 3) COCO-23 keypoints (body + feet) in pixels, [x, y, conf]
            bbx_xys: (B, L, 3) person bounding box (center x, center y, size) in pixels
            K_fullimg: (B, L, 3, 3) camera intrinsics
            cam_angvel: (B, L, 6) camera angular velocity, see exmr.preprocessing
            body_feat: (B, L, 1024) SAM 3D Body image features
            hand_feat: (B, L, 2, 1024) HaMeR image features (left, right)
            length: (B,) valid length of each sequence, default: all L frames
            static_cam: whether the camera is known to be static (better world trajectory)
            postprocess: apply static-joint post-processing to the world-frame outputs
        Returns:
            dict with
              smplx_incam: SMPL-X parameters in the camera frame (each (B, L, D))
              smplx_global: SMPL-X parameters in the gravity-aligned world frame (y up)
              static_conf_logits: (B, L, 6) static-contact logits of
                  [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
        """
        B, L = kp2d.shape[:2]
        if length is None:
            length = torch.full((B,), L, device=kp2d.device)
        obs = normalize_kp2d(kp2d, bbx_xys)
        cliffcam = cliff_bbox_info(bbx_xys, K_fullimg)
        cam_angvel_norm = (cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std

        # 1. initial whole-body estimate
        body_out = self.body_net(length, obs, cliffcam, cam_angvel_norm, body_feat, hand_feat)
        params = self.decoder.decode(body_out["x"])
        smplx_incam = {
            "body_pose": params["body_pose"],
            "betas": params["betas"],
            "global_orient": params["global_orient"],
            "left_hand_pose": params["left_hand_pose"],
            "right_hand_pose": params["right_hand_pose"],
            "transl": full_cam_transl(body_out["pred_cam"], bbx_xys, K_fullimg),
        }
        body_pose = smplx_incam["body_pose"]  # refined in place below

        # 2. FootMR: refine the global ankle rotations, then express them relative to the knee
        fk_rot = self.decoder.fk_rot(body_pose, smplx_incam["global_orient"])  # (B, L, 22, 3, 3)
        knee_rot = fk_rot[:, :, KNEES]
        foot_rot6d = torch.cat([_rot6d(knee_rot), _rot6d(fk_rot[:, :, ANKLES])], dim=-1)
        ankle_rot = self.foot_refiner(
            length, obs[:, :, FOOT_KP2D].clone(), cliffcam.clone(), foot_rot6d
        )
        ankle_rot = rotation_6d_to_matrix(ankle_rot.reshape(B, L, -1, 6))  # (B, L, 2, 3, 3)
        body_pose[:, :, BODY_POSE_ANKLES] = matrix_to_axis_angle(knee_rot.mT @ ankle_rot).flatten(
            -2
        )

        # 3. HandMR: refine global elbows and wrists and the finger rotations
        fk_rot = self.decoder.fk_rot(body_pose, smplx_incam["global_orient"])
        hand_pose_r6d = self.decoder.denormalize(body_out["x"][..., HAND_POSE], HAND_POSE)
        hand_rot6d = torch.cat(
            [_rot6d(fk_rot[:, :, ELBOWS]), _rot6d(fk_rot[:, :, WRISTS]), hand_pose_r6d], dim=-1
        )
        refined = self.hand_refiner(length, hand_feat, cliffcam.clone(), hand_rot6d)
        refined = rotation_6d_to_matrix(refined.reshape(B, L, -1, 6))  # (B, L, 34, 3, 3)
        elbow_rot, wrist_rot, finger_rot = refined[:, :, 0:2], refined[:, :, 2:4], refined[:, :, 4:]
        # global -> parent-relative: elbow relative to the shoulder, wrist to the refined elbow
        shoulder_rot = fk_rot[:, :, SHOULDERS]
        body_pose[:, :, BODY_POSE_ELBOWS] = matrix_to_axis_angle(
            shoulder_rot.mT @ elbow_rot
        ).flatten(-2)
        body_pose[:, :, BODY_POSE_WRISTS] = matrix_to_axis_angle(elbow_rot.mT @ wrist_rot).flatten(
            -2
        )
        finger_aa = matrix_to_axis_angle(finger_rot).flatten(-2)  # (B, L, 90)
        smplx_incam["left_hand_pose"] = finger_aa[:, :, :45]
        smplx_incam["right_hand_pose"] = finger_aa[:, :, 45:]

        # 4. world frame: same articulation, rolled-out root trajectory
        smplx_global = {
            "body_pose": body_pose,
            "betas": smplx_incam["betas"],
            "left_hand_pose": smplx_incam["left_hand_pose"],
            "right_hand_pose": smplx_incam["right_hand_pose"],
            **global_traj.rollout_global_params(
                params["global_orient_gv"],
                params["local_transl_vel"],
                smplx_incam["global_orient"],
                cam_angvel,
            ),
        }
        static_conf_logits = body_out["static_conf_logits"]
        if postprocess:
            if static_cam:
                smplx_global["transl"] = global_traj.static_cam_transl(
                    smplx_incam, smplx_global, static_conf_logits, self.decoder
                )
            else:
                smplx_global["transl"] = global_traj.static_joint_transl(
                    smplx_global, static_conf_logits, self.decoder
                )
            smplx_global["body_pose"] = global_traj.static_joint_ik(
                smplx_global, static_conf_logits, self.decoder
            )

        return {
            "smplx_incam": smplx_incam,
            "smplx_global": smplx_global,
            "static_conf_logits": static_conf_logits,
        }
