"""Initial whole-body estimator: per-frame evidence -> temporal transformer -> body+hand params."""

import torch
import torch.nn as nn

from exmr.model.layers import Mlp, RoPEEncoder, condition_embedder, feature_embedder

# Layout of the body network's (normalized) output vector, see exmr.model.decoder.
BODY_POSE = slice(0, 126)  # 21 body joints, 6D rotations relative to their parent
BETAS = slice(126, 136)  # SMPL-X shape
GLOBAL_ORIENT = slice(136, 142)  # root rotation in the camera frame, 6D
GLOBAL_ORIENT_GV = slice(142, 148)  # root rotation in the gravity-view frame, 6D
LOCAL_TRANSL_VEL = slice(148, 151)  # root velocity in the root's frame
HAND_POSE = slice(151, 331)  # 2 x 15 finger joints (left, right), 6D relative rotations
OUTPUT_DIM = 331


class KeypointEmbedder(nn.Module):
    """2D keypoints (B, L, J, 3) -> (B, L, latent_dim); joints with conf <= 0.5 use a learned
    token instead of their coordinates."""

    def __init__(self, num_joints, latent_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(2, 32)
        self.missing_token = nn.Parameter(torch.randn(num_joints, 32))
        self.mlp = Mlp(num_joints * 32, latent_dim * 2, latent_dim, drop=dropout)

    def forward(self, kp2d):
        B, L, J, _ = kp2d.shape
        visible = kp2d[..., [2]] > 0.5  # (B, L, J, 1)
        # zero invisible coords before the linear (NaN-safe), then swap in the learned token
        xy = torch.where(visible, kp2d[..., :2], kp2d.new_zeros(()))
        f = torch.where(visible, self.linear(xy), self.missing_token)  # (B, L, J, 32)
        return self.mlp(f.view(B, L, -1))


class BodyNet(nn.Module):
    """Predicts normalized SMPL-X parameters (see OUTPUT layout above), a weak-perspective crop
    camera and static-joint logits for every frame."""

    # mean/std of the crop camera [s, tx, ty]
    PRED_CAM_MEAN = (1.0606, -0.0027, 0.2702)
    PRED_CAM_STD = (0.1784, 0.0956, 0.0764)

    def __init__(
        self,
        num_2d_joints=23,
        body_feat_dim=1024,  # SAM 3D Body token
        hand_feat_dim=1024,  # HaMeR token, per hand
        cam_angvel_dim=6,
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        max_len=120,
        static_conf_dim=6,
        dropout=0.1,
    ):
        super().__init__()
        self.kp2d_embedder = KeypointEmbedder(num_2d_joints, latent_dim, dropout)
        self.cliffcam_embedder = condition_embedder(3, latent_dim, dropout)
        self.cam_angvel_embedder = condition_embedder(cam_angvel_dim, latent_dim, dropout)
        self.body_feat_embedder = feature_embedder(body_feat_dim, latent_dim)
        # each hand gets half of the channels
        self.hand_feat_embedder = feature_embedder(hand_feat_dim, latent_dim // 2)

        self.encoder = RoPEEncoder(latent_dim, num_layers, num_heads, mlp_ratio, dropout, max_len)

        self.final_layer = Mlp(latent_dim, out_features=OUTPUT_DIM)
        self.pred_cam_head = Mlp(latent_dim, out_features=3)
        self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)
        self.register_buffer("pred_cam_mean", torch.tensor(self.PRED_CAM_MEAN), False)
        self.register_buffer("pred_cam_std", torch.tensor(self.PRED_CAM_STD), False)

    def forward(self, length, kp2d, cliffcam, cam_angvel, body_feat, hand_feat, frame_valid=None):
        """
        Args:
            length: (B,) valid length of each sequence
            kp2d: (B, L, J, 3) box-normalized 2D keypoints [x, y, conf]
            cliffcam: (B, L, 3) bounding box info, see exmr.geometry.camera.cliff_bbox_info
            cam_angvel: (B, L, 6) normalized camera angular velocity (6D)
            body_feat: (B, L, 1024) SAM 3D Body image features
            hand_feat: (B, L, 2, 1024) HaMeR image features (left, right)
            frame_valid: (B, L) bool or None, False at frames to ignore
        Returns:
            dict with x (B, L, 331) normalized params, pred_cam (B, L, 3),
            static_conf_logits (B, L, 6) for [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
        """
        B, L = kp2d.shape[:2]
        x = self.kp2d_embedder(kp2d)
        x = x + self.cliffcam_embedder(cliffcam)
        x = x + self.hand_feat_embedder(hand_feat).reshape(B, L, -1)
        x = x + self.cam_angvel_embedder(cam_angvel)
        x = x + self.body_feat_embedder(body_feat)

        x, keep = self.encoder(x, length, frame_valid)

        out = self.final_layer(x)  # (B, L, 331)
        # one shape per sequence: average betas over the valid frames
        n_keep = keep.sum(1, keepdim=True).clamp_min(1)  # (B, 1)
        betas = (out[..., BETAS] * keep[..., None]).sum(1) / n_keep  # (B, 10)
        betas = betas[:, None].expand(-1, L, -1)
        out = torch.cat([out[..., : BETAS.start], betas, out[..., BETAS.stop :]], dim=-1)

        pred_cam = self.pred_cam_head(x) * self.pred_cam_std + self.pred_cam_mean
        torch.clamp_min_(pred_cam[..., 0], 0.25)  # keep the scale positive

        return {
            "x": out,
            "pred_cam": pred_cam,
            "static_conf_logits": self.static_conf_head(x),
        }
