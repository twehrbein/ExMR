"""FootMR and HandMR: temporal refiners that predict residual extremity rotations."""

import torch.nn as nn

from exmr.model.body_net import KeypointEmbedder
from exmr.model.layers import Mlp, RoPEEncoder, condition_embedder, feature_embedder


class FootRefiner(nn.Module):
    """FootMR: refines both global ankle rotations from the 2D ankle and foot keypoints.

    Conditioned on the initial global knee and ankle rotations, it predicts the refined global
    ankle rotations as a residual on the initial ones (6D, left then right).
    """

    def __init__(
        self,
        num_2d_joints=8,  # 2 x (ankle, big toe, small toe, heel)
        num_rot_condition=4,  # global knee and ankle rotations, left and right
        output_dim=2 * 6,  # global ankle rotations, left and right
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        mlp_ratio=2.0,
        max_len=120,
        dropout=0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.kp2d_embedder = KeypointEmbedder(num_2d_joints, latent_dim, dropout)
        self.cliffcam_embedder = condition_embedder(3, latent_dim, dropout)
        self.rot_6d_embedder = condition_embedder(6 * num_rot_condition, latent_dim, dropout)
        self.encoder = RoPEEncoder(latent_dim, num_layers, num_heads, mlp_ratio, dropout, max_len)
        self.final_layer = Mlp(latent_dim, out_features=output_dim)

    def forward(self, length, kp2d, cliffcam, rotations_6d, frame_valid=None):
        """
        Args:
            length: (B,)
            kp2d: (B, L, 8, 3) box-normalized foot keypoints
            cliffcam: (B, L, 3)
            rotations_6d: (B, L, 24) global [knees, ankles] 6D rotations
        Returns:
            (B, L, 12) refined global ankle rotations (6D)
        """
        x = self.kp2d_embedder(kp2d)
        x = x + self.cliffcam_embedder(cliffcam)
        x = x + self.rot_6d_embedder(rotations_6d)
        x, _ = self.encoder(x, length, frame_valid)
        return self.final_layer(x) + rotations_6d[:, :, -self.output_dim :]


class HandRefiner(nn.Module):
    """HandMR: refines elbows, wrists and fingers from HaMeR hand features.

    Conditioned on the initial global elbow and wrist rotations and the parent-relative finger
    rotations, it predicts refined ones as a residual (6D): global elbows (2), global wrists (2),
    then the 2 x 15 finger joints.
    """

    def __init__(
        self,
        hand_feat_dim=1024,  # HaMeR token, per hand
        num_rot_condition=2 * (2 + 15),  # elbow, wrist and fingers, left and right
        output_dim=2 * 17 * 6,
        latent_dim=512,
        num_layers=6,
        num_heads=4,
        mlp_ratio=2.0,
        max_len=120,
        dropout=0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        # each hand gets half of the channels
        self.hand_feat_embedder = feature_embedder(hand_feat_dim, latent_dim // 2)
        self.cliffcam_embedder = condition_embedder(3, latent_dim, dropout)
        self.rot_6d_embedder = condition_embedder(6 * num_rot_condition, latent_dim, dropout)
        self.encoder = RoPEEncoder(latent_dim, num_layers, num_heads, mlp_ratio, dropout, max_len)
        self.final_layer = Mlp(latent_dim, out_features=output_dim)

    def forward(self, length, hand_feat, cliffcam, rotations_6d, frame_valid=None):
        """
        Args:
            length: (B,)
            hand_feat: (B, L, 2, 1024) HaMeR image features (left, right)
            cliffcam: (B, L, 3)
            rotations_6d: (B, L, 204) [global elbows, global wrists, fingers] 6D rotations
        Returns:
            (B, L, 204) refined rotations in the same layout (6D)
        """
        B, L = rotations_6d.shape[:2]
        x = self.hand_feat_embedder(hand_feat).reshape(B, L, -1)
        x = x + self.cliffcam_embedder(cliffcam)
        x = x + self.rot_6d_embedder(rotations_6d)
        x, _ = self.encoder(x, length, frame_valid)
        return self.final_layer(x) + rotations_6d[:, :, -self.output_dim :]
