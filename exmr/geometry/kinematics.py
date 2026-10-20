"""Homogeneous transforms and forward kinematics on (..., 4, 4) or (..., 3, 3) matrices."""

import torch
import torch.nn.functional as F


def get_rotation(mat):
    """(..., 4, 4) -> (..., 3, 3)"""
    return mat[..., :-1, :-1]


def get_position(mat):
    """(..., 4, 4) -> (..., 3)"""
    return mat[..., :-1, 3]


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def normalized_matrix(mat):
    """Normalize the columns of the rotation block of (..., 4, 4) or (..., 3, 3) matrices."""
    rot_mat = mat[..., :-1, :-1] if mat.shape[-1] == 4 else mat
    rot_mat_norm = rot_mat / (rot_mat.norm(2, dim=-2, keepdim=True) + 1e-9)
    if mat.shape[-1] != 4:
        return rot_mat_norm
    norm_mat = torch.zeros_like(mat)
    norm_mat[..., :-1, :-1] = rot_mat_norm
    norm_mat[..., :-1, -1] = mat[..., :-1, -1]
    norm_mat[..., -1, -1] = 1.0
    return norm_mat


def get_TRS(rot_mat, pos):
    """Rotations (..., 3, 3) and positions (..., 3) -> transforms (..., 4, 4)."""
    mat = torch.eye(4, device=pos.device).repeat(pos.shape[:-1] + (1, 1))
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    return normalized_matrix(mat)


def get_mat_BfromA(matA, matBtoA):
    """World matrix of B, given world matrix A and matrix B relative to A."""
    return normalized_matrix(torch.matmul(matA, matBtoA))


def get_mat_BtoA(matA, matB):
    """Matrix B expressed in the frame of A, given world matrices A and B."""
    matA_inv = normalized_matrix(torch.inverse(matA))
    return normalized_matrix(torch.matmul(matA_inv, matB))


def forward_kinematics(mat, parent):
    """Local joint transforms (..., J, D, D) -> global ones, for parents[i] < i (-1 = root)."""
    # unbind + one stack instead of per-joint slicing/concatenation: same values, far cheaper
    # (and with a much lighter backward graph)
    local_mats = mat.unbind(dim=-3)
    chain = []
    for i in range(mat.shape[-3]):
        if parent[i] != -1:
            chain.append(get_mat_BfromA(chain[parent[i]], local_mats[i]))
        else:
            chain.append(local_mats[i])
    return torch.stack(chain, dim=-3)


def transform_mat(R, t):
    """Rotations (B, 3, 3) and translations (B, 3) -> transforms (B, 4, 4)."""
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)


def apply_T_on_points(points, T):
    """points (..., N, 3), T (..., 4, 4) -> (..., N, 3)"""
    return torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
