"""Smoothing of sequences along one dimension: Gaussian and moving average."""

import numpy as np
import torch
import torch.nn.functional as F

from exmr.geometry.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


def gaussian_kernel1d(sigma, radius):
    """Normalized 1D Gaussian kernel of size 2 * radius + 1 (float64).

    Same values as scipy.ndimage's (private) _gaussian_kernel1d with order=0.
    """
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / (sigma * sigma) * x**2)
    return phi_x / phi_x.sum()


def gaussian_smooth(x, sigma=3, dim=-1):
    """Gaussian smoothing along dim, with replicate padding at the sequence ends."""
    kernel = gaussian_kernel1d(sigma=sigma, radius=int(4 * sigma + 0.5))
    return _smooth(x, torch.from_numpy(kernel).float(), dim)


def moving_average_smooth(x, window_size=5, dim=-1):
    """Moving average along dim, with replicate padding at the sequence ends."""
    return _smooth(x, torch.ones(window_size).float() / window_size, dim)


def gaussian_smooth_rotations(aa, sigma, dim=-2):
    """Gaussian smoothing of axis-angle rotations (..., J*3) along the time dim.

    Axis-angle and quaternions are discontinuous, so the rotation matrices are smoothed and
    projected back onto SO(3), which gives the chordal mean of the window.
    """
    mats = axis_angle_to_matrix(aa.unflatten(-1, (-1, 3)))  # (..., J, 3, 3)
    mats = gaussian_smooth(mats.flatten(-3), sigma, dim).unflatten(-1, mats.shape[-3:])
    u, _, vh = torch.linalg.svd(mats.double())
    # the nearest rotation, not reflection: flip the least significant direction if det = -1
    flip = torch.ones_like(u[..., 0, :])
    flip[..., -1] = torch.linalg.det(u @ vh).sign()
    rot = (u * flip.unsqueeze(-2)) @ vh
    return matrix_to_axis_angle(rot.to(aa.dtype)).flatten(-2)


def _smooth(x, kernel, dim):
    kernel = kernel[None, None].to(x)  # (1, 1, K)
    rad = kernel.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = x.reshape(-1, 1, x.shape[-1])  # (N, 1, F)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., F)
    return x.transpose(-1, dim)
