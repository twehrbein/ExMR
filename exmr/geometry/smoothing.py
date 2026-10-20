"""Smoothing of sequences along one dimension: Gaussian and moving average."""

import numpy as np
import torch
import torch.nn.functional as F


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
