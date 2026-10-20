# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# Quaternion utilities (real part first) used by the CCD IK solver, from GVHMR's
# hmr4d/utils/geo/quaternion.py, which builds on (see THIRD_PARTY_NOTICES.md):
# - qmul, qrot: QuaterNet (https://github.com/facebookresearch/QuaterNet), common/quaternion.py,
#   Copyright (c) 2018-present, Facebook, Inc., licensed under CC BY-NC 4.0
#   (https://creativecommons.org/licenses/by-nc/4.0/, licenses/QuaterNet_LICENSE.txt)
# - qinv, qnormalize, qpow, qslerp, qbetween: HumanML3D
#   (https://github.com/EricGuo5513/HumanML3D), common/quaternion.py, Copyright (c) 2022
#   Chuan Guo, MIT License (licenses/HumanML3D_LICENSE.txt)
# Modified by GVHMR (qnormalize: clamped norm; qmul: reshape; qpow: broadcasting; qslerp: no
# expand; qbetween: explicit cross dim, handling of opposite vectors) and for ExMR (subset of the
# functions, qpow for scalar powers only, docstrings).

import torch


def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qnormalize(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)


def qmul(q, r):
    """Quaternion product q * r for (*, 4) tensors."""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """Rotate vectors v (*, 3) by quaternions q (*, 4)."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qpow(q0, t, dtype=torch.float):
    """Quaternions q0 raised to the scalar power t."""
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., :1])

    # if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    mask = mask.float()
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / torch.sin(theta0)

    q = torch.zeros(q0.shape, device=q0.device)
    theta = t * theta0

    q[..., :1] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta)

    return q.to(dtype)


def qslerp(q0, q1, t):
    """Spherical linear interpolation from q0 to q1 at the scalar fraction t."""
    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)

    return qmul(q_, q0)


def qbetween(v0, v1):
    """Quaternion that rotates vectors v0 (*, 3) onto v1 (*, 3)."""
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1, dim=-1)

    w = torch.sqrt((v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)) + (
        v0 * v1
    ).sum(dim=-1, keepdim=True)
    y_vec = torch.zeros_like(v)
    y_vec[..., 1] = 1.0
    # v0 = -v1 gives v = 0 and w = 0: rotate by 180 degrees around y instead
    mask = v.norm(dim=-1) == 0
    # v0 = v1 gives v = 0 and w = 2: identity, nothing to do
    mask2 = w.sum(dim=-1).abs() <= 1e-4
    mask = torch.logical_and(mask, mask2)
    v[mask] = y_vec[mask]

    return qnormalize(torch.cat([w, v], dim=-1))
