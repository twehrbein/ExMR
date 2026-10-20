"""Camera conventions: bounding boxes as (cx, cy, size) "xys", full-image intrinsics K."""

import torch

from exmr.geometry.rotation_conversions import matrix_to_rotation_6d


def estimate_K(img_w, img_h):
    """Default intrinsics for an unknown camera: principal point at the image center, focal
    length = image diagonal (53 degrees diagonal field of view)."""
    focal_length = (img_w**2 + img_h**2) ** 0.5
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2, aspect_ratio=192 / 256):
    """Corner boxes (L, 4) [x1, y1, x2, y2] -> square boxes (L, 3) [cx, cy, size].

    The box is first grown to the given width/height aspect ratio, then the longer side is
    enlarged by base_enlarge.
    """
    corners = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)  # (L, 2, 2)
    min_x, min_y = corners.amin(dim=-2).unbind(-1)
    max_x, max_y = corners.amax(dim=-2).unbind(-1)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    h = max_y - min_y
    w = max_x - min_x
    h = torch.maximum(h, w / aspect_ratio)
    w = torch.maximum(w, h * aspect_ratio)
    return torch.stack([center_x, center_y, torch.max(h, w) * base_enlarge], dim=-1)


def cam_angvel_from_rotations(R_w2c):
    """Camera angular velocity, the model's camera-motion input.

    Args:
        R_w2c: (L, 3, 3) world-to-camera rotations of the video frames
    Returns:
        (L, 6) relative rotation from each frame to the next (6D), R @ R_w2c[t] = R_w2c[t+1];
        the last frame repeats the previous one
    """
    cam_angvel = matrix_to_rotation_6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))  # (L-1, 6)
    return torch.cat([cam_angvel, cam_angvel[-1:]], dim=0).float()


def cliff_bbox_info(bbx_xys, K_fullimg):
    """Bounding box relative to the principal point, in units of the focal length (CLIFF/BEDLAM).

    Args:
        bbx_xys: (..., 3) box center and size in pixels
        K_fullimg: (..., 3, 3) intrinsics of the full image
    Returns:
        (..., 3) [(cx - icx) / f, (cy - icy) / f, size / f]
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]

    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    return bbox_info / fl


def full_cam_transl(pred_cam, bbx_xys, K_fullimg):
    """Weak-perspective crop camera [s, tx, ty] -> translation in the full-image camera.

    Args:
        pred_cam: (..., 3)
        bbx_xys: (..., 3)
        K_fullimg: (..., 3, 3)
    Returns:
        (..., 3)
    """
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def normalize_kp2d(kp2d, bbx_xys):
    """Pixel keypoints -> box-normalized coordinates in [-1, 1]; points outside get conf 0.

    Args:
        kp2d: (..., J, 3) [x, y, conf] in pixels
        bbx_xys: (..., 3)
    Returns:
        (..., J, 3) [x, y, conf]
    """
    xy = kp2d[..., :2]
    conf = kp2d[..., 2]
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]

    xy_max = center + scale / 2
    xy_min = center - scale / 2
    outside = (
        (xy[..., 0] < xy_min[..., None, 0])
        + (xy[..., 0] > xy_max[..., None, 0])
        + (xy[..., 1] < xy_min[..., None, 1])
        + (xy[..., 1] > xy_max[..., None, 1])
    )
    conf = conf * ~outside
    normalized_xy = 2 * (xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
    return torch.cat([normalized_xy, conf[..., None]], dim=-1)
