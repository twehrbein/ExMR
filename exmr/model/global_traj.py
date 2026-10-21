"""World-frame trajectory: gravity-view rollout and static-joint post-processing (from GVHMR),
and putting the motion on the ground, optionally snapped to a flat ground (as in PRIMAL)."""

import torch
import torch.nn.functional as F

from exmr.geometry import kinematics
from exmr.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from exmr.geometry.smoothing import gaussian_smooth, moving_average_smooth
from exmr.model.ik import CCDIK

# joints with a static-contact prediction: [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
STATIC_JOINT_IDS = [7, 10, 8, 11, 20, 21]
# snap_to_ground: some body part touches the ground at least this often (seconds), about the hang
# time of a jump (as in PRIMAL, Zhang et al., arXiv 2503.17544)
CONTACT_INTERVAL = 0.5
# static_cam_transl: the world root may deviate this far (m) from the camera-frame root before
# the slow part of the deviation (Gaussian smoothing, sigma in seconds) is removed
CAMERA_DEAD_BAND = 0.25
CAMERA_DRIFT_SIGMA = 2.0


def rollout_global_params(global_orient_gv, local_transl_vel, global_orient_c, cam_angvel):
    """World-frame root orientation and translation, starting at the origin in frame 0.

    The gravity-view (GV) frame of every frame is aligned to the first one by the camera's yaw
    rotation (from the camera angular velocity), then the root velocity is integrated.

    Args:
        global_orient_gv: (B, L, 3) root orientation in the per-frame GV frame
        local_transl_vel: (B, L, 3) root velocity in the root's frame
        global_orient_c: (B, L, 3) root orientation in the camera frame
        cam_angvel: (B, L, 6) camera rotation from frame t to t+1 (6D), R @ R_w2c^t = R_w2c^t+1
    Returns:
        dict with global_orient (B, L, 3) and transl (B, L, 3), y-axis up
    """

    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = as_identity(rotation_6d_to_matrix(cam_angvel))  # (B, L, 3, 3)

    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # camera view direction in the GV frame (its x-axis is estimated, so != 0)
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)
    # ... and in the next frame
    view_axis_gv_next = (R_c2gv @ R_t_to_tp1.mT)[..., 2]

    # yaw between consecutive GV frames: angle between the view directions projected to xz
    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle
    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3, 3)

    # accumulate: GV frame of t -> GV frame of 0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = as_identity(torch.stack(R_t_to_0, dim=1))  # (B, L, 3, 3)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # integrate the root velocity (in the root frame) starting at the origin
    transl_vel = torch.einsum(
        "...lij,...lj->...li", axis_angle_to_matrix(global_orient), local_transl_vel
    )
    transl = torch.cumsum(
        torch.cat([torch.zeros_like(transl_vel[..., :1, :]), transl_vel[..., :-1, :]], dim=-2),
        dim=-2,
    )

    # GV is y-down: rotate by 180 degrees around z to make y point up
    R_flip = axis_angle_to_matrix(torch.tensor([0, 0, torch.pi]).to(global_orient.device))
    global_orient = matrix_to_axis_angle(R_flip @ axis_angle_to_matrix(global_orient))
    transl = torch.einsum("...ij,...j->...i", R_flip, transl)
    return {"global_orient": global_orient, "transl": transl}


def static_joint_transl(smplx_global, static_conf_logits, decoder):
    """Moving camera: correct the world translation so that joints predicted static stay put.

    Returns the post-processed (B, L, 3) translation.
    """
    pred_w_j3d = decoder.fk(**smplx_global)  # (B, L, 22, 3)
    pred_j3d_static = pred_w_j3d.clone()[:, :, STATIC_JOINT_IDS]  # (B, L, J, 3)
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_conf_logits = static_conf_logits[:, :-1].clone()
    static_label = static_conf_logits > 0  # (B, L-1, J)
    # non-static joints get a large negative logit, i.e. zero weight after the softmax
    static_conf_logits = static_conf_logits.float() - (~static_label * 1e6)
    is_static = static_label.sum(dim=-1) > 0  # (B, L-1), at least one joint static
    # per-frame displacement of the static joints, weighted by their confidence
    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = (pred_disp * is_static[..., None, None]).sum(-2)  # (B, L-1, 3)

    # subtract that displacement from the root translation, re-integrate, smooth x and z
    pred_w_transl = smplx_global["transl"].clone()  # (B, L, 3)
    pred_w_disp_new = (pred_w_transl[:, 1:] - pred_w_transl[:, :-1]) - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)
    return post_w_transl


def static_cam_transl(smplx_incam, smplx_global, static_conf_logits, decoder, fps):
    """Static camera: the moving-camera correction (static_joint_transl), then the slow drift of
    the world root away from the camera-frame root, which a static camera places in the world, is
    removed where it exceeds CAMERA_DEAD_BAND. Returns the post-processed (B, L, 3) translation.
    """
    transl = static_joint_transl(smplx_global, static_conf_logits, decoder)
    world_root = decoder.fk(**{**smplx_global, "transl": transl})[:, :, 0]  # (B, L, 3)
    # the camera-frame root, smoothed as it could be noisy
    smplx_incam = {**smplx_incam, "transl": gaussian_smooth(smplx_incam["transl"], 5, dim=-2)}
    cam_root = decoder.fk(**smplx_incam)[:, :, 0]  # (B, L, 3)

    # into the world with the camera-to-world rotation of the first frame, where both roots meet
    R_c2w = (
        axis_angle_to_matrix(smplx_global["global_orient"][:, 0])
        @ axis_angle_to_matrix(smplx_incam["global_orient"][:, 0]).mT
    )  # (B, 3, 3)
    cam_root_in_w = torch.einsum("bij,blj->bli", R_c2w, cam_root - cam_root[:, :1])
    cam_root_in_w = cam_root_in_w + world_root[:, :1]

    # the slow part of the deviation, beyond the dead band (per axis)
    drift = gaussian_smooth(world_root - cam_root_in_w, CAMERA_DRIFT_SIGMA * fps, dim=-2)
    drift = drift.sign() * (drift.abs() - CAMERA_DEAD_BAND).clamp(min=0)
    return transl - drift


def lowest_vertex_height(smplx_global, body_model, chunk_size=256):
    """Height (y) of the lowest SMPL-X mesh vertex in every frame, (B, L).

    The mesh is computed in chunks of frames: all of a long video's would take gigabytes.
    """
    params = {k: v.flatten(0, 1) for k, v in smplx_global.items()}  # (B * L, D)
    lowest = [
        body_model(**{k: v[i : i + chunk_size] for k, v in params.items()})[..., 1].amin(dim=-1)
        for i in range(0, len(params["transl"]), chunk_size)
    ]
    return torch.cat(lowest).view(smplx_global["transl"].shape[:2])


def put_on_ground(smplx_global, body_model):
    """Move the motion vertically so that the lowest mesh vertex over all frames is on the
    ground (y=0). Returns the (B, L, 3) translation."""
    ground = lowest_vertex_height(smplx_global, body_model).amin(dim=-1)  # (B,)
    transl = smplx_global["transl"].clone()
    transl[..., 1] -= ground[:, None]
    return transl


def snap_to_ground(smplx_global, body_model, fps, interval=CONTACT_INTERVAL):
    """Flat ground and nothing to stand on: move the body vertically so that it touches the
    ground (y=0) at least every `interval` seconds, which removes vertical drift.

    The ground height at a frame is the lowest mesh vertex within the window of about `interval`
    seconds around it (a sliding minimum), smoothed over the same window. Vertical motion within
    the window is kept, e.g. jumps with a shorter flight; longer flights are pulled down.
    Returns the (B, L, 3) translation.
    """
    lowest = lowest_vertex_height(smplx_global, body_model)  # (B, L)
    L = lowest.shape[1]
    k = min(2 * int(interval * fps / 2) + 1, L - 1 + L % 2)  # odd window size, at most L
    ground = -F.max_pool1d(-lowest[:, None], k, stride=1)  # (B, 1, L - k + 1)
    # at the sequence ends, the windows are shifted inward instead of cut off
    ground = F.pad(ground, (k // 2, k // 2), mode="replicate")[:, 0]  # (B, L)
    ground = moving_average_smooth(ground, window_size=k, dim=-1)
    transl = smplx_global["transl"].clone()
    transl[..., 1] -= ground
    return transl


def static_joint_ik(smplx_global, static_conf_logits, decoder):
    """Inverse kinematics so that ankles, feet and wrists predicted static stay put.

    Returns the post-processed (B, L, 63) world-frame body pose (axis-angle).
    """
    static_conf = static_conf_logits.sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, _ = decoder.fk(**smplx_global, get_intermediate=True)

    # targets: x[t] = c[t-1] * x[t-1] + (1 - c[t-1]) * fk[t], i.e. a static joint stays where it
    # was in the previous frame, weighted by its static confidence c. Solved as a parallel prefix
    # scan of the affine recursion x[t] = a[t] * x[t-1] + b[t] (a[0] = 0 keeps the first frame).
    post_target_j3d = post_w_j3d.clone()
    a = torch.cat(
        [torch.zeros_like(static_conf[:, :1, :, None]), static_conf[:, :-1, :, None]], dim=1
    )
    b = post_w_j3d[:, :, STATIC_JOINT_IDS] * (1 - a)
    offset = 1
    while offset < b.shape[1]:
        b = torch.cat([b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]], dim=1)
        a = torch.cat([a[:, :offset], a[:, offset:] * a[:, :-offset]], dim=1)
        offset *= 2
    post_target_j3d[:, :, STATIC_JOINT_IDS] = b

    parents = decoder.parents
    chains = [  # (chain, targets: joint ids, their indices within the chain)
        ([0, 1, 4, 7, 10], [7, 10], [3, 4]),  # left leg: ankle and foot
        ([0, 2, 5, 8, 11], [8, 11], [3, 4]),  # right leg
        ([9, 13, 16, 18, 20], [20], [4]),  # left arm: wrist
        ([9, 14, 17, 19, 21], [21], [4]),  # right arm
    ]
    for chain, joint_ids, target_ind in chains:
        solver = CCDIK(local_mat, parents, target_ind, post_target_j3d[:, :, joint_ids], chain)
        chain_rotmat = kinematics.get_rotation(solver.solve())
        local_mat = local_mat.clone()
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]

    body_pose = matrix_to_axis_angle(kinematics.get_rotation(local_mat[:, :, 1:]))
    return body_pose.flatten(2)  # (B, L, 63)
