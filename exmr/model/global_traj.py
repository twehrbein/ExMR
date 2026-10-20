"""World-frame trajectory: gravity-view rollout and static-joint post-processing (from GVHMR)."""

import torch
import torch.nn.functional as F

from exmr.geometry import kinematics
from exmr.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from exmr.geometry.smoothing import gaussian_smooth
from exmr.model.ik import CCDIK

# joints with a static-contact prediction: [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
STATIC_JOINT_IDS = [7, 10, 8, 11, 20, 21]


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

    Returns the post-processed (B, L, 3) translation, with the lowest joint on the ground (y=0).
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

    # put the sequence on the ground (lowest joint over all frames at y=0)
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)
    post_w_transl[..., 1] -= ground_y
    return post_w_transl


def static_cam_transl(smplx_incam, smplx_global, static_conf_logits, decoder):
    """Static camera: keep the world trajectory consistent with the camera-frame one and make
    joints predicted static stay put. Returns the post-processed (B, L, 3) translation."""
    B, L = smplx_incam["transl"].shape[:2]
    assert B == 1
    static_conf_logits = static_conf_logits.clone()[:, :-1]  # (B, L-1, J)

    pred_w_j3d = decoder.fk(**smplx_global)  # (B, L, 22, 3)
    # smooth the camera-frame translation, as it could be noisy
    smplx_incam = {**smplx_incam}
    smplx_incam["transl"] = gaussian_smooth(smplx_incam["transl"], sigma=5, dim=-2)
    pred_c_j3d = decoder.fk(**smplx_incam)  # (B, L, 22, 3)

    # static camera-to-world transform from the first frame
    R_gv = axis_angle_to_matrix(smplx_global["global_orient"][:, 0])  # (B, 3, 3)
    R_c = axis_angle_to_matrix(smplx_incam["global_orient"][:, 0])  # (B, 3, 3)
    R_c2w = R_gv @ R_c.mT  # (B, 3, 3)
    t_c2w = pred_w_j3d[:, 0, 0] - torch.einsum("bij,bj->bi", R_c2w, pred_c_j3d[:, 0, 0])  # (B, 3)
    T_c2w = kinematics.transform_mat(R_c2w, t_c2w)  # (B, 4, 4)
    pred_c_j3d_in_w = kinematics.apply_T_on_points(pred_c_j3d, T_c2w[:, None])

    # 1. pull the world root toward the camera-frame root where they deviate by > 25 cm
    post_w_transl = smplx_global["transl"].clone()  # (B, L, 3)
    post_w_j3d = pred_w_j3d.clone()  # (B, L, 22, 3)
    cp_thr = torch.tensor([0.25, 0.25, 0.25]).to(post_w_j3d)
    for i in range(1, L):
        cp_diff = post_w_j3d[:, i, 0] - pred_c_j3d_in_w[:, i, 0]  # (B, 3)
        cp_diff = cp_diff * ~((cp_diff > -cp_thr) * (cp_diff < cp_thr))
        cp_diff[..., 1] = 0  # y is set by the ground below
        cp_diff = torch.clamp(cp_diff, -0.02, 0.02)
        post_w_transl[:, i:] -= cp_diff
        post_w_j3d[:, i:] -= (cp_diff)[:, None, None]

    # 2. make joints predicted static stay put (average displacement of confident ones)
    pred_j3d_static = post_w_j3d[:, :, STATIC_JOINT_IDS]  # (B, L, J, 3)
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)
    static_label = static_conf_logits.sigmoid() > 0.8  # (B, L-1, J)
    static_label_sum = torch.clamp_min(static_label.sum(-1, keepdim=True), 1)  # (B, L-1, 1)
    pred_disp = (pred_j_disp * static_label[..., None]).sum(-2) / static_label_sum  # (B, L-1, 3)
    pred_disp[:, :, 1] = 0  # do not modify y
    correction = pred_disp.cumsum(dim=1)  # every frame gets all preceding corrections
    post_w_transl[:, 1:] -= correction
    post_w_j3d[:, 1:] -= correction[:, :, None]

    # put the sequence on the ground (lowest joint over all frames at y=0)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)
    post_w_transl[..., 1] -= ground_y
    return post_w_transl


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
