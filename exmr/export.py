"""Export of the world-frame motion as an AMASS SMPL-X file, e.g. for the SMPL-X Blender add-on or
for retargeting to humanoid robots with GMR, and as a BVH skeleton animation (see
docs/OUTPUTS.md)."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from exmr.geometry.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

# ExMR's world frame is y-up, AMASS's z-up: +90 degrees about x
Y_UP_TO_Z_UP = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)
# AMASS SMPL-X files have 16 betas, and loaders such as GMR's build the body model with as many:
# ExMR's 10 betas are padded with zeros (the same body shape)
AMASS_NUM_BETAS = 16

# BVH: the rotation order of each joint (intrinsic, as its channels list them; XYZ if not listed),
# chosen so that the middle angle stays away from +-90 degrees (gimbal lock) in typical motion
BVH_ROTATION_ORDERS = {
    "pelvis": "YZX",  # turns freely about the vertical
    "left_shoulder": "ZXY", "right_shoulder": "ZXY",
    "left_elbow": "YXZ", "right_elbow": "YXZ",  # bend about the vertical in the rest pose
    "left_ankle": "XZY", "right_ankle": "XZY",
}  # fmt: skip
# BVH: the end site of each leaf joint, a point of SmplxModel.joints or the chin
BVH_END_SITES = {
    "jaw": "chin",
    "left_eye_smplhf": "left_eye",
    "right_eye_smplhf": "right_eye",
    **{f"{side}_foot": f"{side}_big_toe" for side in ("left", "right")},
    **{
        f"{side}_{finger}3": f"{side}_{finger}"
        for side in ("left", "right")
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    },
}
CHIN_VERTEX = 8750  # a vertex of the SMPL-X mesh, on the chin's midline


def save_amass(smplx_params, fps, path, smplx):
    """Save world-frame SMPL-X parameters (results["smplx_global"], each (L, D), y up) to path as
    an AMASS SMPL-X .npz (z up, float64).

    Args:
        fps: frame rate of the motion; stored rounded to an integer
        smplx: a SmplxModel of the parameters, for the rest-pose pelvis position
    """
    params = {k: v.detach().cpu().double() for k, v in smplx_params.items()}
    R = Y_UP_TO_Z_UP
    root_orient = matrix_to_axis_angle(R @ axis_angle_to_matrix(params["global_orient"]))
    # SMPL-X rotates the body about its rest-pose pelvis J0, not about the origin: the rotated
    # world moves the pelvis from J0 + transl to R (J0 + transl)
    betas = params["betas"][0]  # the same in all frames
    with torch.no_grad():
        pelvis = smplx.joints(**_rest_pose(params))[0, 0].double()
    trans = (pelvis + params["transl"]) @ R.T - pelvis

    zeros = torch.zeros(len(trans), 3, dtype=torch.float64)
    parts = {
        "root_orient": root_orient,  # (L, 3)
        "pose_body": params["body_pose"],  # (L, 63)
        "pose_jaw": zeros,  # (L, 3), not estimated
        "pose_eye": torch.cat([zeros, zeros], dim=-1),  # (L, 6), not estimated
        "pose_hand": torch.cat([params["left_hand_pose"], params["right_hand_pose"]], dim=-1),
    }
    np.savez(
        path,
        poses=torch.cat(list(parts.values()), dim=-1).numpy(),  # (L, 165): the 55 SMPL-X joints
        **{k: v.numpy() for k, v in parts.items()},
        trans=trans.numpy(),
        betas=np.pad(betas.numpy(), (0, AMASS_NUM_BETAS - len(betas))),
        num_betas=AMASS_NUM_BETAS,
        gender="neutral",
        surface_model_type="smplx",
        # rounded: the SMPL-X Blender add-on cuts the rate to an integer (29.97 -> 29) and rejects
        # rates below its target framerate (default 30)
        mocap_frame_rate=float(round(fps)),
    )


def save_bvh(smplx_params, fps, path, smplx):
    """Save world-frame SMPL-X parameters (results["smplx_global"], each (L, D), y up) to path as
    a BVH animation of the 55 SMPL-X joints, in centimeters: the skeleton is the SMPL-X rest pose
    of the person's shape, and the joints are at SmplxModel.joints(...)[:, :55] in every frame.

    Args:
        fps: frame rate of the motion
        smplx: a SmplxModel of the parameters
    """
    params = {k: v.detach().cpu().float() for k, v in smplx_params.items()}
    L = len(params["transl"])
    names, parents = smplx.joint_names, smplx.model.parents.tolist()
    rest_params = _rest_pose(params)
    with torch.no_grad():
        rest = smplx.joints(**rest_params)[0].double().numpy() * 100  # (76, 3)
        chin = smplx(**rest_params)[0, CHIN_VERTEX].double().numpy() * 100
    points = dict(zip(names, rest)) | {"chin": chin}

    children = [[c for c in range(55) if parents[c] == j] for j in range(55)]
    lines, joint_order = ["HIERARCHY"], []

    def add_joint(j, depth):
        tab = "\t" * depth
        rotation = " ".join(f"{axis}rotation" for axis in BVH_ROTATION_ORDERS.get(names[j], "XYZ"))
        if j == 0:  # the root's channels give its position, not an offset to it
            offset, channels = np.zeros(3), f"CHANNELS 6 Xposition Yposition Zposition {rotation}"
        else:
            offset, channels = rest[j] - rest[parents[j]], f"CHANNELS 3 {rotation}"
        lines.append(f"{tab}{'JOINT' if j else 'ROOT'} {names[j]}")
        lines.extend([tab + "{", f"{tab}\tOFFSET {_xyz(offset)}", f"{tab}\t{channels}"])
        joint_order.append(j)
        for c in children[j]:
            add_joint(c, depth + 1)
        if not children[j]:
            end = points[BVH_END_SITES[names[j]]] - rest[j]
            lines.extend([f"{tab}\tEnd Site", tab + "\t{", f"{tab}\t\tOFFSET {_xyz(end)}"])
            lines.append(tab + "\t}")
        lines.append(tab + "}")

    add_joint(0, 0)

    zeros = torch.zeros(L, 3)
    rotvecs = torch.cat(
        [params["global_orient"], params["body_pose"], zeros, zeros, zeros]  # jaw, eyes: zero
        + [params["left_hand_pose"], params["right_hand_pose"]],
        dim=-1,
    ).reshape(L, 55, 3)
    euler = np.stack(
        [
            Rotation.from_rotvec(rotvecs[:, j].double().numpy()).as_euler(
                BVH_ROTATION_ORDERS.get(names[j], "XYZ"), degrees=True
            )
            for j in range(55)
        ],
        axis=1,
    )  # (L, 55, 3)
    # SMPL-X rotates the body about the rest-pose pelvis: the pelvis is at J0 + transl
    pelvis = rest[0] + 100 * params["transl"].double().numpy()
    motion = np.concatenate([pelvis, _continuous_euler(euler)[:, joint_order].reshape(L, -1)], 1)
    with open(path, "w") as f:
        f.write("\n".join(lines) + f"\nMOTION\nFrames: {L}\nFrame Time: {1 / fps:.8f}\n")
        np.savetxt(f, motion, fmt="%.4f")


def _rest_pose(params):
    """The shape of params in the zero pose at the origin: float32 SMPL-X parameters, each (1, D)"""
    rest = {k: torch.zeros_like(v[:1], dtype=torch.float32) for k, v in params.items()}
    rest["betas"] = params["betas"][:1].float()  # the same in all frames
    return rest


def _continuous_euler(euler):
    """Euler angles (L, J, 3) in degrees -> the same rotations with each frame's angles as close
    as possible to the previous frame's: of the two angle triples of a rotation, (a, b, c) and
    (a + 180, 180 - b, c + 180), the closer one, and each angle shifted by a multiple of 360. So
    the angles do not jump, e.g. from 179 to -179, and software that interpolates between frames
    turns the joints the short way."""
    out = euler.copy()
    joints = np.arange(euler.shape[1])
    for t in range(1, len(euler)):
        a, b, c = np.moveaxis(euler[t], -1, 0)
        candidates = np.stack([euler[t], np.stack([a + 180, 180 - b, c + 180], axis=-1)])
        candidates += 360 * np.round((out[t - 1] - candidates) / 360)
        best = np.abs(candidates - out[t - 1]).sum(axis=-1).argmin(axis=0)  # (J,)
        out[t] = candidates[best, joints]
    return out


def _xyz(v):
    return " ".join(f"{x:.4f}" for x in v)
