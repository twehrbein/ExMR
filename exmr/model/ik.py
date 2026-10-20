"""Cyclic coordinate descent (CCD) inverse kinematics on a single kinematic chain."""

import torch

from exmr.geometry import kinematics
from exmr.geometry.quaternion import qbetween, qmul, qrot, qslerp
from exmr.geometry.rotation_conversions import matrix_to_quaternion


class CCDIK:
    """Rotates the joints of one chain so that its end effectors reach target positions.

    Args:
        local_mat: (*, J, 4, 4) local transforms of the full skeleton
        parent: parent index of each skeleton joint (-1 for the root)
        target_ind: indices *within the chain* of the joints with position targets
        target_pos: (*, len(target_ind), 3) target positions
        kinematic_chain: skeleton joint indices of the chain, root first; the root stays fixed
        max_iter: number of CCD sweeps
    """

    def __init__(self, local_mat, parent, target_ind, target_pos, kinematic_chain, max_iter=2):
        global_mat = kinematics.forward_kinematics(local_mat, parent)

        # chain-only local transforms, with the chain root fixed at its global transform
        local_mat = local_mat.clone()[..., kinematic_chain, :, :]
        local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]

        self.parent = [i - 1 for i in range(len(kinematic_chain))]
        self.local_mat = local_mat
        self.global_mat = kinematics.forward_kinematics(local_mat, self.parent)  # (*, J, 4, 4)
        self.target_ind = target_ind
        self.target_pos = target_pos  # (*, O, 3)
        self.num_joints = local_mat.shape[-3]
        self.max_iter = max_iter

    def solve(self):
        """Returns the chain's (*, J_chain, 4, 4) local transforms."""
        for _ in range(self.max_iter):
            for i in range(1, self.num_joints - 1):  # the chain root is not optimized
                self._optimize_joint(i)
        return self.local_mat

    def _optimize_joint(self, i):
        pos = kinematics.get_position(self.global_mat)[..., i, :]  # (*, 3)
        quat = matrix_to_quaternion(kinematics.get_rotation(self.global_mat)[..., i, :, :])
        x_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        x_vec[..., 0] = 1.0
        x_vec_sum = torch.zeros_like(x_vec)
        y_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        y_vec[..., 1] = 1.0
        y_vec_sum = torch.zeros_like(y_vec)

        count = 0
        for target_i, j in enumerate(self.target_ind):
            if i >= j:  # only joints above the end effector move it
                continue
            end_pos = kinematics.get_position(self.global_mat)[..., j, :]  # (*, 3)
            target_pos = self.target_pos[..., target_i, :]  # (*, 3)
            # rotate joint i so that end_pos moves toward target_pos, damped by the chain depth
            solved_quat = qslerp(
                quat,
                qmul(qbetween(end_pos - pos, target_pos - pos), quat),
                (i + 1) / self.num_joints,
            )
            x_vec_sum += qrot(solved_quat, x_vec)
            y_vec_sum += qrot(solved_quat, y_vec)
            count += 1

        if count > 0:
            # average the solutions of all targets via their x and y axes
            x_vec_avg = kinematics.normalize(x_vec_sum / count)
            y_vec_avg = kinematics.normalize(y_vec_sum / count)
            z_vec_avg = torch.cross(x_vec_avg, y_vec_avg, dim=-1)
            solved_rot = torch.stack([x_vec_avg, y_vec_avg, z_vec_avg], dim=-1)  # columns

            parent_rot = kinematics.get_rotation(self.global_mat)[..., self.parent[i], :, :]
            self.local_mat[..., i, :-1, :-1] = kinematics.get_mat_BtoA(parent_rot, solved_rot)
            self.global_mat = kinematics.forward_kinematics(self.local_mat, self.parent)
