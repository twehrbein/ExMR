"""Result videos: the SMPL-X mesh over the input video (camera view), in the world frame with a
ground plane (world view), and/or both side by side."""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from exmr.geometry.kinematics import apply_T_on_points, transform_mat
from exmr.preprocessing.video import read_frame
from exmr.vis.renderer import Renderer, ground_params_from_points, world_view_camera
from exmr.vis.video_writer import VideoWriter

MESH_COLOR = [86.0 / 255.0, 180.0 / 255.0, 233.0 / 255.0]  # light blue
# the videos: camera view, world view, both side by side
VIEWS = ("incam", "global", "incam_global")


@torch.no_grad()
def render_videos(frame_dir, verts_incam, verts_global, faces, J_regressor, K, fps, out_paths):
    """Render the camera view, the world view and/or both side by side in one pass.

    Args:
        frame_dir: the video's frames %06d.jpg
        verts_incam: (L, V, 3) mesh vertices in the camera frame, with intrinsics K (3, 3)
        verts_global: (L, V, 3) mesh vertices in the world frame
        out_paths: {view: video path} for the views "incam" (camera view), "global" (world view)
            and "incam_global" (both side by side); only the views needed are rendered
    """
    if unknown := set(out_paths) - set(VIEWS):
        raise ValueError(f"unknown views {sorted(unknown)}, expected some of {VIEWS}")
    frame_dir = Path(frame_dir)
    first = read_frame(frame_dir / "000000.jpg")
    # even sizes, for the videos' YUV 4:2:0 format (which stores colors per 2x2 pixels): an odd
    # last row or column is left out
    height, width = (s // 2 * 2 for s in first.shape[:2])
    incam_renderer = world_renderer = None
    if out_paths.keys() & {"incam", "incam_global"}:
        incam_renderer = Renderer(width, height, K=K, faces=faces)
        frame_paths = [frame_dir / f"{i:06d}.jpg" for i in range(len(verts_incam))]
        frames = _decoded_frames(frame_paths, height, width)
    if out_paths.keys() & {"global", "incam_global"}:
        world_renderer, ground, verts_world = _world_view(
            verts_global, faces, J_regressor, width, height
        )

    # write to temporary names, so that an interrupted run leaves no complete-looking videos
    tmp_paths = {
        view: Path(p).with_name(Path(p).name + ".tmp.mp4") for view, p in out_paths.items()
    }
    with ExitStack() as stack:  # closes every writer, also on errors
        writers = {
            view: stack.enter_context(
                VideoWriter(path, 2 * width if view == "incam_global" else width, height, fps=fps)
            )
            for view, path in tmp_paths.items()
        }
        images = {}
        for i in tqdm(range(len(verts_incam)), desc="Rendering"):
            if incam_renderer is not None:
                images["incam"] = incam_renderer.render_mesh(
                    verts_incam[i], next(frames), MESH_COLOR
                )
            if world_renderer is not None:
                images["global"] = world_renderer.render_mesh(
                    verts_world[i], ground, MESH_COLOR, specular_color=(1.0, 1.0, 1.0)
                )
            if "incam_global" in writers:
                images["incam_global"] = torch.cat([images["incam"], images["global"]], dim=1)
            for view, writer in writers.items():
                writer.write_frame(images[view])
    for view, path in out_paths.items():
        os.replace(tmp_paths[view], path)


def _decoded_frames(frame_paths, height, width, ahead=8):
    """The frames, cropped to height x width, on the GPU; decoded in background threads, up to
    `ahead` frames ahead of their use."""
    with ThreadPoolExecutor(4) as pool:
        decoded = deque(pool.submit(read_frame, p) for p in frame_paths[:ahead])
        for i in range(len(frame_paths)):
            frame = decoded.popleft().result()[:height, :width]
            if i + ahead < len(frame_paths):
                decoded.append(pool.submit(read_frame, frame_paths[i + ahead]))
            yield torch.from_numpy(frame).cuda()


def _world_view(vertices, faces, J_regressor, width, height):
    """The world view of the world-frame vertices (L, V, 3): a static camera looks from the side
    at the motion, which starts at the origin, facing +z. Returns its renderer, the ground image
    and the vertices in its camera frame.

    As the camera is static, the ground is rendered only once, and each frame renders only the
    mesh over it. The ground never hides the mesh, as the camera and all vertices are above it.
    """
    vertices = _move_to_start_point_face_z(vertices, J_regressor)
    joints = torch.einsum("jv,lvc->ljc", J_regressor, vertices)  # (L, J, 3)
    scale, cx, cz = ground_params_from_points(joints[:, 0], vertices)

    R, T = (x.to(vertices) for x in world_view_camera(vertices.cpu()))
    # a 24 mm lens (full-frame equivalent) for the image size
    focal_length = (width**2 + height**2) ** 0.5 / (24**2 + 36**2) ** 0.5 * 24
    K = torch.tensor([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]])
    # the light is at the camera
    renderer = Renderer(width, height, K=K, faces=faces, light_location=(0.0, 0.0, 0.0))
    ground = renderer.render_ground(R, T, scale * 1.5, cx, cz)
    return renderer, ground, vertices @ R.mT + T


def _move_to_start_point_face_z(verts, J_regressor):
    """Translate verts (L, V, 3) so that the first frame's root is at the origin (xz) and the
    lowest vertex on the ground (y=0), then rotate about y so that the first frame faces +z."""
    verts = verts.clone()
    offset = (J_regressor @ verts[0])[0]  # first-frame root joint
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset

    joints = (J_regressor @ verts[0])[None]  # (1, J, 3)
    t = joints[:, 0].clone()
    t[:, 1] = 0
    # left direction from the hips and shoulders, projected to the ground
    left = (joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]) + (
        joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]
    )
    x_dir = torch.zeros_like(t)
    x_dir[:, [0, 2]] = F.normalize(left, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[:, 1] = 1
    R = torch.stack([x_dir, y_dir, torch.cross(x_dir, y_dir, dim=-1)], dim=-1)  # facing -> world
    if left.pow(2).sum() < 1e-4:  # facing direction undefined: do not rotate
        R[:] = torch.eye(3).to(R)
    T_inv = transform_mat(R.mT, -torch.einsum("bij,bi->bj", R, t))
    return apply_T_on_points(verts, T_inv)
