"""Mesh rendering with nvdiffrast: overlay on video frames, and a world view with a ground plane."""

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

# Shading with one point light: color * (AMBIENT + DIFFUSE * max(0, cos(normal, to light)))
# + SPECULAR * specular color (Phong specular with shininess 0, which is constant)
AMBIENT, DIFFUSE, SPECULAR = 0.5, 0.3, 0.2


class Renderer:
    """Renders meshes into a width x height image with intrinsics K (pixels), seen by a camera
    at the origin with OpenCV axes (x right, y down, z forward). Images are (H, W, 3) uint8
    CUDA tensors."""

    def __init__(self, width, height, K, faces, light_location=(0.0, 0.0, -10.0), device="cuda"):
        self.width, self.height = width, height
        self.faces = faces.to(device, torch.int32).contiguous()  # (F, 3)
        self.light = torch.tensor(light_location, dtype=torch.float32, device=device)
        self.projection = _projection(K, width, height).to(device)
        self.context = dr.RasterizeCudaContext(device=device)

    def render_mesh(self, vertices, background, color, specular_color=None):
        """The mesh (V, 3) in the camera frame with color (3,) over background.
        specular_color: of the material, default: color."""
        rast = self._rasterize(vertices, self.faces)
        # vertex normals: sums of the adjacent faces' normals, weighted by their areas
        faces = self.faces.long()
        corners = vertices[faces]  # (F, 3, 3)
        face_normals = torch.linalg.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        normals = torch.zeros_like(vertices).index_add_(
            0, faces.flatten(), face_normals.repeat_interleave(3, dim=0)
        )
        attributes = torch.cat([vertices, normals], dim=-1)[None].contiguous()
        pixels, _ = dr.interpolate(attributes, rast, self.faces)  # (1, H, W, 6)
        points, normals = pixels[0, ..., :3], F.normalize(pixels[0, ..., 3:], dim=-1, eps=1e-6)
        to_light = F.normalize(self.light - points, dim=-1, eps=1e-6)
        cos = (normals * to_light).sum(dim=-1, keepdim=True)

        color = torch.tensor(color, dtype=torch.float32, device=vertices.device)
        specular = color if specular_color is None else color.new_tensor(specular_color)
        rgb = (AMBIENT + DIFFUSE * cos.clamp(min=0)) * color + SPECULAR * specular
        return torch.where(rast[0, ..., 3:] > 0, (rgb * 255).to(torch.uint8), background)

    def render_ground(self, R, T, length, center_x, center_z):
        """A checkerboard ground (side length, center in the xz-plane of the world frame) seen
        with the world-to-camera rotation R (3, 3) and translation T (3,), white where there is
        no ground. It gets no diffuse light (its colors stay flat), and a white specular color."""
        verts, faces, colors = (
            torch.from_numpy(a).to(R.device)
            for a in _checkerboard(length=float(length), c1=center_x, c2=center_z)
        )
        rast = self._rasterize(verts @ R.mT + T, faces)
        rgb, _ = dr.interpolate(colors[None].contiguous(), rast, faces)
        rgb = ((AMBIENT * rgb[0] + SPECULAR) * 255).to(torch.uint8)
        return torch.where(rast[0, ..., 3:] > 0, rgb, torch.full_like(rgb, 255))

    def _rasterize(self, vertices, faces):
        """nvdiffrast's (1, H, W, 4) rasterizer output; channel 3 is the face index + 1 (0 where
        no face is visible)."""
        clip = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1) @ self.projection.T
        rast, _ = dr.rasterize(
            self.context, clip[None].contiguous(), faces, (self.height, self.width)
        )
        return rast


def world_view_camera(verts, beta=3.5, cam_height_degree=20, target_center_height=0.9, vec_rot=-45):
    """A static camera looking at the motion of verts (L, V, 3) from the side and above.

    Returns its world-to-camera rotation (3, 3) and translation (3,), with OpenCV camera axes
    (x right, y down, z forward) as in Renderer.
    """
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to the ground plane
    target_center = targets.mean(0)
    target_scale = torch.norm(targets - target_center, dim=-1).max()

    vec_rad = vec_rot / 180 * np.pi
    vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
    vec = vec / torch.norm(vec)

    # the larger beta, the further away the camera
    target_scale = max(target_scale, 1.0) * beta
    position = target_center + vec * target_scale
    position[1] = target_scale * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    target = target_center.clone()
    target[1] = target_center_height
    # the camera axes in the world frame (y up): forward to the target, right, down
    z = F.normalize(target - position, dim=0)
    x = F.normalize(torch.linalg.cross(z, torch.tensor([0.0, 1.0, 0.0])), dim=0)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])
    return R, -R @ position


def ground_params_from_points(root_points, vert_points):
    """Ground (xz-plane) size and center from root (L, 3) and vertex (L, V, 3) trajectories."""
    cx, _, cz = (root_points.max(0)[0] + root_points.min(0)[0]) / 2.0
    vert_extent = vert_points.reshape(-1, 3).max(0)[0] - vert_points.reshape(-1, 3).min(0)[0]
    return float(vert_extent[[0, 2]].max()), float(cx), float(cz)


def _checkerboard(length=12.0, tile_width=0.5, c1=0.0, c2=0.0):
    """Checkerboard ground in the xz-plane: vertices, faces (int32) and vertex colors (RGB)."""
    color0 = np.array([0.8, 0.9, 0.9])
    color1 = np.array([0.6, 0.7, 0.7])
    num = max(2, int(length / tile_width))
    radius = float(num * tile_width) / 2.0
    vertices, faces, colors = [], [], []
    for i in range(num):
        for j in range(num):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            verts = np.stack([us, np.zeros(4), vs], axis=-1)
            verts[:, 0] += c1
            verts[:, 2] += c2
            vertices.append(verts)
            faces.append(np.array([[0, 1, 3], [1, 2, 3]]) + 4 * (i * num + j))
            colors.append(np.repeat([color0 if (i + j) % 2 == 0 else color1], 4, axis=0))
    return (
        np.concatenate(vertices).astype(np.float32),
        np.concatenate(faces).astype(np.int32),
        np.concatenate(colors).astype(np.float32),
    )


def _projection(K, width, height, near=0.1, far=1000.0):
    """(4, 4) projection from the OpenCV camera frame with intrinsics K (3, 3) to clip space,
    with image row 0 as the first row of nvdiffrast's output."""
    fx, fy, cx, cy = (float(K[i, j]) for i, j in [(0, 0), (1, 1), (0, 2), (1, 2)])
    return torch.tensor(
        [
            [2 * fx / width, 0, 2 * cx / width - 1, 0],
            [0, 2 * fy / height, 2 * cy / height - 1, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
