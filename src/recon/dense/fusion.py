"""TSDF fusion of the fitted mono depth maps with Open3D: a gap-filling surface where OpenMVS found none.

Uses the tensor VoxelBlockGrid: this Open3D build's legacy ScalableTSDFVolume extracts an empty mesh."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


BLOCK_RESOLUTION = 16  # voxels per block side
BLOCK_CAPACITY = 100_000  # blocks the grid starts with (it grows)


@dataclass(slots=True)
class FusionOptions:
    enabled: bool = True
    tau: float | None = None  # geometry tolerance; None: as mesh_post (0.5% of the diagonal, 2 cm in metres)
    voxel_tau: float = 0.8  # voxel size, in tau
    trunc_tau: float = 4.8  # truncation distance, in tau
    max_depth_frac: float = 0.7  # depths beyond this fraction of the diagonal are not integrated
    min_inlier_ratio: float = 0.7  # only views whose fit to the SfM points is at least this good are fused
    max_residual: float = 0.05  # ... and whose median relative residual is at most this
    min_weight: float = 2.0  # a surface voxel needs this many views
    point_stride: int = 4  # every n-th pixel of each view goes into the mono point cloud


@dataclass(slots=True)
class FusionView:
    depth: np.ndarray  # (h, w) float32, 0 = no depth
    K: np.ndarray  # (3, 3) intrinsics at the depth map's size
    cam_from_world: np.ndarray  # (4, 4)


def backproject(view: FusionView, stride: int, max_depth: float) -> np.ndarray:
    h, w = view.depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = view.depth[v, u]
    ok = (z > 0) & (z < max_depth)
    x = (u[ok] - view.K[0, 2]) / view.K[0, 0] * z[ok]
    y = (v[ok] - view.K[1, 2]) / view.K[1, 1] * z[ok]
    cam = np.stack([x, y, z[ok]], axis=1)
    world_from_cam = np.linalg.inv(view.cam_from_world)
    return cam @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]


def fuse(
    views: list[FusionView], diag: float, tau: float, options: FusionOptions | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate `views` into a TSDF (voxel `voxel_tau`·tau, truncation `trunc_tau`·tau) on the CPU, extract
    its surface by marching cubes (voxels seen by at least `min_weight` views) and subsample every view's depth
    into a point cloud (voxel `voxel_tau`·tau). Returns the vertices (V, 3), faces (F, 3), points (N, 3) and a
    report."""
    options = options or FusionOptions()
    max_depth = options.max_depth_frac * diag
    voxel = options.voxel_tau * tau
    # the block grid needs a colour attribute to extract a mesh; it carries nothing here
    grid = o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight", "color"),
        (o3d.core.float32, o3d.core.float32, o3d.core.float32),
        ((1), (1), (3)),
        voxel,
        BLOCK_RESOLUTION,
        BLOCK_CAPACITY,
        o3d.core.Device("CPU:0"),
    )
    multiplier = options.trunc_tau * tau / voxel
    cloud, skipped = [], 0
    for view in views:
        h, w = view.depth.shape
        depth_np = np.where(view.depth < max_depth, view.depth, 0.0)
        if not (depth_np > 0).any():  # Open3D fails on a view with no valid depth
            skipped += 1
            continue
        depth = o3d.t.geometry.Image(o3d.core.Tensor(depth_np[:, :, None].astype(np.float32)))
        color = o3d.t.geometry.Image(o3d.core.Tensor(np.zeros((h, w, 3), np.float32)))
        K, T = o3d.core.Tensor(view.K, o3d.core.float64), o3d.core.Tensor(view.cam_from_world, o3d.core.float64)
        coords = grid.compute_unique_block_coordinates(depth, K, T, 1.0, max_depth, multiplier)
        if coords.shape[0] == 0:
            skipped += 1
            continue
        grid.integrate(coords, depth, color, K, K, T, 1.0, max_depth, multiplier)
        cloud.append(backproject(view, options.point_stride, max_depth))
    if skipped:
        log.warning("fusion: skipped %d of %d views with no valid depth", skipped, len(views))
    if cloud:
        mesh = grid.extract_triangle_mesh(weight_threshold=options.min_weight)
        vertices, faces = mesh.vertex.positions.numpy().astype(float), mesh.triangle.indices.numpy().astype(int)
    else:
        vertices, faces = np.zeros((0, 3)), np.zeros((0, 3), int)
    points = np.concatenate(cloud) if cloud else np.zeros((0, 3))
    if len(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        points = np.asarray(pcd.voxel_down_sample(voxel).points)
    report = {
        "views": len(views),
        "skipped_views": skipped,
        "voxel": voxel,
        "truncation": options.trunc_tau * tau,
        "vertices": len(vertices),
        "faces": len(faces),
        "points": len(points),
    }
    return vertices, faces, points, report
