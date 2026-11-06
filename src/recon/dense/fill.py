"""Mono depth points where the dense cloud has none: textureless surfaces that patch match could not match.

The fitted depth of every view is back-projected, binned into voxels, kept where at least `min_views` views agree
(single-view depth errors and flying pixels do not repeat), and dropped where a dense point is within `radius_tau`
(the dense cloud is the more accurate source)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class FillOptions:
    radius_tau: float = 3.0  # a depth point within this (in tau) of a dense point is not added
    voxel_tau: float = 1.0  # depth points are merged per voxel of this size (in tau)
    min_views: int = 2  # a voxel needs depth from this many views
    stride: int = 4  # every n-th pixel of each depth map
    max_depth_frac: float = 0.7  # depths beyond this fraction of the diagonal are ignored


@dataclass(slots=True)
class FillView:
    depth: np.ndarray  # (h, w) float32, 0 = no depth
    K: np.ndarray  # (3, 3) intrinsics at the depth map's size
    cam_from_world: np.ndarray  # (4, 4)
    rgb: np.ndarray  # (h, w, 3) uint8


def _backproject(view: FillView, stride: int, max_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points, their colours and the unit direction towards the camera, of the strided pixels with depth."""
    h, w = view.depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = view.depth[v, u]
    ok = (z > 0) & (z < max_depth)
    v, u, z = v[ok], u[ok], z[ok]
    cam = np.stack([(u - view.K[0, 2]) / view.K[0, 0] * z, (v - view.K[1, 2]) / view.K[1, 1] * z, z], axis=1)
    world_from_cam = np.linalg.inv(view.cam_from_world)
    rotation, centre = world_from_cam[:3, :3], world_from_cam[:3, 3]
    xyz = cam @ rotation.T + centre
    toward = centre - xyz
    toward /= np.linalg.norm(toward, axis=1, keepdims=True)
    return xyz, view.rgb[v, u], toward


def _voxel_keys(xyz: np.ndarray, origin: np.ndarray, voxel: float) -> np.ndarray:
    ijk = np.floor((xyz - origin) / voxel).astype(np.int64)
    span = ijk.max(axis=0) + 2
    return (ijk[:, 0] * span[1] + ijk[:, 1]) * span[2] + ijk[:, 2]


def fill_points(
    dense_xyz: np.ndarray, views: list[FillView], diag: float, tau: float, options: FillOptions | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """The depth points of `views` that no dense point covers: xyz (N, 3), normals (N, 3, towards the cameras) and
    rgb (N, 3) uint8, plus a report."""
    options = options or FillOptions()
    max_depth = options.max_depth_frac * diag
    parts = [_backproject(v, options.stride, max_depth) for v in views]
    counts = [len(p[0]) for p in parts]
    empty = np.zeros((0, 3))
    report = {"views": len(views), "depth_points": int(sum(counts)), "voxels": 0, "multi_view": 0, "added": 0}
    if not sum(counts):
        return empty, empty, empty.astype(np.uint8), report
    xyz = np.concatenate([p[0] for p in parts]).astype(np.float64)
    rgb = np.concatenate([p[1] for p in parts]).astype(np.float64)
    toward = np.concatenate([p[2] for p in parts])
    view_id = np.repeat(np.arange(len(parts)), counts)

    voxel = options.voxel_tau * tau
    origin = xyz.min(axis=0)
    keys = _voxel_keys(xyz, origin, voxel)
    unique, inverse = np.unique(keys, return_inverse=True)
    report["voxels"] = len(unique)
    pairs = np.unique(np.stack([inverse, view_id]), axis=1)
    seen_by = np.bincount(pairs[0], minlength=len(unique))
    keep = seen_by >= options.min_views
    report["multi_view"] = int(keep.sum())

    n = np.bincount(inverse, minlength=len(unique)).astype(np.float64)[:, None]
    mean = lambda a: np.stack([np.bincount(inverse, a[:, i], len(unique)) for i in range(3)], axis=1) / n
    points, colours, normals = mean(xyz)[keep], mean(rgb)[keep], mean(toward)[keep]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    if len(dense_xyz) and len(points):
        # the dense cloud thinned to one point per voxel is enough to ask "is anything within the radius"
        thin = dense_xyz[np.unique(_voxel_keys(dense_xyz, np.minimum(origin, dense_xyz.min(axis=0)), voxel), return_index=True)[1]]
        distance, _ = cKDTree(thin).query(points, distance_upper_bound=options.radius_tau * tau, workers=-1)
        far = ~np.isfinite(distance)
        points, colours, normals = points[far], colours[far], normals[far]
    report["added"] = len(points)
    return points, normals, np.clip(colours, 0, 255).astype(np.uint8), report
