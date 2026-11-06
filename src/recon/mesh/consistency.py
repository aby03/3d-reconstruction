"""Mesh-vs-view consistency: faces the keyframes see through, found by ray casting the mesh into each view and
comparing its depth with the fitted mono depth."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from recon.io.ply import read_mesh


@dataclass(slots=True)
class ConsistencyOptions:
    enabled: bool = True
    pixel_stride: int = 2  # every n-th pixel of each depth map is a ray
    free_rel: float = 0.1  # the mesh is in free space where it is nearer than the view depth by this fraction of it ...
    free_tau: float = 2.0  # ... and by at least this many tau
    min_face_frac: float = 0.5  # a view sees a face as free space when this share of the face's pixels is
    min_views: int = 3  # a face is removed when at least this many views see it as free space
    unsupported_tau: float = 2.0  # only faces farther than this many tau from every dense point can be removed
    fused_tau: float = 2.0  # ... and farther than this many tau from the fused mono-depth (TSDF) surface; 0: off
    min_inlier_ratio: float = 0.7  # only views whose mono depth fits the SfM points at least this well count ...
    max_residual: float = 0.05  # ... with a median relative residual of at most this


@dataclass(slots=True)
class ConsistencyView:
    depth: np.ndarray  # (h, w) float32, 0 = no depth
    K: np.ndarray  # (3, 3) intrinsics at the depth map's size
    cam_from_world: np.ndarray  # (4, 4)


def free_space_votes(
    vertices: np.ndarray, faces: np.ndarray, views: list[ConsistencyView], tau: float, options: ConsistencyOptions
) -> tuple[np.ndarray, np.ndarray]:
    """For each face, the number of views that see it (it is the nearest surface at some pixel) and the number
    that see it as free space (a share of at least `min_face_frac` of its pixels in that view are nearer than
    the fitted depth by `max(free_rel·depth, free_tau·tau)`)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices, o3d.core.float32), o3d.core.Tensor(faces, o3d.core.uint32))
    seen = np.zeros(len(faces), int)
    free = np.zeros(len(faces), int)
    n = len(faces)
    for view in views:
        h, w = view.depth.shape
        s = options.pixel_stride
        v, u = np.mgrid[0:h:s, 0:w:s]
        depth = view.depth[v, u].astype(float)
        # rays of the sampled pixels, through the camera centre
        pix = np.stack([(u - view.K[0, 2]) / view.K[0, 0], (v - view.K[1, 2]) / view.K[1, 1], np.ones_like(u, float)], -1)
        world_from_cam = np.linalg.inv(view.cam_from_world)
        d = pix.reshape(-1, 3) @ world_from_cam[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(world_from_cam[:3, 3], d.shape)
        rays = o3d.core.Tensor(np.concatenate([o, d], axis=1), o3d.core.float32)
        hit = scene.cast_rays(rays)
        t, ids = hit["t_hit"].numpy().astype(float), hit["primitive_ids"].numpy().astype(np.int64)
        ok = np.isfinite(t) & (ids >= 0) & (depth.ravel() > 0)
        if not ok.any():
            continue
        z_mesh = t[ok] / np.linalg.norm(pix.reshape(-1, 3)[ok], axis=1)  # distance along the ray -> depth along z
        z_view = depth.ravel()[ok]
        through = z_mesh + np.maximum(options.free_rel * z_view, options.free_tau * tau) < z_view
        total = np.bincount(ids[ok], minlength=n)
        bad = np.bincount(ids[ok][through], minlength=n)
        visible = total > 0
        seen += visible
        free += visible & (bad >= options.min_face_frac * np.maximum(total, 1))
    return seen, free


def surface_distance(mesh: Path, points: np.ndarray) -> np.ndarray:
    """Distance from each of `points` (N, 3) to the surface of `mesh`."""
    m = read_mesh(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(m.vertices), o3d.core.float32), o3d.core.Tensor(np.asarray(m.faces), o3d.core.uint32))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points), o3d.core.float32)).numpy()


def remove_free_space(
    src: Path,
    out: Path,
    points: np.ndarray,
    views: list[ConsistencyView],
    tau: float,
    options: ConsistencyOptions | None = None,
    fused: Path | None = None,
) -> dict[str, Any]:
    """Write to `out` the mesh `src` without the faces that `options.min_views` views see as free space and
    whose centroid is farther than `unsupported_tau`·tau from every point of `points` (faces backed by dense
    points stay) and, with `fused` and `fused_tau` > 0, farther than `fused_tau`·tau from the surface of the mesh
    `fused` (the depth maps fused over many views outvote a single view's depth). Returns the counts."""
    options = options or ConsistencyOptions()
    mesh = read_mesh(src)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    seen, free = free_space_votes(vertices, faces, views, tau, options)
    limit = options.unsupported_tau * tau
    unsupported = cKDTree(points).query(mesh.triangles_center, distance_upper_bound=limit)[0] > limit
    flagged = free >= options.min_views
    on_fused = np.zeros(len(faces), bool)
    if fused is not None and options.fused_tau > 0 and fused.is_file():
        on_fused = surface_distance(fused, mesh.triangles_center) <= options.fused_tau * tau
    remove = flagged & unsupported & ~on_fused
    area = mesh.area_faces
    report = {
        "views": len(views),
        "faces": len(faces),
        "seen": int((seen > 0).sum()),
        "flagged": int(flagged.sum()),
        "removed": int(remove.sum()),
        "removed_area_frac": float(area[remove].sum() / area.sum()) if area.sum() > 0 else 0.0,
        "kept_supported": int((flagged & ~unsupported).sum()),
        "kept_fused": int((flagged & unsupported & on_fused).sum()),
    }
    kept = trimesh.Trimesh(vertices, faces[~remove], process=False)
    kept.remove_unreferenced_vertices()
    kept.export(out)
    return report
