"""Flat patches for the holes that walls, floor and ceiling have left after the plane snap: a hole in a plane's
raster that the plane's own surface encloses is filled with the plane itself, unless the keyframes see through it
(a door, a window) or see a surface in front of it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

from recon.io.ply import read_mesh
from recon.mesh.consistency import ConsistencyView
from recon.mesh.planes import Plane

SAMPLES_PER_CELL = 4.0  # surface samples per cell when rasterising the plane's faces


@dataclass(slots=True)
class PlaneFillOptions:
    enabled: bool = False
    cell_frac: float = 0.004  # raster cell, as a fraction of the diagonal
    min_plane_area_frac: float = 0.02  # planes with a smaller share of the area are left alone
    max_hole_frac: float = 0.15  # a hole larger than this share of the plane's surface is an opening, not a gap
    close_cells: int = 1  # cells of morphological closing before holes are looked for
    see_through_tau: float = 3.0  # a view sees through a cell when its depth there exceeds the plane's by this many tau ...
    in_front_tau: float = 3.0  # ... and sees a surface in front of it when its depth is smaller by this many tau
    min_votes: int = 2  # views needed to mark a cell as see-through / covered
    max_rejected_frac: float = 0.3  # a hole with more rejected cells than this share is left open


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _votes(points: np.ndarray, views: list[ConsistencyView], tau: float, options: PlaneFillOptions) -> tuple[np.ndarray, np.ndarray]:
    """Per point, the number of views that see through it and that see a surface in front of it."""
    through, front = np.zeros(len(points), int), np.zeros(len(points), int)
    for view in views:
        cam = points @ view.cam_from_world[:3, :3].T + view.cam_from_world[:3, 3]
        z = cam[:, 2]
        uv = cam @ view.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = np.round(uv[:, 0] / uv[:, 2]).astype(int), np.round(uv[:, 1] / uv[:, 2]).astype(int)
        h, w = view.depth.shape
        ok = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        seen = np.zeros(len(points))
        seen[ok] = view.depth[y[ok], x[ok]]
        valid = ok & (seen > 0)
        through += valid & (seen > z + options.see_through_tau * tau)
        front += valid & (seen < z - options.in_front_tau * tau)
    return through, front


def fill_plane_holes(
    src: Path, out: Path, planes: list[Plane], views: list[ConsistencyView], tau: float, diag: float, options: PlaneFillOptions
) -> dict[str, Any]:
    """Write to `out` the mesh `src` plus flat patches over the enclosed holes of its planes (floor, walls and
    ceiling with at least `min_plane_area_frac` of the area). Returns the counts."""
    mesh = read_mesh(src)
    report: dict[str, Any] = {"enabled": True, "planes": 0, "holes": 0, "holes_filled": 0, "cells_added": 0, "cells_rejected": 0}
    cell = options.cell_frac * diag
    vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    centres, normals, area = mesh.triangles_center, mesh.face_normals, mesh.area_faces
    patches = []
    rng = np.random.default_rng(0)
    for plane in planes:
        if plane.label not in ("floor", "wall", "ceiling") or plane.area_frac < options.min_plane_area_frac:
            continue
        n, d = np.asarray(plane.normal), plane.d
        flat = (np.abs(vertices @ n - d) < 0.5 * tau)[faces].all(axis=1) & (normals @ n > 0.9)
        if not flat.any():
            continue
        u, v = _basis(n)
        picks = np.flatnonzero(flat)
        count = np.maximum(1, np.ceil(SAMPLES_PER_CELL * area[picks] / cell**2).astype(int))
        owner = np.repeat(picks, count)
        r = rng.random((len(owner), 2))
        flip = r.sum(axis=1) > 1
        r[flip] = 1 - r[flip]
        tri = vertices[faces[owner]]
        samples = tri[:, 0] + r[:, :1] * (tri[:, 1] - tri[:, 0]) + r[:, 1:] * (tri[:, 2] - tri[:, 0])
        su, sv = samples @ u, samples @ v
        lo = np.array([su.min(), sv.min()]) - cell
        ij = np.floor((np.stack([su, sv], 1) - lo) / cell).astype(int)
        shape = tuple(ij.max(axis=0) + 2)
        occupied = np.zeros(shape, bool)
        occupied[ij[:, 0], ij[:, 1]] = True
        closed = ndimage.binary_closing(occupied, iterations=options.close_cells) if options.close_cells else occupied
        holes = ndimage.binary_fill_holes(closed) & ~closed
        labels, num = ndimage.label(holes)
        report["planes"] += 1
        report["holes"] += num
        if not num:
            continue
        surface = int(closed.sum())
        for k in range(1, num + 1):
            cells = np.argwhere(labels == k)
            if len(cells) > options.max_hole_frac * surface:
                continue
            centre_uv = lo + (cells + 0.5) * cell
            points = np.outer(centre_uv[:, 0], u) + np.outer(centre_uv[:, 1], v) + d * n
            through, front = _votes(points, views, tau, options)
            rejected = (through >= options.min_votes) | (front >= options.min_votes)
            report["cells_rejected"] += int(rejected.sum())
            if rejected.mean() > options.max_rejected_frac:
                continue
            keep = cells[~rejected]
            if not len(keep):
                continue
            corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            quad = lo + (keep[:, None, :] + corner[None]) * cell  # (m, 4, 2)
            xyz = quad[..., :1] * u + quad[..., 1:] * v + d * n  # (m, 4, 3)
            m = len(keep)
            idx = np.arange(m * 4).reshape(m, 4)
            tri_idx = np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]])
            patch = trimesh.Trimesh(xyz.reshape(-1, 3), tri_idx, process=False)
            if (patch.face_normals @ n).mean() < 0:  # keep the plane's orientation
                patch.invert()
            patches.append(patch)
            report["holes_filled"] += 1
            report["cells_added"] += m
    result = trimesh.util.concatenate([mesh, *patches]) if patches else mesh
    result.export(out)
    return report
