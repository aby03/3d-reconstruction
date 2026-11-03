"""Geometry QA of the final mesh against the dense point cloud it was built from."""

from pathlib import Path
from typing import Any

import numpy as np
import pymeshlab
from scipy.spatial import cKDTree

from recon.io.ply import read_mesh

METRIC_TAU = 0.02  # metres
ARBITRARY_TAU_FRAC = 0.005  # of the scene diagonal


def tau(units: str, diag: float, snap_dist: float | None = None) -> float:
    """The geometry tolerance: `snap_dist` when given, else 2 cm in metres or 0.5% of `diag` in arbitrary units."""
    if snap_dist is not None:
        return float(snap_dist)
    return METRIC_TAU if units == "m" else ARBITRARY_TAU_FRAC * float(diag)


def dense_to_mesh(points: np.ndarray, mesh: Path, samples: int = 200_000, seed: int = 0) -> dict[str, Any]:
    """Distance from up to `samples` dense points (a seeded random subset) to the surface of `mesh`: mean, RMS,
    p50, p95 and max, in scene units."""
    rng = np.random.default_rng(seed)
    if len(points) > samples:
        points = points[np.sort(rng.choice(len(points), samples, replace=False))]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh))
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(points, dtype=float)))
    ms.get_hausdorff_distance(
        sampledmesh=1,
        targetmesh=0,
        samplevert=True,
        sampleedge=False,
        sampleface=False,
        samplenum=len(points),
        maxdist=pymeshlab.PercentageValue(50),
    )
    d = ms.mesh(1).vertex_scalar_array()
    return {
        "mean": float(d.mean()),
        "rms": float(np.sqrt((d**2).mean())),
        "p50": float(np.percentile(d, 50)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "samples": len(d),
    }


def unsupported_area_frac(mesh: Path, points: np.ndarray, dist: float) -> float:
    """Share of the mesh area whose face centroids are farther than `dist` from every dense point."""
    m = read_mesh(mesh)
    far = cKDTree(points).query(m.triangles_center, distance_upper_bound=dist)[0] > dist
    area = m.area_faces
    return float(area[far].sum() / area.sum()) if area.sum() > 0 else 0.0
