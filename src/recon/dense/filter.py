"""Outlier removal for the dense point cloud: the room box, then Open3D statistical and radius filters."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d

from recon.mesh.roi import RoiOptions, room_bounds


@dataclass(slots=True)
class FilterOptions:
    enabled: bool = True
    roi: RoiOptions = field(default_factory=RoiOptions)
    stat_neighbors: int = 30  # statistical filter: neighbours per point ...
    stat_std_ratio: float = 3.0  # ... and the allowed std-devs above the mean neighbour distance
    radius_frac: float = 0.01  # radius filter: radius as a fraction of the diagonal ...
    radius_min_neighbors: int = 8  # ... and the neighbours a point needs inside it


def _cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    return cloud


def filter_points(
    xyz: np.ndarray, diag: float, options: FilterOptions | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boolean keep mask over `xyz` (N,3) and a report: the room box, then statistical, then radius outlier
    removal on what is left (the box only when `roi.enabled`). `diag` is the scene diagonal."""
    options = options or FilterOptions()
    keep = np.ones(len(xyz), bool)
    report: dict[str, Any] = {"input": len(xyz), "box": None, "outside_box": 0, "statistical": 0, "radius": 0}
    if options.roi.enabled and len(xyz):
        lo, hi = room_bounds(xyz, diag, options.roi)
        keep = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
        report["box"] = {"min": lo.tolist(), "max": hi.tolist()}
        report["outside_box"] = int((~keep).sum())
    for name, remove in (
        ("statistical", lambda c: c.remove_statistical_outlier(options.stat_neighbors, options.stat_std_ratio)),
        (
            "radius",
            lambda c: c.remove_radius_outlier(options.radius_min_neighbors, options.radius_frac * diag),
        ),
    ):
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            break
        _, inliers = remove(_cloud(xyz[idx]))
        removed = np.ones(len(idx), bool)
        removed[np.asarray(inliers, dtype=int)] = False
        keep[idx[removed]] = False
        report[name] = int(removed.sum())
    report["output"] = int(keep.sum())
    return keep, report
