"""Add the gap-filling mesh (TSDF of the mono depth) where the OpenMVS mesh has no surface: textureless walls,
ceiling and floor leave the dense cloud empty, so the graph cut leaves them out."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from recon.io.ply import read_mesh


@dataclass(slots=True)
class FillOptions:
    enabled: bool = True
    gap_tau: float = 3.0  # a filler face is a gap face when its centroid is farther than this many tau from the mesh
    min_area_frac: float = 0.005  # a connected patch of gap faces is added when its area is at least this share of the mesh area


def fill_gaps(
    mesh: Path, filler: Path, out: Path, tau: float, options: FillOptions, bounds: tuple[np.ndarray, np.ndarray] | None = None
) -> dict[str, Any]:
    """Write to `out` the mesh `mesh` plus the connected patches of `filler` that lie in gaps of it (inside
    `bounds` when given). The patches are separate components; planes and cleanup run after. Returns the counts."""
    a, b = read_mesh(mesh), read_mesh(filler)
    report: dict[str, Any] = {"filler_faces": len(b.faces), "gap_faces": 0, "patches": 0, "added_faces": 0, "added_area_frac": 0.0}
    if not len(b.faces):
        a.export(out)
        return report
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(a.vertices), o3d.core.float32), o3d.core.Tensor(np.asarray(a.faces), o3d.core.uint32))
    centres = b.triangles_center
    closest = scene.compute_closest_points(o3d.core.Tensor(centres, o3d.core.float32))["points"].numpy().astype(float)
    gap = np.linalg.norm(closest - centres, axis=1) > options.gap_tau * tau
    if bounds is not None:
        gap &= np.all((centres >= bounds[0]) & (centres <= bounds[1]), axis=1)
    report["gap_faces"] = int(gap.sum())
    adjacency = b.face_adjacency[gap[b.face_adjacency].all(axis=1)]
    graph = coo_matrix((np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])), shape=(len(b.faces),) * 2)
    count, labels = connected_components(graph, directed=False)
    area = np.bincount(labels, weights=b.area_faces, minlength=count)
    big = area >= options.min_area_frac * a.area
    keep = gap & big[labels]
    report["patches"] = int((big & (np.bincount(labels[gap], minlength=count) > 0)).sum())
    report["added_faces"] = int(keep.sum())
    report["added_area_frac"] = float(b.area_faces[keep].sum() / a.area)
    if keep.any():
        a = trimesh.util.concatenate([a, b.submesh([np.flatnonzero(keep)], append=True)])
    a.export(out)
    return report
