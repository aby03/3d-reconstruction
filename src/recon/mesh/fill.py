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
from scipy.spatial import cKDTree

from recon.io.ply import read_mesh


@dataclass(slots=True)
class FillOptions:
    enabled: bool = True
    gap_tau: float = 3.0  # a filler face is a gap face when its centroid is farther than this many tau from the mesh
    min_area_frac: float = 0.005  # a connected patch of gap faces is added when its area is at least this share of the mesh area
    overlap_tau: float = 2.0  # patches grow by this many tau beyond gap_tau, so they overlap the mesh
    normal_test: bool = False  # gap = no mesh surface within `distance_voxels` along the filler face's normal (False: centroid farther than gap_tau from the mesh)
    distance_voxels: float = 1.5  # reach of the normal test, in TSDF voxels
    weld: bool = False  # drop the mesh faces under a patch's overlap and join the two along the cut
    weld_tau: float = 1.0  # mesh and patch border vertices closer than this many tau are merged


def fill_gaps(
    mesh: Path,
    filler: Path,
    out: Path,
    tau: float,
    options: FillOptions,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    voxel: float | None = None,
    patch_out: Path | None = None,
) -> dict[str, Any]:
    """Write to `out` the mesh `mesh` plus the connected patches of `filler` that lie in gaps of it (inside
    `bounds` when given). `patch_out` receives the added patches alone. `voxel` is the TSDF voxel size (the normal test falls back to gap_tau without it). The patches are separate components; planes and cleanup run after. Returns the counts."""
    a, b = read_mesh(mesh), read_mesh(filler)
    report: dict[str, Any] = {"filler_faces": len(b.faces), "gap_faces": 0, "patches": 0, "added_faces": 0, "added_area_frac": 0.0}
    if not len(b.faces):
        a.export(out)
        return report
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(a.vertices), o3d.core.float32), o3d.core.Tensor(np.asarray(a.faces), o3d.core.uint32))
    centres = b.triangles_center
    if options.normal_test and voxel:
        reach = options.distance_voxels * voxel
        gap = ~_covered(scene, centres, b.face_normals, reach)
        grow = reach  # the patch edge lies at the reach from the mesh
    else:
        closest = scene.compute_closest_points(o3d.core.Tensor(centres, o3d.core.float32))["points"].numpy().astype(float)
        gap = np.linalg.norm(closest - centres, axis=1) > options.gap_tau * tau
        grow = options.gap_tau * tau
    if bounds is not None:
        gap &= np.all((centres >= bounds[0]) & (centres <= bounds[1]), axis=1)
    report["gap_faces"] = int(gap.sum())
    adjacency = b.face_adjacency[gap[b.face_adjacency].all(axis=1)]
    graph = coo_matrix((np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])), shape=(len(b.faces),) * 2)
    count, labels = connected_components(graph, directed=False)
    area = np.bincount(labels, weights=b.area_faces, minlength=count)
    big = area >= options.min_area_frac * a.area
    keep = gap & big[labels]
    if keep.any() and options.overlap_tau > 0:
        # the gap is measured from the mesh, so a patch stops gap_tau short of it: grow it across that ring
        reach = grow + options.overlap_tau * tau
        near = cKDTree(centres[keep]).query(centres, distance_upper_bound=reach)[0] <= reach
        keep |= near & (bounds is None or np.all((centres >= bounds[0]) & (centres <= bounds[1]), axis=1))
    report["patches"] = int((big & (np.bincount(labels[gap], minlength=count) > 0)).sum())
    report["added_faces"] = int(keep.sum())
    report["added_area_frac"] = float(b.area_faces[keep].sum() / a.area)
    if keep.any():
        patch = b.submesh([np.flatnonzero(keep)], append=True)
        if patch_out is not None:
            patch.export(patch_out)
        if options.weld:
            a, report["welded_vertices"] = _weld(a, patch, tau, options)
        else:
            a = trimesh.util.concatenate([a, patch])
    a.export(out)
    return report


def _covered(scene: o3d.t.geometry.RaycastingScene, centres: np.ndarray, normals: np.ndarray, reach: float) -> np.ndarray:
    """True where a mesh surface lies within `reach` of the point along the normal, on either side."""
    covered = np.zeros(len(centres), bool)
    for sign in (1.0, -1.0):
        rays = np.hstack([centres, sign * normals]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.float32))["t_hit"].numpy()
        covered |= hit <= reach
    return covered


def _border_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Indices of the vertices on boundary edges (edges used by one face)."""
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])


def _weld(mesh: trimesh.Trimesh, patch: trimesh.Trimesh, tau: float, options: FillOptions) -> tuple[trimesh.Trimesh, int]:
    """Remove the mesh faces that lie under the patch (within the overlap width), then snap the cut border of
    the mesh onto the border of the patch and merge the coincident vertices."""
    under = cKDTree(patch.triangles_center).query(mesh.triangles_center)[0] <= options.overlap_tau * tau
    mesh = mesh.copy()
    mesh.update_faces(~under)
    mesh.remove_unreferenced_vertices()
    border_m, border_p = _border_vertices(mesh), _border_vertices(patch)
    snapped = 0
    if len(border_m) and len(border_p):
        dist, idx = cKDTree(patch.vertices[border_p]).query(mesh.vertices[border_m], distance_upper_bound=options.weld_tau * tau)
        close = np.isfinite(dist)
        mesh.vertices[border_m[close]] = patch.vertices[border_p[idx[close]]]
        snapped = int(close.sum())
    joined = trimesh.util.concatenate([mesh, patch])
    joined.merge_vertices()
    joined.update_faces(joined.nondegenerate_faces())
    for _ in range(5):  # snapping many border vertices onto one makes edges shared by 3+ faces: drop those faces
        unique, inverse, counts = np.unique(joined.edges_sorted, axis=0, return_inverse=True, return_counts=True)
        bad = (counts[inverse.ravel()] > 2).reshape(-1, 3).any(axis=1)
        if not bad.any():
            break
        joined.update_faces(~bad)
    joined.remove_unreferenced_vertices()
    return joined, snapped
