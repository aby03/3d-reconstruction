"""Adopt a gap-filling mesh (TSDF of the mono depth) where the OpenMVS mesh is unsupported: the unsupported
faces are subdivided and their vertices moved onto the gap filler. The mesh keeps its topology."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pymeshlab
import trimesh
from scipy.spatial import cKDTree

from recon.io.ply import read_mesh

SUBDIVIDE_ITERATIONS = 10


@dataclass(slots=True)
class MergeOptions:
    enabled: bool = True
    unsupported_tau: float = 2.0  # a face farther than this many tau from every dense point is unsupported
    reach_tau: float = 2.0  # an unsupported face adopts the gap filler within this many tau of it
    min_coverage: float = 0.5  # a region of unsupported faces moves when this share of its area has the gap filler within reach


def _closest(scene: o3d.t.geometry.RaycastingScene, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point on the scene's mesh, its distance and the id of its face, per query point."""
    answer = scene.compute_closest_points(o3d.core.Tensor(query, o3d.core.float32))
    points = answer["points"].numpy().astype(float)
    return points, np.linalg.norm(points - query, axis=1), answer["primitive_ids"].numpy().astype(int)


def merge_unsupported(
    openmvs: Path, filler: Path, points: np.ndarray, out: Path, tau: float, options: MergeOptions | None = None
) -> dict[str, Any]:
    """Write to `out` the OpenMVS mesh `openmvs` with its unsupported part conformed to `filler`. Edges longer
    than `reach_tau`·tau are split first (the surface is unchanged), then a face is *unsupported* when its
    centroid is farther than `unsupported_tau`·tau from every point of `points`, and *adoptable* when it is
    unsupported and `filler` lies within `reach_tau`·tau of its centroid. A connected region of unsupported
    faces moves when at least `min_coverage` of its area is adoptable; its interior vertices (every incident
    face in a moved region) go to the closest point of `filler` when that is within reach. Returns the counts."""
    options = options or MergeOptions()
    reach, limit = options.reach_tau * tau, options.unsupported_tau * tau
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(openmvs))
    faces_before = ms.current_mesh().face_number()
    ms.meshing_surface_subdivision_midpoint(iterations=SUBDIVIDE_ITERATIONS, threshold=pymeshlab.PureValue(reach))
    v, f = ms.current_mesh().vertex_matrix().astype(float), ms.current_mesh().face_matrix().astype(int)
    mesh = trimesh.Trimesh(v, f, process=False)
    b = read_mesh(filler)
    bf = np.asarray(b.faces)
    report: dict[str, Any] = {
        "input_faces": faces_before, "faces": len(f), "unsupported": 0, "adoptable": 0, "regions": 0,
        "regions_moved": 0, "vertices_moved": 0, "move_mean": 0.0, "move_max": 0.0, "filler_faces": len(b.faces),
    }
    centres = mesh.triangles_center
    far = cKDTree(points).query(centres, distance_upper_bound=limit)[0] > limit
    report["unsupported"] = int(far.sum())
    moved = v.copy()
    if far.any() and len(b.faces):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(np.asarray(b.vertices), o3d.core.float32), o3d.core.Tensor(np.asarray(b.faces), o3d.core.uint32)
        )
        _, face_dist, _ = _closest(scene, centres[far])
        adoptable = np.zeros(len(f), bool)
        adoptable[np.flatnonzero(far)[face_dist <= reach]] = True
        report["adoptable"] = int(adoptable.sum())
        # regions of unsupported faces; a region moves when enough of it can adopt the gap filler
        adjacency = mesh.face_adjacency[far[mesh.face_adjacency].all(axis=1)]
        labels = trimesh.graph.connected_component_labels(adjacency, node_count=len(f))
        ids = np.flatnonzero(far)
        area = np.bincount(labels[ids], weights=mesh.area_faces[ids], minlength=labels.max() + 1)
        adopt_area = np.bincount(labels[ids], weights=(mesh.area_faces * adoptable)[ids], minlength=labels.max() + 1)
        ok = (area > 0) & (adopt_area >= options.min_coverage * area)
        report["regions"] = int((area > 0).sum())
        report["regions_moved"] = int(ok.sum())
        move_face = far & ok[labels]
        # interior vertices: all incident faces move
        incident = np.zeros(len(v), int)
        np.add.at(incident, f.ravel(), 1)
        moving = np.zeros(len(v), int)
        np.add.at(moving, f.ravel(), np.repeat(move_face, 3))
        interior = np.flatnonzero((incident > 0) & (moving == incident))
        if len(interior):
            target, dist, face = _closest(scene, v[interior])
            # onto the plane of the closest gap-filler face, not to the point: vertices keep their spread along
            # the surface, so none of them coincide
            n = np.cross(np.asarray(b.vertices)[bf[face, 1]] - np.asarray(b.vertices)[bf[face, 0]],
                         np.asarray(b.vertices)[bf[face, 2]] - np.asarray(b.vertices)[bf[face, 0]])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            offset = ((v[interior] - target) * n).sum(axis=1)
            go = (dist <= reach) & (np.abs(offset) > 0)
            moved[interior[go]] = v[interior[go]] - offset[go, None] * n[go]
            report["vertices_moved"] = int(go.sum())
            if go.any():
                report["move_mean"], report["move_max"] = float(np.abs(offset[go]).mean()), float(np.abs(offset[go]).max())
    trimesh.Trimesh(moved, f, process=False).export(out)
    return report
