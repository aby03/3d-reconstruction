"""Mesh cleanup with pymeshlab: duplicates, degenerate faces, unreferenced vertices, non-manifold elements and
small connected components."""

from dataclasses import dataclass
from pathlib import Path

import pymeshlab


@dataclass(slots=True)
class CleanupOptions:
    min_component_frac: float = 0.005  # components with fewer faces than this fraction go ...
    min_component_diam_frac: float = 0.02  # ... and those smaller than this fraction of the bbox diagonal


@dataclass(slots=True)
class CleanupResult:
    before: dict[str, int]
    after: dict[str, int]
    components_removed: int
    non_manifold_fixed: dict[str, int]  # edges, vertices


def topology(ms: pymeshlab.MeshSet) -> dict[str, int]:
    """Vertex, face, component, boundary-edge and non-manifold counts of the current mesh."""
    t = ms.get_topological_measures()
    return {
        "vertices": t["vertices_number"],
        "faces": t["faces_number"],
        "components": t["connected_components_number"],
        "boundary_edges": t["boundary_edges"],
        "non_manifold_edges": t["non_two_manifold_edges"],
        "non_manifold_vertices": t["non_two_manifold_vertices"],
    }


def load(path: Path) -> pymeshlab.MeshSet:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    return ms


def save(ms: pymeshlab.MeshSet, path: Path) -> Path:
    """Binary PLY without vertex colours (OpenMVS's meshes have none worth keeping)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(path), binary=True, save_vertex_color=False)
    return path


def cleanup(src: Path, out: Path, options: CleanupOptions | None = None) -> CleanupResult:
    """Clean the mesh `src` into the PLY `out`, in this order: duplicate vertices and faces, zero-area faces,
    unreferenced vertices, non-manifold edges (their faces removed) and vertices (split), then components below
    `min_component_frac` of the faces or `min_component_diam_frac` of the bbox diagonal."""
    options = options or CleanupOptions()
    ms = load(src)
    before = topology(ms)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices()
    min_faces = int(options.min_component_frac * ms.current_mesh().face_number())
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_faces)
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pymeshlab.PercentageValue(100 * options.min_component_diam_frac)
    )
    ms.meshing_remove_unreferenced_vertices()
    after = topology(ms)
    save(ms, out)
    fixed = {
        "edges": before["non_manifold_edges"] - after["non_manifold_edges"],
        "vertices": before["non_manifold_vertices"] - after["non_manifold_vertices"],
    }
    return CleanupResult(before, after, before["components"] - after["components"], fixed)

