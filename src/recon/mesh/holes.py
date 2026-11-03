"""Small-hole filling: boundary loops of the mesh, the outer boundary of a partly captured room left open, short
loops closed with pymeshlab."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymeshlab
import trimesh

from recon.io.ply import read_mesh
from recon.mesh.cleanup import save


@dataclass(slots=True)
class HoleOptions:
    small_hole_max_edges: int = 100  # loops with at most this many edges are closed
    max_loop_perimeter_frac: float = 0.3  # loops longer than this fraction of the bbox perimeter are the outer boundary


def _boundary_edges(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Directed boundary edges (B, 2), oriented as in their one face, and that face (B,)."""
    unique = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return mesh.edges[unique], mesh.edges_face[unique]


def _walk(edges: np.ndarray) -> list[np.ndarray]:
    nexts: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        nexts.setdefault(a, []).append(b)
    loops = []
    while nexts:
        start = next(iter(nexts))
        loop, v = [start], start
        while True:
            succ = nexts[v]
            w = succ.pop()
            if not succ:
                del nexts[v]
            if w == start or w not in nexts:
                break
            loop.append(w)
            v = w
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Boundary loops (edges used by one face, walked head to tail) as vertex index arrays."""
    return _walk(_boundary_edges(mesh)[0])


def _perimeter(vertices: np.ndarray, loop: np.ndarray) -> float:
    pts = vertices[loop]
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())


def fill_holes(src: Path, out: Path, options: HoleOptions | None = None) -> dict[str, Any]:
    """Close the short boundary loops of the mesh `src` into the PLY `out`: loops longer than
    `max_loop_perimeter_frac` of the bbox perimeter (the sum of its 12 edges) are the outer boundary and stay open,
    loops with more than `small_hole_max_edges` edges stay open too; the rest are triangulated from their own
    vertices (no new vertices, so there is nothing to fair). `out` is a copy of `src` when nothing is closed."""
    options = options or HoleOptions()
    mesh = read_mesh(src)
    vertices = np.asarray(mesh.vertices)
    loops = boundary_loops(mesh)
    bbox_perimeter = 4 * float(np.ptp(vertices, axis=0).sum()) if len(vertices) else 0.0
    outer = [_perimeter(vertices, loop) > options.max_loop_perimeter_frac * bbox_perimeter for loop in loops]
    large = [not o and len(loop) > options.small_hole_max_edges for loop, o in zip(loops, outer)]
    small = [loop for loop, o, lg in zip(loops, outer, large) if not o and not lg]
    result = {
        "loops": len(loops),
        "filled": 0,
        "filled_faces": 0,
        "skipped_outer": sum(outer),
        "skipped_large": sum(large),
        "largest_loop_edges": max((len(loop) for loop in loops), default=0),
    }
    if not small:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        return result

    # select the faces along the small loops: close_holes(selected=True) closes only loops touching the selection
    edges, edge_faces = _boundary_edges(mesh)
    selected = np.zeros(len(mesh.faces))
    selected[edge_faces[np.isin(edges[:, 0], np.concatenate(small))]] = 1
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=np.asarray(mesh.faces), f_scalar_array=selected))
    holes_before = ms.get_topological_measures()["number_holes"]  # clears the selection: measure first
    ms.compute_selection_by_condition_per_face(condselect="fq > 0")
    ms.meshing_close_holes(
        maxholesize=options.small_hole_max_edges, selected=True, newfaceselected=False, selfintersection=False
    )
    ms.compute_normal_per_vertex()
    result["filled"] = holes_before - ms.get_topological_measures()["number_holes"]
    result["filled_faces"] = ms.current_mesh().face_number() - len(mesh.faces)
    save(ms, out)
    return result
