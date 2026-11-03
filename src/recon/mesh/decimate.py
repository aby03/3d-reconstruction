"""Quadric edge-collapse decimation to the texturing face budget."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recon.mesh.cleanup import load, save

QUALITY_THRESHOLD = 0.3  # pymeshlab's qualitythr: penalizes collapses that create slivers


@dataclass(slots=True)
class DecimateOptions:
    texture_face_budget: int = 600_000  # decimate only meshes with more faces
    planar_quadric: bool = True  # planar regions collapse to few triangles


def decimate(src: Path, out: Path, options: DecimateOptions | None = None) -> dict[str, Any]:
    """Decimate the mesh `src` to `texture_face_budget` faces (normals and boundary preserved, vertex normals
    recomputed) into the PLY `out`; a mesh within the budget is copied unchanged."""
    options = options or DecimateOptions()
    ms = load(src)
    faces_in = ms.current_mesh().face_number()
    if faces_in <= options.texture_face_budget:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        return {"decimated": False, "faces_in": faces_in, "faces_out": faces_in}
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=options.texture_face_budget,
        planarquadric=options.planar_quadric,
        preservenormal=True,
        preserveboundary=True,
        qualitythr=QUALITY_THRESHOLD,
    )
    ms.meshing_remove_unreferenced_vertices()
    ms.compute_normal_per_vertex()
    save(ms, out)
    return {"decimated": True, "faces_in": faces_in, "faces_out": ms.current_mesh().face_number()}
