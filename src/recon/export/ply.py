"""Geometry-only exports in world W (Z up): a binary PLY with per-vertex colours sampled from the texture atlases,
and a binary STL, each with a check of the written file."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from recon.io.ply import read_mesh
from recon.mesh.texturing import read_mtl_maps, read_obj_faces, read_obj_uv


def _check(mesh: trimesh.Trimesh, name: str, faces: int, bounds_w: np.ndarray, diag: float) -> None:
    if len(mesh.faces) != faces:
        raise ValueError(f"{name}: {len(mesh.faces)} faces, the source has {faces}")
    if np.abs(np.asarray(mesh.bounds) - np.asarray(bounds_w)).max() > 1e-4 * diag:
        raise ValueError(f"{name}: bounds {mesh.bounds.tolist()} != the source's {np.asarray(bounds_w).tolist()}")


def vertex_colors(textured_obj: Path, num_vertices: int) -> np.ndarray:
    """(V, 3) uint8 RGB per vertex of the textured OBJ: the mean of the atlas texels at every face corner (UV) the
    vertex has."""
    mtllib, uv, uv_faces, _ = read_obj_uv(textured_obj)
    _, vertex_faces = read_obj_faces(textured_obj)
    maps = read_mtl_maps(textured_obj.parent / mtllib)
    total, count = np.zeros((num_vertices, 3)), np.zeros(num_vertices)
    for name, idx in uv_faces.items():
        image = cv2.imread(str(textured_obj.parent / maps[name]))[..., ::-1]  # RGB
        h, w = image.shape[:2]
        corners, vertices = idx.ravel(), vertex_faces[name].ravel()
        has_uv = corners >= 0
        corners, vertices = corners[has_uv], vertices[has_uv]
        x = np.clip((uv[corners, 0] * w).astype(int), 0, w - 1)
        y = np.clip(((1.0 - uv[corners, 1]) * h).astype(int), 0, h - 1)
        np.add.at(total, vertices, image[y, x])
        np.add.at(count, vertices, 1)
    return np.round(total / np.maximum(count, 1)[:, None]).astype(np.uint8)


def write_vertex_color_ply(textured_obj: Path, geometry_ply: Path, out: Path) -> dict[str, Any]:
    """`out`: the geometry of `geometry_ply` (the textured mesh's vertices and faces, welded) as a binary PLY with
    per-vertex colours from the atlases of `textured_obj`. Raises ValueError when the two meshes differ."""
    geometry = read_mesh(geometry_ply)
    vertices, faces = read_obj_faces(textured_obj)
    if len(vertices) != len(geometry.vertices) or not np.array_equal(
        np.concatenate(list(faces.values())), np.asarray(geometry.faces)
    ):
        raise ValueError(f"{textured_obj.name} and {geometry_ply.name} have different vertices or faces")
    colors = np.c_[vertex_colors(textured_obj, len(vertices)), np.full(len(vertices), 255, np.uint8)]
    mesh = trimesh.Trimesh(geometry.vertices, geometry.faces, vertex_colors=colors, process=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(trimesh.exchange.ply.export_ply(mesh, encoding="binary", include_attributes=False))
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}


def validate_vertex_color_ply(path: Path, faces: int, bounds_w: np.ndarray, diag: float) -> dict[str, Any]:
    """Reload the PLY: `faces` faces, the W bounds, one colour per vertex. Raises ValueError on a failure."""
    mesh = read_mesh(path)
    _check(mesh, path.name, faces, bounds_w, diag)
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        raise ValueError(f"{path.name}: no vertex colours")
    return {"faces": len(mesh.faces), "vertices": len(mesh.vertices), "textures": []}


def write_stl(geometry_ply: Path, out: Path) -> dict[str, Any]:
    """`out`: the geometry of `geometry_ply` as a binary STL (W, Z up)."""
    mesh = read_mesh(geometry_ply)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    return {"faces": len(mesh.faces)}


def validate_stl(path: Path, faces: int, bounds_w: np.ndarray, diag: float) -> dict[str, Any]:
    """Reload the STL: `faces` faces and the W bounds. Raises ValueError on a failure."""
    mesh = trimesh.load(path, process=False, force="mesh")
    _check(mesh, path.name, faces, bounds_w, diag)
    return {"faces": len(mesh.faces), "vertices": len(mesh.vertices), "textures": []}
