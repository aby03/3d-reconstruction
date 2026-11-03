"""OpenMVS TextureMesh at full resolution on the final geometry, and statistics of the atlases it writes.

TextureMesh runs in the dense stage's OpenMVS working folder (the image paths in `scene_dense.mvs` are relative
to it); the input mesh and the output folder are passed relative to that folder.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recon.dense.openmvs import SCENE_DENSE, TEXTURE_MESH, MeshResult, check_output, run_with_cpu_fallback
from recon.io.tools import ToolRunner

TEXTURED_OBJ = "mesh.obj"
EMPTY_COLOR = 0xFF00FF  # magenta: R = B, so the packed integer's byte order doesn't matter
SENTINEL_TOL = 40  # max channel distance to magenta that still counts as empty (JPEG blurs patch borders)
BLACK_MAX = 16  # a used texel with every channel below this is black


@dataclass(slots=True)
class TextureOptions:
    resolution_level: int = 0  # full-resolution keyframes
    min_resolution: int = 640
    max_texture_size: int = 8192  # larger atlases are split into several textures
    outlier_threshold: float = 0.06
    cost_smoothness_ratio: float = 0.1
    virtual_face_images: int = 0
    sharpness_weight: float = 0.5
    patch_packing_heuristic: int = 3


def texture_args(options: TextureOptions, device: int, mesh_rel: str, out_rel: str) -> list[str]:
    """TextureMesh argv. Seam leveling is off (OpenMVS 2.4.0 turns textures black with it), and holes are neither
    closed nor faces decimated, so the textured mesh has exactly the faces of the input mesh."""
    o = options
    items = [
        "-w", ".", "-i", SCENE_DENSE, "-m", mesh_rel, "-o", f"{out_rel}/{TEXTURED_OBJ}",
        "--export-type", "obj",
        "--resolution-level", o.resolution_level,
        "--min-resolution", o.min_resolution,
        "--max-texture-size", o.max_texture_size,
        "--outlier-threshold", o.outlier_threshold,
        "--cost-smoothness-ratio", o.cost_smoothness_ratio,
        "--virtual-face-images", o.virtual_face_images,
        "--sharpness-weight", o.sharpness_weight,
        "--patch-packing-heuristic", o.patch_packing_heuristic,
        "--global-seam-leveling", 0,
        "--local-seam-leveling", 0,
        "--close-holes", 0,
        "--decimate", 1,
        "--empty-color", EMPTY_COLOR,
        "--cuda-device", device,
    ]  # fmt: skip
    return [str(item) for item in items]


def texture_mesh(
    runner: ToolRunner,
    work_dir: Path,
    mesh: Path,
    out_dir: Path,
    options: TextureOptions,
    device: int,
    cpu_fallback: bool = False,
) -> MeshResult:
    """Texture `mesh` with the views of `work_dir/scene_dense.mvs` into `out_dir/mesh.obj` (+ `mesh.mtl` and the
    JPEG atlases); the result's `ply` is the OBJ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_rel, out_rel = os.path.relpath(mesh, work_dir), os.path.relpath(out_dir, work_dir)
    used, results, retries = run_with_cpu_fallback(
        runner, TEXTURE_MESH, lambda d: texture_args(options, d, mesh_rel, out_rel), work_dir, device,
        cpu_fallback, options.resolution_level,
    )
    return MeshResult(check_output(TEXTURE_MESH, out_dir / TEXTURED_OBJ), used, False, retries, results)


def read_mtl_maps(mtl: Path) -> dict[str, str]:
    """material name -> its `map_Kd` file name."""
    maps, name = {}, None
    for line in mtl.read_text().splitlines():
        words = line.split(maxsplit=1)
        if words[:1] == ["newmtl"]:
            name = words[1].strip()
        elif words[:1] == ["map_Kd"] and name is not None:
            maps[name] = words[1].strip()
    return maps


def opaque_mtl(text: str, textures: dict[str, str] | None = None) -> str:
    """The MTL with every material opaque (OpenMVS writes `Tr 1`, which some readers take as fully transparent, so
    `Tr`/`d` are replaced by `Tr 0` and `d 1`) and `map_Kd` renamed per `textures` (old -> new name)."""
    lines = []
    for line in text.splitlines():
        words = line.split(maxsplit=1)
        if words[:1] in (["Tr"], ["d"]):
            continue
        if words[:1] == ["map_Kd"] and textures:
            line = f"map_Kd {textures[words[1].strip()]}"
        lines.append(line)
        if words[:1] == ["newmtl"]:
            lines += ["Tr 0.000000", "d 1.0"]
    return "\n".join(lines) + "\n"


def read_obj_uv(obj: Path) -> tuple[str | None, np.ndarray, dict[str, np.ndarray], int]:
    """`mtllib` name, texture coordinates (T, 2), per material the faces' texture-coordinate indices (F, 3; -1
    where a face has none) and the face count of a triangle OBJ."""
    mtllib, uvs, faces, current = None, [], {}, None
    with obj.open() as f:
        for line in f:
            if line.startswith("vt "):
                uvs.append(line.split()[1:3])
            elif line.startswith("f "):
                corners = line.split()[1:]
                faces.setdefault(current, []).append(
                    [int(c.split("/")[1]) - 1 if "/" in c and c.split("/")[1] else -1 for c in corners[:3]]
                )
            elif line.startswith("usemtl "):
                current = line.split(maxsplit=1)[1].strip()
            elif line.startswith("mtllib "):
                mtllib = line.split(maxsplit=1)[1].strip()
    uv = np.asarray(uvs, dtype=float).reshape(-1, 2)
    per_material = {name: np.asarray(idx, dtype=np.int64).reshape(-1, 3) for name, idx in faces.items()}
    return mtllib, uv, per_material, sum(len(idx) for idx in per_material.values())


def read_obj_faces(obj: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vertex positions (V, 3) and, per material, the faces' vertex indices (F, 3) of a triangle OBJ, grouped like
    `read_obj_uv`."""
    vertices, faces, current = [], {}, None
    with obj.open() as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.setdefault(current, []).append([int(c.split("/")[0]) - 1 for c in line.split()[1:4]])
            elif line.startswith("usemtl "):
                current = line.split(maxsplit=1)[1].strip()
    per_material = {name: np.asarray(idx, dtype=np.int64).reshape(-1, 3) for name, idx in faces.items()}
    return np.asarray(vertices, dtype=float).reshape(-1, 3), per_material


def _is_sentinel(rgb: np.ndarray) -> np.ndarray:
    """Texels (..., 3) within SENTINEL_TOL of magenta in every channel (BGR or RGB alike)."""
    target = np.array([255, 0, 255], dtype=np.int16)
    return (np.abs(rgb.astype(np.int16) - target) < SENTINEL_TOL).all(axis=-1)


def atlas_stats(obj: Path) -> dict[str, Any]:
    """Atlas count and (largest) size, the empty share of all atlas texels, the black share of the used ones, and
    `untextured_face_frac`: faces whose UV-triangle centroid samples the empty colour, or that have no UVs."""
    mtllib, uv, per_material, num_faces = read_obj_uv(obj)
    maps = read_mtl_maps(obj.parent / mtllib) if mtllib else {}
    sizes, texels, empty, black, untextured = [], 0, 0, 0, 0
    for name, idx in per_material.items():
        missing = (idx < 0).any(axis=1)
        untextured += int(missing.sum())
        image = cv2.imread(str(obj.parent / maps[name])) if name in maps else None
        if image is None:
            untextured += int((~missing).sum())
            continue
        h, w = image.shape[:2]
        sizes.append([w, h])
        sentinel = _is_sentinel(image)
        texels += sentinel.size
        empty += int(sentinel.sum())
        black += int(((image < BLACK_MAX).all(axis=-1) & ~sentinel).sum())
        centroid = uv[idx[~missing]].mean(axis=1)
        x = np.clip((centroid[:, 0] * w).astype(int), 0, w - 1)
        y = np.clip(((1.0 - centroid[:, 1]) * h).astype(int), 0, h - 1)  # OBJ v points up
        untextured += int(sentinel[y, x].sum())
    used = texels - empty
    return {
        "atlas_count": len(sizes),
        "atlas_size": max(sizes, key=lambda s: s[0] * s[1]) if sizes else None,
        "atlas_empty_frac": empty / texels if texels else None,
        "black_texel_frac": black / used if used else None,
        "untextured_face_frac": untextured / num_faces if num_faces else None,
        "faces": num_faces,
    }
