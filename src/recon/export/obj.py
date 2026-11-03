"""Wavefront OBJ copy of the textured mesh: Z up (world W), renamed files, an opaque MTL, and a check of the copy."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from recon.mesh.texturing import TEXTURED_OBJ, opaque_mtl, read_mtl_maps

OBJ_NAME = "model.obj"
MTL_NAME = "model.mtl"
TEXTURE_DIR = "textures"


def read_mtllib(obj: Path) -> str:
    """The `mtllib` file name of an OBJ (read up to the first face)."""
    with obj.open() as f:
        for line in f:
            if line.startswith("mtllib "):
                return line.split(maxsplit=1)[1].strip()
            if line.startswith("f "):
                break
    raise ValueError(f"{obj.name} has no mtllib")


def write_obj(textured_dir: Path, out_dir: Path, units: str) -> dict[str, Any]:
    """Copy OpenMVS's textured mesh in `textured_dir` to `out_dir/model.obj`, `model.mtl` and
    `textures/model_NN.<ext>`, with an opaque MTL and a header comment naming the frame and units. Returns
    {source texture name: copied path relative to `out_dir`}."""
    src = textured_dir / TEXTURED_OBJ
    mtl = textured_dir / read_mtllib(src)
    textures: dict[str, str] = {}
    for name in read_mtl_maps(mtl).values():
        if name not in textures:
            textures[name] = f"{TEXTURE_DIR}/model_{len(textures):02d}{Path(name).suffix.lower()}"
    (out_dir / TEXTURE_DIR).mkdir(parents=True, exist_ok=True)
    for name, rel in textures.items():
        shutil.copyfile(textured_dir / name, out_dir / rel)
    (out_dir / MTL_NAME).write_text(opaque_mtl(mtl.read_text(), textures))
    with src.open() as f_in, (out_dir / OBJ_NAME).open("w") as f_out:
        f_out.write(f"# world W: +Z up, units {units}\n")
        for line in f_in:
            f_out.write(f"mtllib {MTL_NAME}\n" if line.startswith("mtllib ") else line)
    return {"textures": textures}


def validate_obj(path: Path, faces: int, bounds_w: np.ndarray, diag: float) -> dict[str, Any]:
    """Check the OBJ copy `path`: its MTL and every texture resolve, and it reloads with `faces` faces and the W
    bounds `bounds_w` (tolerance 1e-4 · `diag`). Raises ValueError on the first failure."""
    mtl = path.parent / read_mtllib(path)
    if not mtl.is_file():
        raise ValueError(f"{path.name}: {mtl.name} missing")
    maps = read_mtl_maps(mtl)
    if missing := [name for name in maps.values() if not (path.parent / name).is_file()]:
        raise ValueError(f"{mtl.name}: textures missing: {', '.join(missing)}")
    scene = trimesh.load(path, force="scene", process=False)
    geometries = list(scene.geometry.values())
    num_faces = sum(len(g.faces) for g in geometries)
    if num_faces != faces:
        raise ValueError(f"{path.name}: {num_faces} faces, the source has {faces}")
    vertices = np.concatenate([np.asarray(g.vertices) for g in geometries])
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    if np.abs(bounds - np.asarray(bounds_w)).max() > 1e-4 * diag:
        raise ValueError(f"{path.name}: bounds {bounds.tolist()} != source bounds {np.asarray(bounds_w).tolist()}")
    if any(getattr(g.visual.material, "image", None) is None for g in geometries):
        raise ValueError(f"{path.name}: a material's texture did not load")
    textures = []
    for name in dict.fromkeys(maps.values()):
        with Image.open(path.parent / name) as image:
            textures.append(list(image.size))
    return {"faces": num_faces, "vertices": len(vertices), "textures": textures, "bytes": path.stat().st_size}
