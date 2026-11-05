"""A light copy of the textured mesh for quick viewing: quadric decimation that keeps the UVs, same atlases."""

import shutil
import tempfile
from pathlib import Path

import pymeshlab

from recon.export import GlbOptions
from recon.export.gltf import write_glb
from recon.mesh.texturing import read_mtl_maps


def write_preview_glb(textured_obj: Path, out: Path, faces: int, options: GlbOptions | None = None) -> int:
    """Write to the GLB `out` the textured OBJ `textured_obj` decimated to about `faces` faces (copied as is when it
    has fewer). Returns the face count written."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(textured_obj))
        count = ms.current_mesh().face_number()
        if count > faces:
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=faces, preserveboundary=True, preservenormal=True
            )
        ms.save_current_mesh(str(work / "mesh.obj"), save_textures=False)
        for atlas in read_mtl_maps(textured_obj.with_suffix(".mtl")).values():
            shutil.copyfile(textured_obj.parent / atlas, work / atlas)
        written = ms.current_mesh().face_number()
        write_glb(work / "mesh.obj", out, options)
    return written
