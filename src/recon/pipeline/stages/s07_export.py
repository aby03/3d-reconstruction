"""Step 7: export the textured mesh as a Y-up GLB with embedded textures and a Z-up OBJ, the geometry as a
vertex-colour PLY and an STL, the point clouds and the cameras (COLMAP text, nerfstudio `transforms.json`, TUM
trajectories, intrinsics), plus `metadata.json`, in `07_export/`. Every file is reloaded and checked."""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import trimesh

from recon.export import ExportOptions
from recon.export.cameras import transforms_json, validate_transforms, write_colmap_text
from recon.export.gltf import validate_glb, write_glb
from recon.export.manifest import file_entry, write_metadata
from recon.export.preview import write_preview_glb
from recon.export.obj import OBJ_NAME, validate_obj, write_obj
from recon.export.ply import validate_stl, validate_vertex_color_ply, write_stl, write_vertex_color_ply
from recon.io.ply import read_mesh
from recon.io.tables import read_json
from recon.mesh.planes import LABELS as PLANE_LABELS
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s03_sfm, s04_align, s05_dense, s06_mesh_post

log = logging.getLogger(__name__)

STAGE = "export"
DIR_NAME = "07_export"
REPO_ROOT = Path(__file__).resolve().parents[4]

ARTIFACTS = {
    "glb": "model.glb",
    "glb_preview": "model_preview.glb",
    "obj": f"obj/{OBJ_NAME}",
    "mtl": "obj/model.mtl",
    "textures": "obj/textures",
    "vertex_color_ply": "model_vertex_color.ply",
    "stl": "model.stl",
    "sparse_ply": "point_clouds/sparse.ply",
    "dense_ply": "point_clouds/dense.ply",
    "colmap": "cameras/colmap",
    "colmap_undistorted": "cameras/colmap_undistorted",
    "transforms": "cameras/transforms.json",
    "trajectory": "cameras/trajectory_tum",
    "intrinsics": "cameras/intrinsics.json",
    "metadata": "metadata.json",
}
WORLD = {"up": "+Z", "gravity": [0.0, 0.0, -1.0], "floor_z": 0.0}


def git_version(repo: Path = REPO_ROOT) -> str | None:
    """Short commit hash of `repo`, None outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copyfile(src, dst)
    return dst


def _dir_entries(out: Path, rel: str, up: str | None) -> dict[str, dict[str, Any]]:
    """`files` entries of every file under the export folder `out / rel`, keyed by their path relative to `out`."""
    return {
        path.relative_to(out).as_posix(): file_entry(path, up)
        for path in sorted((out / rel).rglob("*"))
        if path.is_file()
    }


def run_export(
    mesh_post_dir: Path,
    align_dir: Path,
    dense_dir: Path,
    sfm_dir: Path,
    out: Path,
    options: ExportOptions | None,
    scene: str,
    run_id: str,
    sequence_ids: list[str],
    pipeline_version: str | None,
) -> StageOutput:
    """Run step 7 on the textured mesh of the mesh_post stage dir `mesh_post_dir` (units, cameras and sparse points
    from the align stage dir `align_dir`, dense points and undistorted cameras from `dense_dir`, intrinsics from
    `sfm_dir`), writing into the stage dir `out`. A file that fails its check raises ValueError."""
    options = options or ExportOptions()
    timings = {}

    t0 = time.perf_counter()
    mesh_report = read_json(mesh_post_dir / s06_mesh_post.ARTIFACTS["report"])
    faces, diag = mesh_report["final"]["faces"], mesh_report["diag"]
    geometry = mesh_post_dir / s06_mesh_post.ARTIFACTS["mesh_final_geometry"]
    textured_obj = mesh_post_dir / s06_mesh_post.ARTIFACTS["textured_fixed_obj"]
    bounds_w = np.asarray(read_mesh(geometry).bounds)
    units = read_json(align_dir / s04_align.ARTIFACTS["transform"])["units"]
    num_cameras = pycolmap.Reconstruction(align_dir / s04_align.ARTIFACTS["model"]).num_reg_images()
    labels = [p["label"] for p in read_json(mesh_post_dir / s06_mesh_post.ARTIFACTS["planes"])]
    num_planes = {label: labels.count(label) for label in PLANE_LABELS}
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    glb = out / ARTIFACTS["glb"]
    write_glb(textured_obj, glb, options.glb)
    glb_info = validate_glb(glb, faces, bounds_w, diag)
    preview_info = None
    if options.preview_faces:
        preview = out / ARTIFACTS["glb_preview"]
        preview_faces = write_preview_glb(textured_obj, preview, options.preview_faces, options.glb)
        preview_info = validate_glb(preview, preview_faces, bounds_w, diag, tolerance=0.01)
    timings["glb"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    obj = out / ARTIFACTS["obj"]
    write_obj(mesh_post_dir / s06_mesh_post.ARTIFACTS["textured_fixed"], obj.parent, units)
    obj_info = validate_obj(obj, faces, bounds_w, diag)
    timings["obj"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vertex_color = out / ARTIFACTS["vertex_color_ply"]
    write_vertex_color_ply(textured_obj, geometry, vertex_color)
    vertex_color_info = validate_vertex_color_ply(vertex_color, faces, bounds_w, diag)
    stl = out / ARTIFACTS["stl"]
    write_stl(geometry, stl)
    stl_info = validate_stl(stl, faces, bounds_w, diag)
    timings["ply_stl"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _copy(align_dir / s04_align.ARTIFACTS["points"], out / ARTIFACTS["sparse_ply"])
    _copy(dense_dir / s05_dense.ARTIFACTS["dense_points"], out / ARTIFACTS["dense_ply"])
    timings["point_clouds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    undistorted = dense_dir / s05_dense.ARTIFACTS["undistorted_model"]
    write_colmap_text(align_dir / s04_align.ARTIFACTS["model"], out / ARTIFACTS["colmap"])
    write_colmap_text(undistorted, out / ARTIFACTS["colmap_undistorted"])
    transforms = out / ARTIFACTS["transforms"]
    transforms_json(undistorted, dense_dir / s05_dense.ARTIFACTS["undistorted_images"], transforms)
    transforms_info = validate_transforms(transforms, undistorted)
    _copy(align_dir / s04_align.ARTIFACTS["trajectory"], out / ARTIFACTS["trajectory"])
    _copy(sfm_dir / s03_sfm.ARTIFACTS["intrinsics"], out / ARTIFACTS["intrinsics"])
    timings["cameras"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    files = {
        ARTIFACTS["glb"]: file_entry(glb, "+Y", glb_info),
        ARTIFACTS["obj"]: file_entry(obj, "+Z", obj_info),
        ARTIFACTS["vertex_color_ply"]: file_entry(vertex_color, "+Z", vertex_color_info),
        ARTIFACTS["stl"]: file_entry(stl, "+Z", stl_info),
    }
    if preview_info:
        files[ARTIFACTS["glb_preview"]] = file_entry(out / ARTIFACTS["glb_preview"], "+Y", preview_info)
    for key in ("sparse_ply", "dense_ply", "transforms"):
        files[ARTIFACTS[key]] = file_entry(out / ARTIFACTS[key], "+Z")
    for key in ("colmap", "colmap_undistorted", "trajectory"):
        files |= _dir_entries(out, ARTIFACTS[key], "+Z")
    files[ARTIFACTS["intrinsics"]] = file_entry(out / ARTIFACTS["intrinsics"])
    write_metadata(
        out / ARTIFACTS["metadata"],
        scene=scene,
        run_id=run_id,
        sequence_ids=sequence_ids,
        pipeline_version=pipeline_version,
        units=units,
        world=WORLD,
        files=files,
        bbox_world=bounds_w.tolist(),
        num_cameras=num_cameras,
        num_planes=num_planes,
        transforms_check=transforms_info,
    )
    timings["metadata"] = time.perf_counter() - t0
    log.info(
        "model.glb %.1f MB (%d faces, %d vertices, %d texture(s)); transforms.json re-projects within %.2f px;"
        " times %s",
        glb_info["bytes"] / 2**20,
        glb_info["faces"],
        glb_info["vertices"],
        len(glb_info["textures"]),
        transforms_info["max_px"],
        ", ".join(f"{k} {v:.1f} s" for k, v in timings.items()),
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "glb_mb": round(glb_info["bytes"] / 2**20, 1),
            "glb_faces": glb_info["faces"],
            "glb_vertices": glb_info["vertices"],
            "texture_count": len(glb_info["textures"]),
            "files": len(files),
            "transforms_max_px": round(transforms_info["max_px"], 3),
        },
        tools={"trimesh": trimesh.__version__, "pycolmap": pycolmap.__version__},
    )


@register_stage
class ExportStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s06_mesh_post.STAGE,)
    config_section = "export"

    def run(self) -> StageOutput:
        return run_export(
            self.ctx.stage_dir(s06_mesh_post.STAGE),
            self.ctx.stage_dir(s04_align.STAGE),
            self.ctx.stage_dir(s05_dense.STAGE),
            self.ctx.stage_dir(s03_sfm.STAGE),
            self.out,
            self.options,
            self.ctx.scene_dir.name,
            self.ctx.run_dir.name,
            self.ctx.sequence_ids,
            git_version(),
        )

    def expected_artifacts(self) -> list[Path]:
        skip = {"glb_preview"} if not self.options.preview_faces else set()
        return [self.out / rel for key, rel in ARTIFACTS.items() if key not in skip]
