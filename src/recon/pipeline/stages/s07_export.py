"""Step 7: export the textured mesh as a Y-up GLB with embedded textures and a Z-up OBJ, plus `metadata.json`, in
`07_export/`. Every file is reloaded and checked."""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import trimesh

from recon.export import ExportOptions
from recon.export.gltf import validate_glb, write_glb
from recon.export.manifest import file_entry, write_metadata
from recon.export.obj import OBJ_NAME, validate_obj, write_obj
from recon.io.ply import read_mesh
from recon.io.tables import read_json
from recon.mesh.planes import LABELS as PLANE_LABELS
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s04_align, s06_mesh_post

log = logging.getLogger(__name__)

STAGE = "export"
DIR_NAME = "07_export"
REPO_ROOT = Path(__file__).resolve().parents[4]

ARTIFACTS = {
    "glb": "model.glb",
    "obj": f"obj/{OBJ_NAME}",
    "mtl": "obj/model.mtl",
    "textures": "obj/textures",
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


def run_export(
    mesh_post_dir: Path,
    align_dir: Path,
    out: Path,
    options: ExportOptions | None,
    scene: str,
    run_id: str,
    sequence_ids: list[str],
    pipeline_version: str | None,
) -> StageOutput:
    """Run step 7 on the textured mesh of the mesh_post stage dir `mesh_post_dir` (units and cameras from the align
    stage dir `align_dir`), writing into the stage dir `out`. A file that fails its check raises ValueError."""
    options = options or ExportOptions()
    timings = {}

    t0 = time.perf_counter()
    mesh_report = read_json(mesh_post_dir / s06_mesh_post.ARTIFACTS["report"])
    faces, diag = mesh_report["final"]["faces"], mesh_report["diag"]
    final = read_mesh(mesh_post_dir / s06_mesh_post.ARTIFACTS["mesh_final_geometry"])
    bounds_w = np.asarray(final.bounds)
    units = read_json(align_dir / s04_align.ARTIFACTS["transform"])["units"]
    num_cameras = pycolmap.Reconstruction(align_dir / s04_align.ARTIFACTS["model"]).num_reg_images()
    labels = [p["label"] for p in read_json(mesh_post_dir / s06_mesh_post.ARTIFACTS["planes"])]
    num_planes = {label: labels.count(label) for label in PLANE_LABELS}
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    glb = out / ARTIFACTS["glb"]
    write_glb(mesh_post_dir / s06_mesh_post.ARTIFACTS["textured_obj"], glb, options.glb)
    glb_info = validate_glb(glb, faces, bounds_w, diag)
    timings["glb"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    obj = out / ARTIFACTS["obj"]
    write_obj(mesh_post_dir / s06_mesh_post.ARTIFACTS["textured"], obj.parent, units)
    obj_info = validate_obj(obj, faces, bounds_w, diag)
    timings["obj"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_metadata(
        out / ARTIFACTS["metadata"],
        scene=scene,
        run_id=run_id,
        sequence_ids=sequence_ids,
        pipeline_version=pipeline_version,
        units=units,
        world=WORLD,
        files={
            ARTIFACTS["glb"]: file_entry(glb, "+Y", glb_info),
            ARTIFACTS["obj"]: file_entry(obj, "+Z", obj_info),
        },
        bbox_world=bounds_w.tolist(),
        num_cameras=num_cameras,
        num_planes=num_planes,
    )
    timings["metadata"] = time.perf_counter() - t0
    log.info(
        "model.glb %.1f MB (%d faces, %d texture(s)) in %.1f s; obj in %.1f s",
        glb_info["bytes"] / 2**20,
        glb_info["faces"],
        len(glb_info["textures"]),
        timings["glb"],
        timings["obj"],
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "glb_mb": round(glb_info["bytes"] / 2**20, 1),
            "glb_faces": glb_info["faces"],
            "texture_count": len(glb_info["textures"]),
        },
        tools={"trimesh": trimesh.__version__},
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
            self.out,
            self.options,
            self.ctx.scene_dir.name,
            self.ctx.run_dir.name,
            self.ctx.sequence_ids,
            git_version(),
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
