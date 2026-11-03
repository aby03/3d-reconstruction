"""Step 6: clean and decimate the dense mesh, then texture it at full resolution with OpenMVS, in world W, in
`06_mesh_post/`."""

import dataclasses
import importlib.metadata
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon.dense.openmvs import TEXTURE_MESH, cuda_device
from recon.io.ply import mesh_stats, read_mesh
from recon.io.tables import read_json, write_json
from recon.io.tools import ToolRunner
from recon.mesh import MeshPostOptions
from recon.mesh.cleanup import cleanup
from recon.mesh.decimate import decimate
from recon.mesh.texturing import atlas_stats, texture_mesh
from recon.pipeline.device import Backend
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s05_dense
from recon.pipeline.stages.s05_dense import inside_frac, retry_warning, tool_rows

log = logging.getLogger(__name__)

STAGE = "mesh_post"
DIR_NAME = "06_mesh_post"
MAX_UNTEXTURED_FRAC = 0.01

ARTIFACTS = {
    "mesh_clean": "mesh_clean.ply",
    "mesh_final_geometry": "mesh_final_geometry.ply",
    "textured": "textured",
    "textured_obj": "textured/mesh.obj",
    "textured_mtl": "textured/mesh.mtl",
    "report": "reports/mesh_report.json",
}
INPUTS = {"refined": "mesh_refined", "raw": "mesh_raw"}


def _final_report(path: Path, box: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    mesh = read_mesh(path)
    return {**mesh_stats(mesh), "inside_sparse_bbox_frac": inside_frac(np.asarray(mesh.vertices), box)}


def run_mesh_post(
    dense_dir: Path,
    out: Path,
    options: MeshPostOptions | None,
    runner: ToolRunner,
    backend: Backend = "cuda",
    gpu_index: int = 0,
    cpu_fallback: bool = False,
) -> StageOutput:
    """Run step 6 on the mesh of the dense stage dir `dense_dir`, writing into the stage dir `out`. TextureMesh
    runs through `runner` in the dense stage's OpenMVS folder; `cpu_fallback`: after a CUDA OOM it runs again on
    the CPU."""
    options = options or MeshPostOptions()
    timings, warnings = {}, []

    t0 = time.perf_counter()
    dense = read_json(dense_dir / s05_dense.ARTIFACTS["report"])
    box = (np.array(dense["sparse_bbox"]["min"]), np.array(dense["sparse_bbox"]["max"]))
    src_rel = s05_dense.ARTIFACTS[INPUTS[options.input]]
    src = dense_dir / src_rel
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cleaned = cleanup(src, out / ARTIFACTS["mesh_clean"], options.cleanup)
    timings["cleanup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    decimated = decimate(out / ARTIFACTS["mesh_clean"], out / ARTIFACTS["mesh_final_geometry"], options.decimate)
    timings["decimate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    device = cuda_device(backend, gpu_index)
    textured = texture_mesh(
        runner,
        dense_dir / s05_dense.WORK_DIR,
        out / ARTIFACTS["mesh_final_geometry"],
        out / ARTIFACTS["textured"],
        options.texture,
        device,
        cpu_fallback,
    )
    timings["texture"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    atlas = atlas_stats(textured.ply)
    final = _final_report(out / ARTIFACTS["mesh_final_geometry"], box)
    timings["stats"] = time.perf_counter() - t0

    if atlas["untextured_face_frac"] is not None and atlas["untextured_face_frac"] > MAX_UNTEXTURED_FRAC:
        warnings.append(
            f"{atlas['untextured_face_frac']:.1%} of the faces untextured (> {MAX_UNTEXTURED_FRAC:.0%})"
        )
    if final["largest_component_frac"] < s05_dense.MIN_LARGEST_COMPONENT_FRAC:
        warnings.append(
            f"final mesh: largest component has {final['largest_component_frac']:.1%} of the faces"
            f" ({final['components']} components)"
        )
    if final["inside_sparse_bbox_frac"] < s05_dense.MIN_INSIDE_FRAC:
        warnings.append(
            f"final mesh: {final['inside_sparse_bbox_frac']:.1%} of the vertices inside the sparse bbox"
            f" (< {s05_dense.MIN_INSIDE_FRAC:.0%})"
        )
    if atlas["faces"] != final["faces"]:
        warnings.append(f"{TEXTURE_MESH} wrote {atlas['faces']} faces for {final['faces']} input faces")
    warnings += [retry_warning(r) for r in textured.oom_retries]
    for warning in warnings:
        log.warning(warning)

    texture_backend: Backend = "cpu" if textured.cuda_device < 0 else backend
    openmvs_version = runner.version(TEXTURE_MESH)
    pymeshlab_version = importlib.metadata.version("pymeshlab")
    report: dict[str, Any] = {
        "input": f"{s05_dense.DIR_NAME}/{src_rel}",
        "input_faces": cleaned.before["faces"],
        "units": dense["units"],
        "diag": dense["diag"],
        "cleanup": dataclasses.asdict(cleaned),
        "decimate": {**decimated, "budget": options.decimate.texture_face_budget},
        "final": final,
        "texture": {
            "resolution_level": options.texture.resolution_level,
            "cuda_device": textured.cuda_device,
            **atlas,
        },
        "oom_retries": [dataclasses.asdict(r) for r in textured.oom_retries],
        "tools": tool_rows(runner, out),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / ARTIFACTS["report"])
    log.info(
        "mesh %d -> %d faces (cleanup %.1f s, decimate %.1f s); textured in %.1f s on %s: %d atlas(es) %s,"
        " %.2f%% faces untextured",
        cleaned.before["faces"],
        final["faces"],
        timings["cleanup"],
        timings["decimate"],
        timings["texture"],
        texture_backend,
        atlas["atlas_count"],
        atlas["atlas_size"],
        100 * (atlas["untextured_face_frac"] or 0.0),
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "final_faces": final["faces"],
            "untextured_face_frac": atlas["untextured_face_frac"],
            "atlas_count": atlas["atlas_count"],
            "texture_s": round(timings["texture"], 1),
            "backend": backend,
        },
        warnings=list(warnings),
        backends={"texture": texture_backend},
        tools={"OpenMVS": openmvs_version, "pymeshlab": pymeshlab_version},
    )


@register_stage
class MeshPostStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s05_dense.STAGE,)
    config_section = "mesh_post"

    def hash_inputs(self) -> dict[str, Any]:
        return {"device": self.ctx.config.device}

    def run(self) -> StageOutput:
        device = self.ctx.config.device
        bin_dir = self.ctx.config.dense.openmvs.bin_dir
        runner = ToolRunner(self.ctx.run_dir / "logs" / STAGE, bin_dir, device.gpu_index)
        return run_mesh_post(
            self.ctx.stage_dir(s05_dense.STAGE),
            self.out,
            self.options,
            runner,
            self.ctx.backend("dense"),
            device.gpu_index,
            cpu_fallback=device.preference == "auto",
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
