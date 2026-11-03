"""Step 6: clean the dense mesh, close its small holes, flatten its walls, floor and ceiling and decimate it,
texture it at full resolution with OpenMVS and fix the atlases, in world W, in
`06_mesh_post/`, and check the final geometry against the dense points."""

import dataclasses
import importlib.metadata
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon.dense.openmvs import TEXTURE_MESH, cuda_device
from recon.io.ply import mesh_stats, read_mesh, read_openmvs_points
from recon.io.tables import read_json, write_json
from recon.io.tools import ToolRunner
from recon.mesh import MeshPostOptions
from recon.mesh.cleanup import cleanup
from recon.mesh.decimate import decimate
from recon.mesh.holes import fill_holes
from recon.mesh.merge import merge_unsupported
from recon.mesh.planes import regularize, write_planes
from recon.mesh.roi import room_bounds
from recon.mesh.texture_fix import fix_textures
from recon.mesh.qa import dense_to_mesh, tau, unsupported_area_frac
from recon.mesh.texturing import atlas_stats, texture_mesh
from recon.pipeline.device import Backend
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s05_dense
from recon.pipeline.stages.s05_dense import inside_frac, retry_warning, tool_rows

log = logging.getLogger(__name__)

STAGE = "mesh_post"
DIR_NAME = "06_mesh_post"
MAX_UNTEXTURED_FRAC = 0.01
MAX_WALL_RMS_FRAC = 0.002  # flatness of every wall with >= 1% of the area after snapping, x diag
MAX_QA_P95_TAU = 2.0  # dense -> mesh p95 distance above this many tau is a warning
UNSUPPORTED_DIST_TAU = 2.0  # mesh area farther than this many tau from every dense point is unsupported

ARTIFACTS = {
    "mesh_merged": "mesh_merged.ply",
    "mesh_clean": "mesh_clean.ply",
    "mesh_holes_filled": "mesh_holes_filled.ply",
    "mesh_regularized": "mesh_regularized.ply",
    "planes": "planes.json",
    "mesh_final_geometry": "mesh_final_geometry.ply",
    "textured": "textured",
    "textured_obj": "textured/mesh.obj",
    "textured_mtl": "textured/mesh.mtl",
    "textured_fixed": "textured_fixed",
    "textured_fixed_obj": "textured_fixed/mesh.obj",
    "textured_fixed_mtl": "textured_fixed/mesh.mtl",
    "report": "reports/mesh_report.json",
}
INPUTS = {"refined": "mesh_refined", "raw": "mesh_raw"}


def _final_report(path: Path, box: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    mesh = read_mesh(path)
    return {**mesh_stats(mesh), "inside_sparse_bbox_frac": inside_frac(np.asarray(mesh.vertices), box)}


def _qa(dense_dir: Path, final: Path, t: float, diag: float) -> dict[str, Any]:
    points = read_openmvs_points(dense_dir / s05_dense.ARTIFACTS["dense_points"])[0]
    distances = dense_to_mesh(points, final)
    qa = {
        "tau": t,
        "dense_to_mesh": distances,
        "p95_frac": distances["p95"] / diag,
        "unsupported_area_frac": unsupported_area_frac(final, points, UNSUPPORTED_DIST_TAU * t),
        "unsupported_area_frac_mono": None,
    }
    mono = dense_dir / s05_dense.FUSION_ARTIFACTS["mono_points"]
    if mono.is_file():  # support from the dense points and the mono depth together
        both = np.concatenate([points, read_openmvs_points(mono)[0]])
        qa["unsupported_area_frac_mono"] = unsupported_area_frac(final, both, UNSUPPORTED_DIST_TAU * t)
    return qa


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
    t = tau(dense["units"], dense["diag"], options.planes.snap_dist)
    dense_points = read_openmvs_points(dense_dir / s05_dense.ARTIFACTS["dense_points"])[0]
    tsdf = dense_dir / s05_dense.FUSION_ARTIFACTS["mesh_tsdf"]
    merge: dict[str, Any] = {"enabled": options.merge.enabled}
    if options.merge.enabled and tsdf.is_file():
        merge.update(merge_unsupported(src, tsdf, dense_points, out / ARTIFACTS["mesh_merged"], t, options.merge))
        cleanup_src = out / ARTIFACTS["mesh_merged"]
    else:
        cleanup_src = src
        merge["enabled"] = False
        shutil.copyfile(src, out / ARTIFACTS["mesh_merged"])
    timings["merge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bounds = room_bounds(dense_points, dense["diag"], options.roi) if options.roi.enabled else None
    cleaned = cleanup(cleanup_src, out / ARTIFACTS["mesh_clean"], options.cleanup, bounds)
    merge["output_faces"] = cleaned.before["faces"]
    timings["cleanup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    holes = fill_holes(out / ARTIFACTS["mesh_clean"], out / ARTIFACTS["mesh_holes_filled"], options.holes)
    timings["holes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    planes_report, planes = regularize(
        out / ARTIFACTS["mesh_holes_filled"], out / ARTIFACTS["mesh_regularized"], t, options.planes
    )
    write_planes(out / ARTIFACTS["planes"], planes)
    if planes_report["enabled"]:
        rms = planes_report["wall_rms_max"]
        planes_report["wall_rms_max_frac"] = rms / dense["diag"] if rms is not None else None
    timings["planes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    decimated = decimate(
        out / ARTIFACTS["mesh_regularized"], out / ARTIFACTS["mesh_final_geometry"], options.decimate
    )
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
    fixed = fix_textures(
        out / ARTIFACTS["textured"],
        out / ARTIFACTS["textured_fixed"],
        dense_dir / s05_dense.ARTIFACTS["dense_points"],
        options.texture_fix,
    )
    timings["texture_fix"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    atlas = atlas_stats(textured.ply)
    fixed["atlas"] = atlas_stats(out / ARTIFACTS["textured_fixed_obj"])
    final = _final_report(out / ARTIFACTS["mesh_final_geometry"], box)
    timings["stats"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qa = _qa(dense_dir, out / ARTIFACTS["mesh_final_geometry"], t, dense["diag"])
    timings["qa"] = time.perf_counter() - t0

    untextured = fixed["atlas"]["untextured_face_frac"]
    if untextured is not None and untextured > MAX_UNTEXTURED_FRAC:
        warnings.append(f"{untextured:.1%} of the faces untextured (> {MAX_UNTEXTURED_FRAC:.0%})")
    if fixed["sentinel_in_faces"]:
        warnings.append(f"texture fix: {fixed['sentinel_in_faces']} empty texels left inside faces")
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
    if planes_report["enabled"]:
        if not planes_report["by_label"]["floor"]:
            warnings.append("planes: no floor found")
        if (planes_report["wall_rms_max_frac"] or 0.0) >= MAX_WALL_RMS_FRAC:
            warnings.append(
                f"planes: a wall has RMS {planes_report['wall_rms_max_frac']:.3%} of diag after snapping"
                f" (>= {MAX_WALL_RMS_FRAC:.1%})"
            )
        if planes_report["flipped_faces"]:
            warnings.append(f"planes: {planes_report['flipped_faces']} faces flipped by snapping")
    if qa["dense_to_mesh"]["p95"] > MAX_QA_P95_TAU * qa["tau"]:
        warnings.append(
            f"dense -> mesh p95 distance {qa['dense_to_mesh']['p95']:.4g} > {MAX_QA_P95_TAU:g} tau"
            f" ({MAX_QA_P95_TAU * qa['tau']:.4g})"
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
        "input_faces": merge.get("input_faces", cleaned.before["faces"]),
        "units": dense["units"],
        "diag": dense["diag"],
        "merge": merge,
        "roi": {"min": bounds[0].tolist(), "max": bounds[1].tolist()} if bounds else None,
        "cleanup": dataclasses.asdict(cleaned),
        "holes": holes,
        "planes": planes_report,
        "decimate": {**decimated, "budget": options.decimate.texture_face_budget},
        "final": final,
        "texture": {
            "resolution_level": options.texture.resolution_level,
            "cuda_device": textured.cuda_device,
            **atlas,
        },
        "texture_fix": fixed,
        "qa": qa,
        "oom_retries": [dataclasses.asdict(r) for r in textured.oom_retries],
        "tools": tool_rows(runner, out),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / ARTIFACTS["report"])
    log.info(
        "mesh %d -> %d faces (cleanup %.1f s, %d hole(s) closed, %d planes, decimate %.1f s); textured in %.1f s"
        " on %s: %d atlas(es) %s, %.2f%% faces untextured (%d filled); dense -> mesh p95 %.2f%% of diag",
        cleaned.before["faces"],
        final["faces"],
        timings["cleanup"],
        holes["filled"],
        len(planes),
        timings["decimate"],
        timings["texture"],
        texture_backend,
        atlas["atlas_count"],
        atlas["atlas_size"],
        100 * (atlas["untextured_face_frac"] or 0.0),
        fixed["filled_faces"],
        100 * qa["p95_frac"],
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "final_faces": final["faces"],
            "untextured_face_frac": untextured,
            "untextured_face_frac_raw": atlas["untextured_face_frac"],
            "atlas_count": atlas["atlas_count"],
            "texture_s": round(timings["texture"], 1),
            "qa_p95_frac": round(qa["p95_frac"], 6),
            "num_planes": len(planes),
            "planar_coverage": round(planes_report.get("coverage", 0.0), 4),
            "wall_rms_max_frac": planes_report.get("wall_rms_max_frac"),
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
