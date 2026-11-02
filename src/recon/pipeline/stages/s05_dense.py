"""Step 5: undistort the aligned model, densify it with OpenMVS and mesh the dense cloud, in world W, in `05_dense/`."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.dense import DenseOptions
from recon.dense.openmvs import DENSIFY, cuda_device, densify, import_colmap, reconstruct_mesh
from recon.dense.undistort import undistort
from recon.io.colmap import camera_intrinsics, camera_keys, colmap_log
from recon.io.ply import mesh_stats, read_mesh, read_openmvs_points, write_points
from recon.io.tables import read_json, write_json
from recon.io.tools import ToolRunner
from recon.pipeline.device import Backend
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s03_sfm, s04_align
from recon.report import plots
from recon.sfm.quality import camera_centres

log = logging.getLogger(__name__)

STAGE = "dense"
DIR_NAME = "05_dense"
WORK_DIR = "openmvs"
MIN_INSIDE_FRAC = 0.99
MIN_LARGEST_COMPONENT_FRAC = 0.9
MIN_DENSE_POINTS = 1000
FLOOR_BAND_FRAC = 0.01  # floor band: |z| below this fraction of the scene diagonal

ARTIFACTS = {
    "undistorted": "undistorted",
    "undistorted_model": "undistorted/sparse",
    "undistorted_images": "undistorted/images",
    "scene": f"{WORK_DIR}/scene.mvs",
    "scene_dense": f"{WORK_DIR}/scene_dense.mvs",
    "scene_dense_points": f"{WORK_DIR}/scene_dense.ply",
    "mesh_openmvs": f"{WORK_DIR}/scene_dense_mesh.ply",
    "dense_points": "dense_points.ply",
    "mesh_raw": "mesh_raw.ply",
    "report": "reports/dense_report.json",
    "plot_height": "reports/plots/dense_height_hist.png",
    "plot_topdown": "reports/plots/dense_topdown.png",
}


def _inside_frac(xyz: np.ndarray, box: tuple[np.ndarray, np.ndarray]) -> float:
    """Share of `xyz` inside the axis-aligned box (lo, hi), bounds included."""
    lo, hi = box
    return float(((xyz >= lo) & (xyz <= hi)).all(axis=1).mean()) if len(xyz) else 0.0


def _disk_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20, 1)


def _cloud_stats(xyz: np.ndarray, box: tuple[np.ndarray, np.ndarray], diag: float) -> dict[str, Any]:
    floor = xyz[np.abs(xyz[:, 2]) < FLOOR_BAND_FRAC * diag, 2]
    p1, p99 = np.percentile(xyz, [1, 99], axis=0) if len(xyz) else (np.zeros(3), np.zeros(3))
    return {
        "bbox": {"p1": p1.tolist(), "p99": p99.tolist()},
        "inside_sparse_bbox_frac": _inside_frac(xyz, box),
        "floor_band_frac": float(len(floor) / len(xyz)) if len(xyz) else 0.0,
        "floor_band_median_z": float(np.median(floor)) if len(floor) else None,
    }


def run_dense(
    align_dir: Path,
    sfm_dir: Path,
    out: Path,
    options: DenseOptions | None,
    runner: ToolRunner,
    backend: Backend = "cuda",
    gpu_index: int = 0,
) -> StageOutput:
    """Run step 5 on the aligned model of the align stage dir `align_dir` (images from the sfm stage dir
    `sfm_dir`), writing into the stage dir `out`; the OpenMVS tools run through `runner`."""
    options = options or DenseOptions()
    timings, warnings = {}, []

    t0 = time.perf_counter()
    model_dir = align_dir / s04_align.ARTIFACTS["model"]
    rec = pycolmap.Reconstruction(model_dir)
    names = sorted(rec.images[i].name for i in rec.reg_image_ids())
    centres = np.array(list(camera_centres(rec).values()))
    points = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    everything = np.concatenate([points, centres])
    box = (everything.min(axis=0), everything.max(axis=0))
    lo, hi = np.percentile(points, [2, 98], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    units = read_json(align_dir / s04_align.ARTIFACTS["transform"])["units"]
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    undistorted_dir = out / ARTIFACTS["undistorted"]
    with colmap_log(runner.log_dir / "colmap.log"):
        urec = undistort(model_dir, sfm_dir / s03_sfm.IMAGE_ROOT, names, undistorted_dir, options.undistort)
    keys = camera_keys(urec)
    cameras = {keys[cid]: camera_intrinsics(cam) for cid, cam in sorted(urec.cameras.items()) if cid in keys}
    long_side = max(max(c["width"], c["height"]) for c in cameras.values())
    timings["undistort"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    work_dir = out / WORK_DIR
    import_colmap(runner, work_dir, undistorted_dir)
    timings["import"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    device = cuda_device(backend, gpu_index)
    dense = densify(runner, work_dir, options.openmvs, device, long_side)
    timings["densify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = reconstruct_mesh(runner, work_dir, options.mesh, device, options.openmvs.crop_to_roi)
    timings["mesh"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    xyz, normals, rgb, _ = read_openmvs_points(dense.ply)
    write_points(out / ARTIFACTS["dense_points"], xyz, normals, rgb)
    shutil.copyfile(mesh.ply, out / ARTIFACTS["mesh_raw"])
    raw_mesh = read_mesh(out / ARTIFACTS["mesh_raw"])
    mesh_raw = {**mesh_stats(raw_mesh), "inside_sparse_bbox_frac": _inside_frac(np.asarray(raw_mesh.vertices), box)}
    dense_stats = _cloud_stats(xyz, box, diag)
    plots.plot_height_histogram(xyz[:, 2], 0.0, centres[:, 2], out / ARTIFACTS["plot_height"])
    plots.plot_trajectory_topdown(centres, xyz, out / ARTIFACTS["plot_topdown"], axes="xy")
    timings["outputs"] = time.perf_counter() - t0

    if len(xyz) < MIN_DENSE_POINTS:
        warnings.append(f"only {len(xyz)} dense points")
    if dense.num_points != len(xyz):
        warnings.append(f"{DENSIFY} logged {dense.num_points} points, {dense.ply.name} has {len(xyz)}")
    for name, stats in (("dense cloud", dense_stats), ("raw mesh", mesh_raw)):
        frac = stats["inside_sparse_bbox_frac"]
        if frac < MIN_INSIDE_FRAC:
            warnings.append(f"{name}: {frac:.1%} of the vertices inside the sparse bbox (< {MIN_INSIDE_FRAC:.0%})")
    if mesh_raw["largest_component_frac"] < MIN_LARGEST_COMPONENT_FRAC:
        warnings.append(
            f"raw mesh: largest component has {mesh_raw['largest_component_frac']:.1%} of the faces"
            f" ({mesh_raw['components']} components)"
        )
    for warning in warnings:
        log.warning(warning)

    tools = {
        r.tool: {
            "duration_s": r.duration_s,
            "peak_vram_mb": r.peak_vram_mb,
            "returncode": r.returncode,
            "log": os.path.relpath(r.log_path, out.parent),
        }
        for r in runner.results
    }
    vram = [r.peak_vram_mb for r in runner.results if r.peak_vram_mb is not None]
    openmvs_version = runner.version(DENSIFY)
    o = options.openmvs
    report: dict[str, Any] = {
        "input_model": f"{s04_align.DIR_NAME}/{s04_align.ARTIFACTS['model']}",
        "num_images": len(names),
        "units": units,
        "undistorted": cameras,
        "backend": backend,
        "cuda_device": device,
        "openmvs_version": openmvs_version,
        "densify": {
            "resolution_level": o.resolution_level,
            "effective_level": dense.effective_level,
            "effective_long_side": dense.effective_long_side,
            "number_views": o.number_views,
            "number_views_fuse": o.number_views_fuse,
            "iters": o.iters,
            "geometric_iters": o.geometric_iters,
            "crop_to_roi": o.crop_to_roi,
        },
        "num_dense_points": len(xyz),
        "sparse_bbox": {"min": box[0].tolist(), "max": box[1].tolist()},
        "diag": diag,
        "dense": dense_stats,
        "mesh_raw": mesh_raw,
        "tools": tools,
        "peak_vram_mb": max(vram) if vram else None,
        "disk_mb": {"undistorted": _disk_mb(undistorted_dir), "openmvs": _disk_mb(work_dir)},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / ARTIFACTS["report"])
    log.info(
        "%d dense points (%.1f%% inside the sparse bbox) in %.1f s on %s; raw mesh %d faces, largest component %.1f%%",
        len(xyz),
        100 * dense_stats["inside_sparse_bbox_frac"],
        timings["densify"],
        backend,
        mesh_raw["faces"],
        100 * mesh_raw["largest_component_frac"],
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "num_dense_points": len(xyz),
            "mesh_raw_faces": mesh_raw["faces"],
            "inside_sparse_bbox_frac": dense_stats["inside_sparse_bbox_frac"],
            "largest_component_frac": mesh_raw["largest_component_frac"],
            "densify_s": round(timings["densify"], 1),
            "peak_vram_mb": report["peak_vram_mb"],
            "backend": backend,
        },
        warnings=list(warnings),
        backends={"densify": backend, "mesh": backend},
        tools={"OpenMVS": openmvs_version, "pycolmap": pycolmap.__version__},
    )


@register_stage
class DenseStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s04_align.STAGE,)
    config_section = "dense"

    def hash_inputs(self) -> dict[str, Any]:
        return {"device": self.ctx.config.device}

    def run(self) -> StageOutput:
        gpu_index = self.ctx.config.device.gpu_index
        runner = ToolRunner(self.ctx.run_dir / "logs" / STAGE, self.options.openmvs.bin_dir, gpu_index)
        return run_dense(
            self.ctx.stage_dir(s04_align.STAGE),
            self.ctx.stage_dir(s03_sfm.STAGE),
            self.out,
            self.options,
            runner,
            self.ctx.backend("dense"),
            gpu_index,
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
