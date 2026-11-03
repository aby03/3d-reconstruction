"""Step 5: undistort the aligned model, densify it with OpenMVS, mesh, refine and preview-texture the mesh, in
world W, in `05_dense/`."""

import dataclasses
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pycolmap

from recon.dense import DenseOptions
from recon.dense.openmvs import (
    DENSIFY,
    REFINED_PLY,
    REFINE_MESH,
    DensifyResult,
    MeshResult,
    OomRetry,
    cuda_device,
    densify,
    import_colmap,
    reconstruct_mesh,
    refine_mesh,
    texture_preview,
)
from recon.dense.filter import filter_points
from recon.dense.mono import MonoDepth, MonoOptions, ViewFit, fetch_model, fit_view, fitted_depth
from recon.dense.undistort import undistort
from recon.io.colmap import camera_intrinsics, camera_keys, colmap_log
from recon.io.ply import mesh_stats, read_mesh, read_openmvs_points, select_openmvs_points, write_points
from recon.io.tables import read_json, write_csv, write_json, write_npz
from recon.io.tools import ToolRunner
from recon.mesh.cleanup import cleanup
from recon.pipeline.device import Backend
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s03_sfm, s04_align
from recon.report import plots
from recon.sfm.quality import camera_centres

log = logging.getLogger(__name__)

STAGE = "dense"
DIR_NAME = "05_dense"
WORK_DIR = "openmvs"
FILTERED_PLY = "scene_dense_filtered.ply"  # the dense cloud after outlier removal, with views
CROPPED_PLY = "scene_dense_mesh_crop.ply"  # the graph-cut mesh cut to the room box, RefineMesh's input
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
    "mesh_refined": "mesh_refined.ply",
    "report": "reports/dense_report.json",
    "plot_height": "reports/plots/dense_height_hist.png",
    "plot_topdown": "reports/plots/dense_topdown.png",
}
# written only when refine ran / the preview or the mono depth is enabled
OPTIONAL_ARTIFACTS = {
    "mesh_refine_openmvs": f"{WORK_DIR}/{REFINED_PLY}",
    "preview": "preview/mesh.obj",
}
MONO_ARTIFACTS = {
    "mono": "mono",
    "mono_fit": "reports/mono_fit.csv",
    "plot_mono_fit": "reports/plots/mono_fit.png",
    "plot_mono_example": "reports/plots/mono_example.png",
}
PREVIEW_DIR = "preview"
MONO_COLUMNS = ["image", "points", "inliers", "inlier_ratio", "median_rel_residual", "median_rel_residual_all",
                "scale", "shift", "fitted", "reason"]


def inside_frac(xyz: np.ndarray, box: tuple[np.ndarray, np.ndarray]) -> float:
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
        "inside_sparse_bbox_frac": inside_frac(xyz, box),
        "floor_band_frac": float(len(floor) / len(xyz)) if len(xyz) else 0.0,
        "floor_band_median_z": float(np.median(floor)) if len(floor) else None,
    }


def _mesh_report(mesh, box: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    return {**mesh_stats(mesh), "inside_sparse_bbox_frac": inside_frac(np.asarray(mesh.vertices), box)}


def _preview_report(obj: Path) -> dict[str, Any]:
    faces = uvs = 0
    with obj.open() as f:
        for line in f:
            faces += line.startswith("f ")
            uvs += line.startswith("vt ")
    atlases = sorted(p for p in obj.parent.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    size = list(cv2.imread(str(atlases[0])).shape[1::-1]) if atlases else None
    return {"faces": faces, "texture_coords": uvs, "textures": len(atlases), "atlas_size": size}


def tool_rows(runner: ToolRunner, out: Path) -> dict[str, dict[str, Any]]:
    """One row per tool; a retried tool sums its durations and keeps the worst VRAM peak and the last status."""
    rows: dict[str, dict[str, Any]] = {}
    for r in runner.results:
        row = rows.setdefault(r.tool, {"calls": 0, "duration_s": 0.0, "peak_vram_mb": None})
        row["calls"] += 1
        row["duration_s"] = round(row["duration_s"] + r.duration_s, 3)
        if r.peak_vram_mb is not None:
            row["peak_vram_mb"] = max(row["peak_vram_mb"] or 0, r.peak_vram_mb)
        row["returncode"] = r.returncode
        row["log"] = os.path.relpath(r.log_path, out.parent)
    return rows


def retry_warning(retry: OomRetry) -> str:
    if retry.cpu_fallback:
        return f"oom_cpu_fallback: {retry.tool}"
    return f"oom_retry: {retry.tool} at level {retry.from_level} -> {retry.to_level}"


def _backend(result: DensifyResult | MeshResult | None, backend: Backend) -> Backend:
    """The backend a tool really ran on: the CPU after an OOM fallback."""
    return "cpu" if result is not None and result.cuda_device < 0 else backend


def _view_points(rec: pycolmap.Reconstruction, image: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 2) and depths (N,) of the SfM points `image` observes, in front of its camera."""
    obs = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in image.points2D if p.has_point3D()]
    if not obs:
        return np.zeros((0, 2)), np.zeros(0)
    uv, xyz = np.array([o[0] for o in obs]), np.array([o[1] for o in obs])
    cam = image.cam_from_world().matrix()
    z = xyz @ cam[2, :3] + cam[2, 3]
    return uv[z > 0], z[z > 0]


def _run_mono(
    rec: pycolmap.Reconstruction, images_dir: Path, out: Path, options: MonoOptions, use_cuda: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mono depth of every keyframe of the undistorted model `rec`, fitted to its SfM points; the fitted depth
    and confidence of each view that fits go to `out/mono/<image>.npz`. Returns the summary and one row per view."""
    t0 = time.perf_counter()
    model, source = fetch_model(options)
    net = MonoDepth(model, use_cuda, options)
    t_load = time.perf_counter() - t0
    rows, fits, example = [], {}, None
    t0 = time.perf_counter()
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        bgr = cv2.imread(str(images_dir / image.name))
        disparity = net.predict(bgr)
        uv, z = _view_points(rec, image)
        h, w = bgr.shape[:2]
        fit = fit_view(disparity, uv * np.array([disparity.shape[1] / w, disparity.shape[0] / h]), z, options)
        fits[image.name] = fit
        if fit.fitted:
            depth, confidence = fitted_depth(disparity, fit, options)
            write_npz(
                out / MONO_ARTIFACTS["mono"] / Path(image.name).with_suffix(".npz"),
                True,
                depth=depth.astype(np.float16),
                confidence=confidence,
            )
            example = example or (image.name, bgr, depth, confidence)
        rows.append({"image": image.name, **dataclasses.asdict(fit)})
    timings = {"load": t_load, "predict_fit": time.perf_counter() - t0}
    fitted = [f for f in fits.values() if f.fitted]
    res = np.array([f.median_rel_residual for f in fitted])
    ratio = np.array([f.inlier_ratio for f in fitted])
    summary = {
        "enabled": True,
        "model": source,
        "provider": net.provider,
        "input_long_side": options.input_long_side,
        "views": len(fits),
        "fitted": len(fitted),
        "skipped": {n: f.reason for n, f in fits.items() if not f.fitted},
        "median_rel_residual": {"median": float(np.median(res)), "max": float(res.max())} if len(res) else None,
        "inlier_ratio": {"median": float(np.median(ratio)), "min": float(ratio.min())} if len(ratio) else None,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "example": example[0] if example else None,
    }
    if example:
        plots.plot_mono_example(example[1], example[2], example[3], out / MONO_ARTIFACTS["plot_mono_example"])
    plots.plot_mono_fit(res, ratio, out / MONO_ARTIFACTS["plot_mono_fit"])
    return summary, rows


def run_dense(
    align_dir: Path,
    sfm_dir: Path,
    out: Path,
    options: DenseOptions | None,
    runner: ToolRunner,
    backend: Backend = "cuda",
    gpu_index: int = 0,
    cpu_fallback: bool = False,
    hardware: str | None = None,
) -> StageOutput:
    """Run step 5 on the aligned model of the align stage dir `align_dir` (images from the sfm stage dir
    `sfm_dir`), writing into the stage dir `out`; the OpenMVS tools run through `runner`. `cpu_fallback`: a
    CUDA tool that runs out of memory (after the lower-resolution retry for densify) runs again on the CPU;
    `hardware` (the config's preset name) only goes into the report."""
    options = options or DenseOptions()
    timings, warnings = {}, []
    artifacts = dict(ARTIFACTS)

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
    dense = densify(runner, work_dir, options.openmvs, device, long_side, cpu_fallback)
    timings["densify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    xyz, normals, rgb, _ = read_openmvs_points(dense.ply)
    num_raw_points, filter_report, filtered = len(xyz), None, None
    if options.filter.enabled:
        keep, filter_report = filter_points(xyz, diag, options.filter)
        select_openmvs_points(dense.ply, work_dir / FILTERED_PLY, keep)
        xyz, normals, rgb = xyz[keep], normals[keep], rgb[keep]
        filtered = FILTERED_PLY
    timings["filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = reconstruct_mesh(
        runner, work_dir, options.mesh, device, options.openmvs.crop_to_roi, cpu_fallback, points=filtered
    )
    mesh_ply, crop_report = mesh.ply, None
    if filter_report is not None and filter_report["box"] is not None:
        box_room = (np.array(filter_report["box"]["min"]), np.array(filter_report["box"]["max"]))
        cropped = cleanup(mesh.ply, work_dir / CROPPED_PLY, bounds=box_room)
        mesh_ply = work_dir / CROPPED_PLY
        crop_report = {"faces_before": cropped.before["faces"], "faces_after": cropped.after["faces"]}
    timings["mesh"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    refine_backend: Backend = backend if options.refine.use_cuda else "cpu"
    refine_device = cuda_device(refine_backend, gpu_index)
    refined = None
    if options.refine.enabled:
        refined = refine_mesh(runner, work_dir, options.refine, refine_device, cpu_fallback, mesh_ply.name)
        if refined.timed_out:
            warnings.append(
                f"{REFINE_MESH} stopped after {options.refine.max_minutes:g} min; mesh_refined.ply is the raw mesh"
            )
        else:
            artifacts["mesh_refine_openmvs"] = OPTIONAL_ARTIFACTS["mesh_refine_openmvs"]
    refined_ply = refined.ply if refined is not None and refined.ply is not None else mesh_ply
    shutil.copyfile(refined_ply, out / ARTIFACTS["mesh_refined"])
    timings["refine"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    preview = None
    if options.preview.enabled:
        preview = texture_preview(
            runner, work_dir, refined_ply.name, out / PREVIEW_DIR, options.preview, device, cpu_fallback
        )
        artifacts["preview"] = OPTIONAL_ARTIFACTS["preview"]
    timings["preview"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mono_report, mono_rows = {"enabled": False}, []
    if options.mono.enabled:
        mono_report, mono_rows = _run_mono(urec, undistorted_dir / "images", out, options.mono, backend == "cuda")
        write_csv(pd.DataFrame(mono_rows), out / MONO_ARTIFACTS["mono_fit"], MONO_COLUMNS)
        artifacts.update(MONO_ARTIFACTS)
    timings["mono"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_points(out / ARTIFACTS["dense_points"], xyz, normals, rgb)
    shutil.copyfile(mesh_ply, out / ARTIFACTS["mesh_raw"])
    mesh_raw = _mesh_report(read_mesh(out / ARTIFACTS["mesh_raw"]), box)
    ran = refined is not None and refined.ply is not None
    mesh_refined = _mesh_report(read_mesh(out / ARTIFACTS["mesh_refined"]), box) if ran else dict(mesh_raw)
    dense_stats = _cloud_stats(xyz, box, diag)
    plots.plot_height_histogram(xyz[:, 2], 0.0, centres[:, 2], out / ARTIFACTS["plot_height"])
    plots.plot_trajectory_topdown(centres, xyz, out / ARTIFACTS["plot_topdown"], axes="xy")
    timings["outputs"] = time.perf_counter() - t0

    if len(xyz) < MIN_DENSE_POINTS:
        warnings.append(f"only {len(xyz)} dense points")
    if dense.num_points != num_raw_points:
        warnings.append(f"{DENSIFY} logged {dense.num_points} points, {dense.ply.name} has {num_raw_points}")
    retries = [r for result in (dense, mesh, refined, preview) if result is not None for r in result.oom_retries]
    warnings += [retry_warning(r) for r in retries]
    if options.mono.enabled:
        warnings += [f"mono depth: {n} not fitted: {why}" for n, why in mono_report["skipped"].items()]
        worst = (mono_report["median_rel_residual"] or {}).get("max", 0.0)
        if worst > options.mono.max_median_residual:
            warnings.append(f"mono depth: a view has median relative residual {worst:.1%} (> {options.mono.max_median_residual:.0%})")
    meshes = [("raw mesh", mesh_raw)] + ([("refined mesh", mesh_refined)] if ran else [])
    for name, stats in [("dense cloud", dense_stats), *meshes]:
        frac = stats["inside_sparse_bbox_frac"]
        if frac < MIN_INSIDE_FRAC:
            warnings.append(f"{name}: {frac:.1%} of the vertices inside the sparse bbox (< {MIN_INSIDE_FRAC:.0%})")
    for name, stats in meshes:
        if stats["largest_component_frac"] < MIN_LARGEST_COMPONENT_FRAC:
            warnings.append(
                f"{name}: largest component has {stats['largest_component_frac']:.1%} of the faces"
                f" ({stats['components']} components)"
            )
    for warning in warnings:
        log.warning(warning)

    tools = tool_rows(runner, out)
    vram = [r.peak_vram_mb for r in runner.results if r.peak_vram_mb is not None]
    openmvs_version = runner.version(DENSIFY)
    o = options.openmvs
    report: dict[str, Any] = {
        "input_model": f"{s04_align.DIR_NAME}/{s04_align.ARTIFACTS['model']}",
        "num_images": len(names),
        "units": units,
        "undistorted": cameras,
        "backend": backend,
        "cuda_device": dense.cuda_device,
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
        "num_dense_points_raw": num_raw_points,
        "filter": {"enabled": options.filter.enabled, **(filter_report or {})},
        "mesh_crop": crop_report,
        "mono": mono_report,
        "sparse_bbox": {"min": box[0].tolist(), "max": box[1].tolist()},
        "diag": diag,
        "dense": dense_stats,
        "mesh_raw": mesh_raw,
        "refine": {
            "enabled": options.refine.enabled,
            "ran": ran,
            "timed_out": refined is not None and refined.timed_out,
            "cuda_device": refined.cuda_device if refined is not None else refine_device,
            **mesh_refined,
        },
        "preview": {"enabled": options.preview.enabled, **(_preview_report(preview.ply) if preview else {})},
        "oom_retries": [dataclasses.asdict(r) for r in retries],
        "hardware": hardware,
        "tools": tools,
        "peak_vram_mb": max(vram) if vram else None,
        "disk_mb": {"undistorted": _disk_mb(undistorted_dir), "openmvs": _disk_mb(work_dir)},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / ARTIFACTS["report"])
    log.info(
        "%d dense points (%.1f%% inside the sparse bbox) in %.1f s on %s; raw mesh %d faces, largest component %.1f%%;"
        " refined mesh %d faces in %.1f s",
        len(xyz),
        100 * dense_stats["inside_sparse_bbox_frac"],
        timings["densify"],
        backend,
        mesh_raw["faces"],
        100 * mesh_raw["largest_component_frac"],
        mesh_refined["faces"],
        timings["refine"],
    )
    backends = {"densify": _backend(dense, backend), "mesh": _backend(mesh, backend)}
    if refined is not None:
        backends["refine"] = _backend(refined, refine_backend)
    if preview is not None:
        backends["texture"] = _backend(preview, backend)
    if options.mono.enabled:
        backends["mono"] = "cuda" if mono_report["provider"] == "CUDAExecutionProvider" else "cpu"
    return StageOutput(
        artifacts=artifacts,
        metrics={
            "num_dense_points": len(xyz),
            "mesh_raw_faces": mesh_raw["faces"],
            "mesh_refined_faces": mesh_refined["faces"],
            "inside_sparse_bbox_frac": dense_stats["inside_sparse_bbox_frac"],
            "largest_component_frac": mesh_raw["largest_component_frac"],
            "mono_views_fitted": mono_report.get("fitted"),
            "densify_s": round(timings["densify"], 1),
            "refine_s": round(timings["refine"], 1),
            "peak_vram_mb": report["peak_vram_mb"],
            "backend": backend,
        },
        warnings=list(warnings),
        backends=backends,
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
        device = self.ctx.config.device
        runner = ToolRunner(self.ctx.run_dir / "logs" / STAGE, self.options.openmvs.bin_dir, device.gpu_index)
        return run_dense(
            self.ctx.stage_dir(s04_align.STAGE),
            self.ctx.stage_dir(s03_sfm.STAGE),
            self.out,
            self.options,
            runner,
            self.ctx.backend("dense"),
            device.gpu_index,
            cpu_fallback=device.preference == "auto",
            hardware=self.ctx.hardware,
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
