"""Step 5: undistort the aligned model, densify it with OpenMVS, mesh, refine and preview-texture the mesh, in
world W, in `05_dense/`."""

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pycolmap
import trimesh

from recon.dense import DenseOptions
from recon.dense.openmvs import (
    DENSIFY,
    MESH_PLY,
    PREVIEW_OBJ,
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
from recon.dense.fusion import FusionOptions, FusionView, fuse
from recon.dense.mono import (
    MonoDepth, MonoOptions, ViewFit, fetch_model, fit_to_mesh, fit_view, fitted_depth, median_rel_residual, mesh_depth,
    view_points,
)
from recon.dense.simplerecon import (
    DepthRunner, SimpleReconError, SimpleReconOptions, check_installed, fetch_weights, find_scale, start_scale,
)
from recon.dense.undistort import undistort
from recon.io.colmap import camera_intrinsics, camera_keys, colmap_log
from recon.io.ply import mesh_stats, read_mesh, read_openmvs_points, select_openmvs_points, write_points
from recon.io.tables import read_json, write_csv, write_json, write_npz
from recon.io.tools import ToolRunner
from recon.mesh.cleanup import cleanup
from recon.mesh.qa import tau as geometry_tau
from recon.pipeline.device import Backend
from recon.pipeline.run_dir import manifest_hash
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
FUSION_ARTIFACTS = {
    "mesh_tsdf": "mesh_tsdf.ply",
    "mono_points": "mono_points.ply",
}
MONO_ARTIFACTS = {
    "mono": "mono",
    "mono_fit": "reports/mono_fit.csv",
    "plot_mono_fit": "reports/plots/mono_fit.png",
    "plot_mono_example": "reports/plots/mono_example.png",
}
PREVIEW_DIR = "preview"
MONO_COLUMNS = ["image", "points", "inliers", "inlier_ratio", "median_rel_residual", "median_rel_residual_all",
                "scale", "shift", "fitted", "reason", "mesh_used", "mesh_pixels_frac", "mesh_median_rel_residual"]


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


def _mesh_scene(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = read_mesh(path)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices), o3d.core.float32), o3d.core.Tensor(np.asarray(mesh.faces), o3d.core.uint32)
    )
    return scene


def _view_camera(rec: pycolmap.Reconstruction, image: pycolmap.Image, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Intrinsics at the size `shape` = (h, w) and the 4x4 cam_from_world of an undistorted (PINHOLE) view."""
    cam = rec.cameras[image.camera_id]
    fx, fy, cx, cy = cam.params
    sx, sy = shape[1] / cam.width, shape[0] / cam.height
    T = np.eye(4)
    T[:3] = image.cam_from_world().matrix()
    return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]), T


def _run_mono(
    rec: pycolmap.Reconstruction,
    images_dir: Path,
    out: Path,
    options: MonoOptions,
    use_cuda: bool,
    mesh: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mono depth of every keyframe of the undistorted model `rec`, fitted to its SfM points and, with `mesh` (and
    `options.mesh_fit`), refitted to the mesh depth with a smooth local correction; the fitted depth
    and confidence of each view that fits go to `out/mono/<image>.npz`. Returns the summary and one row per view."""
    scene = _mesh_scene(mesh) if mesh is not None and options.mesh_fit else None
    t0 = time.perf_counter()
    model, source = fetch_model(options)
    net = MonoDepth(model, use_cuda, options)
    t_load = time.perf_counter() - t0
    rows, fits, example = [], {}, None
    t0 = time.perf_counter()
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        bgr = cv2.imread(str(images_dir / image.name))
        disparity = net.predict(bgr)
        uv, z = view_points(rec, image)
        h, w = bgr.shape[:2]
        fit = fit_view(disparity, uv * np.array([disparity.shape[1] / w, disparity.shape[0] / h]), z, options)
        correction = None
        if scene is not None and fit.fitted:
            scaled = uv * np.array([disparity.shape[1] / w, disparity.shape[0] / h])
            K, T = _view_camera(rec, image, disparity.shape)
            refit, correction = fit_to_mesh(disparity, fit, mesh_depth(scene, K, T, disparity.shape), options)
            # the mesh must not pull the view away from the SfM points
            worst = max(2 * fit.median_rel_residual, options.inlier_rel)
            if refit.mesh_used and median_rel_residual(disparity, scaled, z, refit.scale, refit.shift) > worst:
                fit.mesh_pixels_frac = refit.mesh_pixels_frac
                correction = None
            else:
                fit = refit
        fits[image.name] = fit
        if fit.fitted:
            depth, confidence = fitted_depth(disparity, fit, options, correction)
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
        "source": "mono",
        "model": source,
        "provider": net.provider,
        "backend": "cuda" if net.provider == "CUDAExecutionProvider" else "cpu",
        "input_long_side": options.input_long_side,
        "views": len(fits),
        "fitted": len(fitted),
        "mesh_fitted": sum(f.mesh_used for f in fitted),
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


def _accelerometer_scale(sfm_dir: Path, align_dir: Path) -> float | None:
    """Model units (of the aligned model) per metre from the SfM stage's accelerometer check, None without one."""
    report = read_json(sfm_dir / s03_sfm.ARTIFACTS["report"])
    median = ((report.get("motion") or {}).get("scale") or {}).get("median")
    if not median:
        return None
    return float(median) * float(read_json(align_dir / s04_align.ARTIFACTS["transform"])["W_from_sfm"]["scale"])


def _run_simplerecon(
    rec: pycolmap.Reconstruction,
    images_dir: Path,
    out: Path,
    options: SimpleReconOptions,
    mono_options: MonoOptions,
    use_cuda: bool,
    accelerometer: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """SimpleRecon depth of every keyframe of the undistorted model `rec` (scale searched on a sample of views unless
    `options.scale` is set); each view's depth and confidence go to `out/mono/<image>.npz`, the layout of the mono
    depth, and its agreement with the SfM points is its row. Returns the summary and one row per view; raises
    `SimpleReconError` when no scale makes the depth agree with the SfM points."""
    check_installed(options)
    import torch

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    weights = fetch_weights(options)[1]
    runner = DepthRunner(rec, images_dir, options, device, mono_options)
    start, start_source, curve = None, "given", []
    if options.scale:
        scale = options.scale
    else:
        start, start_source = start_scale(rec, accelerometer, options)
        scale, curve = find_scale(runner, start, options)
    t_search = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows, fits, example = [], {}, None
    for i, name in enumerate(runner.names):
        result = runner.depth(i, scale)
        fits[name] = result.fit
        if result.fit.fitted:
            write_npz(
                out / MONO_ARTIFACTS["mono"] / Path(name).with_suffix(".npz"),
                True,
                depth=result.depth.astype(np.float16),
                confidence=result.confidence,
            )
            if example is None:
                example = (name, cv2.imread(str(images_dir / name)), result.depth, result.confidence)
        rows.append({"image": name, **dataclasses.asdict(result.fit)})
    fitted = [f for f in fits.values() if f.fitted]
    res = np.array([f.median_rel_residual for f in fitted])
    ratio = np.array([f.inlier_ratio for f in fitted])
    summary = {
        "enabled": True,
        "source": "simplerecon",
        "model": weights,
        "backend": device,
        "scale": scale,
        "scale_source": start_source,
        "scale_start": start,
        "accelerometer_scale": accelerometer,
        "scale_search": curve,
        "views": len(fits),
        "fitted": len(fitted),
        "mesh_fitted": 0,
        "skipped": {n: f.reason for n, f in fits.items() if not f.fitted},
        "median_rel_residual": {"median": float(np.median(res)), "max": float(res.max())} if len(res) else None,
        "inlier_ratio": {"median": float(np.median(ratio)), "min": float(ratio.min())} if len(ratio) else None,
        "seconds_per_view": round(float(np.mean(runner.seconds)), 3),
        "peak_vram_gb": round(runner.peak_vram_gb(), 2),
        "input_sizes": [f"{w}x{h}" for w, h in runner.nets],
        "timings": {"search": round(t_search, 3), "predict": round(time.perf_counter() - t0, 3)},
        "example": example[0] if example else None,
    }
    if example:
        plots.plot_mono_example(example[1], example[2], example[3], out / MONO_ARTIFACTS["plot_mono_example"])
    plots.plot_mono_fit(res, ratio, out / MONO_ARTIFACTS["plot_mono_fit"])
    return summary, rows


def load_mono_views(
    dense_dir: Path, fits: pd.DataFrame, min_inlier_ratio: float, max_residual: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(depth, K, cam_from_world) of every view of the dense stage dir `dense_dir` whose mono depth fits the SfM
    points well (`fits`: rows of `mono_fit.csv`): the depth with the unconfident pixels at 0, the intrinsics at
    the depth map's size, the 4x4 pose."""
    good = fits[fits["fitted"] & (fits["inlier_ratio"] >= min_inlier_ratio) & (fits["median_rel_residual"] <= max_residual)]
    rec = pycolmap.Reconstruction(dense_dir / ARTIFACTS["undistorted_model"])
    by_name = {im.name: im for im in rec.images.values()}
    views = []
    for name in sorted(good["image"]):
        image = by_name[name]
        data = np.load(dense_dir / MONO_ARTIFACTS["mono"] / Path(name).with_suffix(".npz"))
        depth = data["depth"].astype(np.float32)
        depth[data["confidence"] == 0] = 0.0
        cam = rec.cameras[image.camera_id]
        h, w = depth.shape
        fx, fy, cx, cy = cam.params
        sx, sy = w / cam.width, h / cam.height
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        T = np.eye(4)
        T[:3] = image.cam_from_world().matrix()
        views.append((depth, K, T))
    return views


def _run_fusion(
    out: Path,
    fits: list[dict[str, Any]],
    options: FusionOptions,
    t: float,
    diag: float,
    box: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    """TSDF of the mono depth of the views that fit well, into `out/mesh_tsdf.ply` (faces inside `box` only) and
    `out/mono_points.ply` (the depth as points). Returns the report."""
    table = pd.DataFrame(fits)
    views = [
        FusionView(*v) for v in load_mono_views(out, table, options.min_inlier_ratio, options.max_residual)
    ]
    vertices, faces, points, report = fuse(views, diag, t, options)
    if box is not None and len(faces):
        inside = ((vertices >= box[0]) & (vertices <= box[1])).all(axis=1)
        faces = faces[inside[faces].all(axis=1)]
    trimesh.Trimesh(vertices, faces, process=False).export(out / FUSION_ARTIFACTS["mesh_tsdf"])
    write_points(out / FUSION_ARTIFACTS["mono_points"], points, np.zeros_like(points), np.zeros((len(points), 3), np.uint8))
    return {"enabled": True, "tau": t, "skipped_views": len(fits) - len(views), **report, "faces_in_box": len(faces)}


CHAIN_FILE = "chain.json"  # what the OpenMVS steps leave for the next run: their key and the facts the report needs
CHAIN_VERSION = 1
CHAIN_STEPS = ("undistort", "import", "densify", "filter", "mesh", "refine", "preview")
# kept for a re-run whose OpenMVS inputs are unchanged; everything else of a stage dir is rewritten
CHAIN_KEEP = [ARTIFACTS["undistorted"], WORK_DIR, "preview", ARTIFACTS["dense_points"], ARTIFACTS["mesh_raw"], ARTIFACTS["mesh_refined"], CHAIN_FILE]


@dataclasses.dataclass
class Chain:
    """Result of the undistort, OpenMVS and filter steps of the dense stage (everything the later steps use)."""

    urec: pycolmap.Reconstruction
    cameras: dict[str, Any]
    long_side: int
    dense: DensifyResult
    mesh: MeshResult
    refined: MeshResult | None
    preview: MeshResult | None
    xyz: np.ndarray
    normals: np.ndarray
    rgb: np.ndarray
    num_raw_points: int
    filter_report: dict[str, Any] | None
    crop_report: dict[str, Any] | None
    mesh_ply: Path
    refined_ply: Path
    refine_backend: Backend
    refine_device: int
    artifacts: dict[str, str]
    warnings: list[str]
    timings: dict[str, float]
    tools: dict[str, dict[str, Any]]
    vram: list[float]
    openmvs_version: str | None


def _chain_key(options: DenseOptions, align_dir: Path) -> str:
    """sha1 of what the OpenMVS steps depend on: their options and the aligned model they start from."""
    data = {
        "version": CHAIN_VERSION,
        "options": {k: dataclasses.asdict(getattr(options, k)) for k in ("undistort", "openmvs", "filter", "mesh", "refine", "preview")},
        "align": manifest_hash(align_dir),
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _clear_chain(out: Path) -> None:
    """Remove what an earlier run left of the OpenMVS steps (the tools overwrite files in place, which would write
    through hard links to another run)."""
    for rel in CHAIN_KEEP:
        path = out / rel
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)


def _rel(path: Path | None, out: Path) -> str | None:
    return None if path is None else os.path.relpath(path, out)


def _result_json(r: MeshResult | DensifyResult | None, out: Path) -> dict[str, Any] | None:
    if r is None:
        return None
    d = {"ply": _rel(r.ply, out), "cuda_device": r.cuda_device, "oom_retries": [dataclasses.asdict(x) for x in r.oom_retries]}
    if isinstance(r, DensifyResult):
        d |= {"num_points": r.num_points, "effective_level": r.effective_level, "effective_long_side": r.effective_long_side}
    else:
        d["timed_out"] = r.timed_out
    return d


def _result_load(d: dict[str, Any] | None, out: Path, dense: bool = False) -> Any:
    if d is None:
        return None
    ply = None if d["ply"] is None else out / d["ply"]
    retries = [OomRetry(**x) for x in d["oom_retries"]]
    if dense:
        return DensifyResult(ply, d["num_points"], d["effective_level"], d["effective_long_side"], d["cuda_device"], retries)
    return MeshResult(ply, d["cuda_device"], d["timed_out"], retries)


def _save_chain(out: Path, key: str, c: Chain) -> None:
    write_json(
        {
            "key": key,
            "cameras": c.cameras,
            "long_side": c.long_side,
            "dense": _result_json(c.dense, out),
            "mesh": _result_json(c.mesh, out),
            "refined": _result_json(c.refined, out),
            "preview": _result_json(c.preview, out),
            "num_raw_points": c.num_raw_points,
            "filter_report": c.filter_report,
            "crop_report": c.crop_report,
            "mesh_ply": _rel(c.mesh_ply, out),
            "refined_ply": _rel(c.refined_ply, out),
            "refine_backend": c.refine_backend,
            "refine_device": c.refine_device,
            "artifacts": c.artifacts,
            "warnings": c.warnings,
            "timings": c.timings,
            "tools": c.tools,
            "vram": c.vram,
            "openmvs_version": c.openmvs_version,
        },
        out / CHAIN_FILE,
    )


def _load_chain(out: Path, key: str) -> Chain | None:
    """The saved chain of `out` when it was made with `key` and its files are all there, else None."""
    path = out / CHAIN_FILE
    if not path.is_file():
        return None
    d = read_json(path)
    if d.get("key") != key or any(not (out / rel).exists() for rel in CHAIN_KEEP):
        return None
    dense = _result_load(d["dense"], out, dense=True)
    work_dir = out / WORK_DIR
    filtered = work_dir / FILTERED_PLY
    xyz, normals, rgb, _ = read_openmvs_points(filtered if d["filter_report"] is not None else dense.ply)
    return Chain(
        urec=pycolmap.Reconstruction(out / ARTIFACTS["undistorted_model"]),
        cameras=d["cameras"],
        long_side=d["long_side"],
        dense=dense,
        mesh=_result_load(d["mesh"], out),
        refined=_result_load(d["refined"], out),
        preview=_result_load(d["preview"], out),
        xyz=xyz,
        normals=normals,
        rgb=rgb,
        num_raw_points=d["num_raw_points"],
        filter_report=d["filter_report"],
        crop_report=d["crop_report"],
        mesh_ply=out / d["mesh_ply"],
        refined_ply=out / d["refined_ply"],
        refine_backend=d["refine_backend"],
        refine_device=d["refine_device"],
        artifacts=d["artifacts"],
        warnings=d["warnings"],
        timings=d["timings"],
        tools=d["tools"],
        vram=d["vram"],
        openmvs_version=d["openmvs_version"],
    )


def adopt_chain(out: Path, key: str) -> None:
    """Write `chain.json` for a finished dense stage dir `out` that has none (made before the steps were reusable),
    from its report and manifest, so that a run with the same options and aligned model reuses it."""
    report, manifest = read_json(out / ARTIFACTS["report"]), read_json(out / "stage_manifest.json")
    if manifest["status"] != "success" or (out / CHAIN_FILE).exists():
        raise RuntimeError(f"{out}: no finished dense stage, or it already has {CHAIN_FILE}")
    work = Path(WORK_DIR)
    retries = report["oom_retries"]
    cuda = report["cuda_device"]
    refine = report["refine"]
    filter_report = {k: v for k, v in report["filter"].items() if k != "enabled"} if report["filter"]["enabled"] else None
    crop = report["mesh_crop"]
    mesh_ply = work / (CROPPED_PLY if crop else MESH_PLY)
    refined_ply = work / REFINED_PLY if refine["ran"] else mesh_ply
    refined = {"ply": str(work / REFINED_PLY) if refine["ran"] else None, "cuda_device": refine["cuda_device"], "oom_retries": [], "timed_out": refine["timed_out"]}
    write_json(
        {
            "key": key,
            "cameras": report["undistorted"],
            "long_side": max(max(c["width"], c["height"]) for c in report["undistorted"].values()),
            "dense": {
                "ply": ARTIFACTS["scene_dense_points"], "cuda_device": cuda, "oom_retries": retries,
                "num_points": report["num_dense_points_raw"], "effective_level": report["densify"]["effective_level"],
                "effective_long_side": report["densify"]["effective_long_side"],
            },
            "mesh": {"ply": str(work / MESH_PLY), "cuda_device": cuda, "oom_retries": [], "timed_out": False},
            "refined": refined if report["refine"]["enabled"] else None,
            "preview": (
                {"ply": f"{PREVIEW_DIR}/{PREVIEW_OBJ}", "cuda_device": cuda, "oom_retries": [], "timed_out": False}
                if report["preview"]["enabled"] else None
            ),
            "num_raw_points": report["num_dense_points_raw"],
            "filter_report": filter_report,
            "crop_report": crop,
            "mesh_ply": str(mesh_ply),
            "refined_ply": str(refined_ply),
            "refine_backend": "cuda" if refine["cuda_device"] >= 0 else "cpu",
            "refine_device": refine["cuda_device"],
            "artifacts": {k: OPTIONAL_ARTIFACTS[k] for k in ("mesh_refine_openmvs", "preview") if k in manifest["artifacts"]},
            "warnings": [w for w in manifest["warnings"] if w.startswith(f"{REFINE_MESH} stopped")],
            "timings": {k: v for k, v in report["timings"].items() if k in CHAIN_STEPS},
            "tools": report["tools"],
            "vram": [] if report["peak_vram_mb"] is None else [report["peak_vram_mb"]],
            "openmvs_version": report["openmvs_version"],
        },
        out / CHAIN_FILE,
    )


def _run_chain(
    model_dir: Path,
    sfm_dir: Path,
    out: Path,
    options: DenseOptions,
    runner: ToolRunner,
    backend: Backend,
    gpu_index: int,
    cpu_fallback: bool,
    names: list[str],
    diag: float,
    warnings: list[str],
    timings: dict[str, float],
) -> Chain:
    """Undistort, import, densify, filter, mesh, refine and preview; writes `dense_points.ply`, `mesh_raw.ply` and
    `mesh_refined.ply` too. `warnings` and `timings` collect those of the steps."""
    chain_warnings, chain_timings, artifacts = [], {}, {}
    t0 = time.perf_counter()
    undistorted_dir = out / ARTIFACTS["undistorted"]
    with colmap_log(runner.log_dir / "colmap.log"):
        urec = undistort(model_dir, sfm_dir / s03_sfm.IMAGE_ROOT, names, undistorted_dir, options.undistort)
    keys = camera_keys(urec)
    cameras = {keys[cid]: camera_intrinsics(cam) for cid, cam in sorted(urec.cameras.items()) if cid in keys}
    long_side = max(max(c["width"], c["height"]) for c in cameras.values())
    chain_timings["undistort"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    work_dir = out / WORK_DIR
    import_colmap(runner, work_dir, undistorted_dir)
    chain_timings["import"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    device = cuda_device(backend, gpu_index)
    dense = densify(runner, work_dir, options.openmvs, device, long_side, cpu_fallback)
    chain_timings["densify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    xyz, normals, rgb, _ = read_openmvs_points(dense.ply)
    num_raw_points, filter_report, filtered = len(xyz), None, None
    if options.filter.enabled:
        keep, filter_report = filter_points(xyz, diag, options.filter)
        select_openmvs_points(dense.ply, work_dir / FILTERED_PLY, keep)
        xyz, normals, rgb = xyz[keep], normals[keep], rgb[keep]
        filtered = FILTERED_PLY
    chain_timings["filter"] = time.perf_counter() - t0

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
    chain_timings["mesh"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    refine_backend: Backend = backend if options.refine.use_cuda else "cpu"
    refine_device = cuda_device(refine_backend, gpu_index)
    refined = None
    if options.refine.enabled:
        refined = refine_mesh(runner, work_dir, options.refine, refine_device, cpu_fallback, mesh_ply.name)
        if refined.timed_out:
            chain_warnings.append(
                f"{REFINE_MESH} stopped after {options.refine.max_minutes:g} min; mesh_refined.ply is the raw mesh"
            )
        else:
            artifacts["mesh_refine_openmvs"] = OPTIONAL_ARTIFACTS["mesh_refine_openmvs"]
    refined_ply = refined.ply if refined is not None and refined.ply is not None else mesh_ply
    shutil.copyfile(refined_ply, out / ARTIFACTS["mesh_refined"])
    chain_timings["refine"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    preview = None
    if options.preview.enabled:
        preview = texture_preview(
            runner, work_dir, refined_ply.name, out / PREVIEW_DIR, options.preview, device, cpu_fallback
        )
        artifacts["preview"] = OPTIONAL_ARTIFACTS["preview"]
    chain_timings["preview"] = time.perf_counter() - t0

    shutil.copyfile(mesh_ply, out / ARTIFACTS["mesh_raw"])
    write_points(out / ARTIFACTS["dense_points"], xyz, normals, rgb)
    warnings += chain_warnings
    timings.update(chain_timings)
    return Chain(
        urec, cameras, long_side, dense, mesh, refined, preview, xyz, normals, rgb, num_raw_points, filter_report,
        crop_report, mesh_ply, refined_ply, refine_backend, refine_device, artifacts, chain_warnings, chain_timings,
        tool_rows(runner, out), [r.peak_vram_mb for r in runner.results if r.peak_vram_mb is not None],
        runner.version(DENSIFY),
    )


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

    undistorted_dir = out / ARTIFACTS["undistorted"]
    work_dir = out / WORK_DIR
    key = _chain_key(options, align_dir)
    t0 = time.perf_counter()
    chain = _load_chain(out, key)
    reused = chain is not None
    if chain is None:
        _clear_chain(out)
        chain = _run_chain(
            model_dir, sfm_dir, out, options, runner, backend, gpu_index, cpu_fallback, names, diag, warnings, timings
        )
        _save_chain(out, key, chain)
    else:
        log.info("reusing the OpenMVS steps of %s (same inputs and options)", out)
        timings.update(chain.timings)
        timings["reuse"] = time.perf_counter() - t0
        warnings += chain.warnings
    urec, cameras, long_side = chain.urec, chain.cameras, chain.long_side
    dense, mesh, refined, preview = chain.dense, chain.mesh, chain.refined, chain.preview
    xyz, normals, rgb, num_raw_points = chain.xyz, chain.normals, chain.rgb, chain.num_raw_points
    filter_report, crop_report, mesh_ply, refined_ply = chain.filter_report, chain.crop_report, chain.mesh_ply, chain.refined_ply
    refine_backend, refine_device = chain.refine_backend, chain.refine_device
    artifacts.update(chain.artifacts)

    t0 = time.perf_counter()
    mono_report, mono_rows = {"enabled": False}, []
    if options.depth_source == "simplerecon":
        try:
            mono_report, mono_rows = _run_simplerecon(
                urec, undistorted_dir / "images", out, options.simplerecon, options.mono, backend == "cuda",
                _accelerometer_scale(sfm_dir, align_dir),
            )
        except (SimpleReconError, ImportError, RuntimeError, OSError) as e:
            warnings.append(f"SimpleRecon depth: {e}; using the mono depth")
            shutil.rmtree(out / MONO_ARTIFACTS["mono"], ignore_errors=True)
    if not mono_rows and options.mono.enabled:
        mono_report, mono_rows = _run_mono(urec, undistorted_dir / "images", out, options.mono, backend == "cuda", refined_ply)
    if mono_rows:
        write_csv(pd.DataFrame(mono_rows), out / MONO_ARTIFACTS["mono_fit"], MONO_COLUMNS)
        artifacts.update(MONO_ARTIFACTS)
    timings["mono"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fusion_report: dict[str, Any] = {"enabled": False}
    if options.fusion.enabled and mono_rows:
        room = (
            (np.array(filter_report["box"]["min"]), np.array(filter_report["box"]["max"]))
            if filter_report and filter_report["box"]
            else None
        )
        t = options.fusion.tau if options.fusion.tau is not None else geometry_tau(units, diag)
        fusion_report = _run_fusion(out, mono_rows, options.fusion, t, diag, room)
        artifacts.update(FUSION_ARTIFACTS)
    timings["fusion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
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
    if mono_report["enabled"]:
        label = f"{mono_report['source']} depth"
        warnings += [f"{label}: {n} not fitted: {why}" for n, why in mono_report["skipped"].items()]
        worst = (mono_report["median_rel_residual"] or {}).get("max", 0.0)
        if worst > options.mono.max_median_residual:
            warnings.append(f"{label}: a view has median relative residual {worst:.1%} (> {options.mono.max_median_residual:.0%})")
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

    tools, vram, openmvs_version = chain.tools, chain.vram, chain.openmvs_version
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
            "fusion_filter": o.fusion_filter,
            "iters": o.iters,
            "geometric_iters": o.geometric_iters,
            "crop_to_roi": o.crop_to_roi,
        },
        "num_dense_points": len(xyz),
        "num_dense_points_raw": num_raw_points,
        "filter": {"enabled": options.filter.enabled, **(filter_report or {})},
        "mesh_crop": crop_report,
        "mono": mono_report,
        "fusion": fusion_report,
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
        "reused_openmvs_steps": reused,  # the timings of undistort..preview are then those of the run that made them
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
    if mono_report["enabled"]:
        backends["mono"] = mono_report["backend"]
    return StageOutput(
        artifacts=artifacts,
        metrics={
            "num_dense_points": len(xyz),
            "mesh_raw_faces": mesh_raw["faces"],
            "mesh_refined_faces": mesh_refined["faces"],
            "inside_sparse_bbox_frac": dense_stats["inside_sparse_bbox_frac"],
            "largest_component_frac": mesh_raw["largest_component_frac"],
            "depth_source": mono_report.get("source"),
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

    def prepare(self) -> None:
        """Keep the OpenMVS steps' outputs of an earlier run (`run_dense` reuses them when their inputs are unchanged
        and clears them otherwise); the rest is removed."""
        out = self.out
        if not (out / CHAIN_FILE).is_file():
            return super().prepare()
        for path in out.iterdir():
            if path.name not in CHAIN_KEEP:
                shutil.rmtree(path) if path.is_dir() else path.unlink()

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
