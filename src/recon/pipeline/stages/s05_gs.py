"""Step 5 with `dense.backend: 2dgs`: undistort and import the aligned model (TextureMesh in mesh_post needs the
scene), fit the mono depth, train 2D Gaussian splats (kept in `gs/` for a re-run with the same training inputs),
TSDF-fuse their rendered depth into the mesh and write the Gaussian centres as the dense points; then the 2DGS
figures and a novel-view video. Writes the OpenMVS backend's artifacts except its own clouds and meshes, and no
`mesh_tsdf.ply` (mesh_post then skips the merge and the gap fill). The undistortion, import and mono depth are kept
for the next run too (`gs_prep.json`), so a training continued in chunks does not redo them."""

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pycolmap

from recon.dense import DenseOptions
from recon.dense.fusion import FusionView, backproject
from recon.dense.openmvs import INTERFACE_COLMAP, SCENE, SCENE_DENSE, import_colmap
from recon.dense.undistort import undistort
from recon.io.colmap import camera_intrinsics, camera_keys, colmap_log
from recon.io.ply import read_mesh, write_points
from recon.io.tables import read_json, write_csv, write_json
from recon.io.tools import ToolRunner
from recon.io.video import write_mp4
from recon.mesh.qa import tau as geometry_tau
from recon.pipeline.config import ConfigError
from recon.pipeline.device import Backend
from recon.pipeline.run_dir import manifest_hash
from recon.pipeline.stage import StageOutput
from recon.pipeline.stages import s03_sfm, s04_align
from recon.pipeline.stages import s05_dense as d
from recon.report import plots

log = logging.getLogger(__name__)

CACHE_VERSION = 1
CKPT = f"{d.GS_DIR}/ckpt.pt"  # training checkpoint: continues an unfinished training, or its final parameters
NOT_TRAINING = ("render_long_side", "depth", "min_alpha", "novel_views", "max_minutes")  # gs options the training ignores
# dense points: centres of the Gaussians at least this opaque and within this many tau of the TSDF mesh (the rest
# are floaters and the half-transparent fill, far from any surface: mesh_post's QA reads these points)
MIN_POINT_OPACITY, MAX_POINT_DIST_TAU = 0.5, 2.0
MONO_POINT_KEYS = ("min_inlier_ratio", "max_residual", "max_depth_frac", "point_stride", "voxel_tau")  # fusion keys
FIGURE_VIEWS = (4, 6)  # gs_holdout / gs_depth_vs_mono: up to 6 held-out views, or 4 training views without any
STRIP_FRAMES, VIDEO_FPS = 6, 24


def train_key(options: DenseOptions, align_dir: Path, seed: int) -> str:
    """sha1 of what the training depends on: the aligned model, the undistortion, the mono depth and the mono
    points made from it, the training options, the seed and the gsplat version."""
    data = {
        "version": CACHE_VERSION,
        "align": manifest_hash(align_dir),
        "undistort": dataclasses.asdict(options.undistort),
        "mono": dataclasses.asdict(options.mono),
        "fusion": {k: getattr(options.fusion, k) for k in MONO_POINT_KEYS},
        "gs": {k: v for k, v in dataclasses.asdict(options.gs).items() if k not in NOT_TRAINING},
        "seed": seed,
        "gsplat": importlib.metadata.version("gsplat"),
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def prep_key(options: DenseOptions, align_dir: Path) -> str:
    """sha1 of what the undistortion, the import, the mono depth and the mono points depend on."""
    data = {
        "version": CACHE_VERSION,
        "align": manifest_hash(align_dir),
        "undistort": dataclasses.asdict(options.undistort),
        "mono": dataclasses.asdict(options.mono),
        "fusion": {k: getattr(options.fusion, k) for k in MONO_POINT_KEYS},
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _prep_files(options: DenseOptions) -> list[str]:
    """What a kept undistortion, import and mono depth must have."""
    files = [d.ARTIFACTS["undistorted_model"], d.ARTIFACTS["undistorted_images"], d.ARTIFACTS["scene"],
             d.ARTIFACTS["scene_dense"]]
    if options.mono.enabled:
        files += [d.MONO_ARTIFACTS["mono"], d.MONO_ARTIFACTS["mono_fit"], d.FUSION_ARTIFACTS["mono_points"]]
    return files


def _mono_points(views: list[FusionView], diag: float, t: float, options: Any) -> np.ndarray:
    """Every `point_stride`-th pixel of the views' depth in the world, on a `voxel_tau`·t grid (as `fusion.fuse`)."""
    clouds = [backproject(v, options.point_stride, options.max_depth_frac * diag) for v in views]
    points = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    if len(points):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        points = np.asarray(pcd.voxel_down_sample(options.voxel_tau * t).points)
    return points


def _mesh_distance(mesh: Any, points: np.ndarray) -> np.ndarray:
    """Distance of each point to the trimesh `mesh` (inf for an empty mesh)."""
    if not len(mesh.faces):
        return np.full(len(points), np.inf)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32))).numpy()


def run_gs(
    align_dir: Path,
    sfm_dir: Path,
    out: Path,
    options: DenseOptions,
    runner: ToolRunner,
    backend: Backend,
    names: list[str],
    centres: np.ndarray,
    box: tuple[np.ndarray, np.ndarray],
    diag: float,
    units: str,
    timings: dict[str, float],
    seed: int,
    hardware: str | None,
) -> StageOutput:
    """The 2DGS dense stage after `run_dense` loaded the aligned model (`names`, camera `centres`, sparse `box`,
    `diag`, `units`, its `timings` so far); `seed` is `gs.seed` or `run.seed`."""
    import torch

    from recon.dense.gs_render import fuse_mesh, holdout_metrics, splat_points
    from recon.dense.gs_train import Splats, init_points, load_mono_depth, load_views, train

    if backend != "cuda" or not torch.cuda.is_available():
        raise ConfigError("dense.backend: 2dgs needs a CUDA GPU (device.preference cpu, or none found)")
    gs, fusion, warnings = options.gs, options.fusion, []
    artifacts = {k: v for k, v in (d.ARTIFACTS | d.GS_ARTIFACTS).items() if k not in d.OPENMVS_ONLY}
    t = geometry_tau(units, diag)
    if options.depth_source != "mono":
        warnings.append(f"depth_source {options.depth_source} is not used by backend 2dgs; using the mono depth")

    undistorted_dir, work_dir = out / d.ARTIFACTS["undistorted"], out / d.WORK_DIR
    pkey = prep_key(options, align_dir)
    prep = read_json(out / d.GS_PREP) if (out / d.GS_PREP).is_file() else {}
    prep_reused = prep.get("key") == pkey and all((out / rel).exists() for rel in _prep_files(options))
    if prep_reused:
        log.info("reusing the undistortion, import and mono depth of %s (same inputs and options)", out)
    else:
        for rel in d.GS_PREP_KEEP:
            path = out / rel
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    if prep_reused:
        urec = pycolmap.Reconstruction(out / d.ARTIFACTS["undistorted_model"])
    else:
        with colmap_log(runner.log_dir / "colmap.log"):
            urec = undistort(align_dir / s04_align.ARTIFACTS["model"], sfm_dir / s03_sfm.IMAGE_ROOT, names,
                             undistorted_dir, options.undistort)
    keys = camera_keys(urec)
    cameras = {keys[cid]: camera_intrinsics(cam) for cid, cam in sorted(urec.cameras.items()) if cid in keys}
    timings["undistort"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if not prep_reused:
        import_colmap(runner, work_dir, undistorted_dir)
        shutil.copyfile(work_dir / SCENE, work_dir / SCENE_DENSE)  # TextureMesh's scene: cameras and images only
    timings["import"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mono_report, good = {"enabled": False}, pd.DataFrame({"image": []})
    if options.mono.enabled:
        if prep_reused:
            mono_report, table = prep["mono"], pd.read_csv(out / d.MONO_ARTIFACTS["mono_fit"])
        else:
            mono_report, mono_rows = d._run_mono(urec, undistorted_dir / "images", out, options.mono, True, None)
            table = pd.DataFrame(mono_rows)
            write_csv(table, out / d.MONO_ARTIFACTS["mono_fit"], d.MONO_COLUMNS)
        artifacts.update({k: v for k, v in d.MONO_ARTIFACTS.items() if (out / v).exists()})
        good = d.good_fits(table, fusion.min_inlier_ratio, fusion.max_residual)
        if not prep_reused:
            mono_views = [FusionView(*v) for v in
                          d.load_mono_views(out, table, fusion.min_inlier_ratio, fusion.max_residual)]
            mono_points = _mono_points(mono_views, diag, t, fusion)
            zeros = np.zeros_like(mono_points)
            write_points(out / d.FUSION_ARTIFACTS["mono_points"], mono_points, zeros, zeros.astype(np.uint8))
        artifacts["mono_points"] = d.FUSION_ARTIFACTS["mono_points"]
    if not prep_reused:
        write_json({"key": pkey, "mono": mono_report}, out / d.GS_PREP)
    timings["mono"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gs_dir, key, ckpt = out / d.GS_DIR, train_key(options, align_dir, seed), out / CKPT
    views = load_views(urec, undistorted_dir / "images", gs.long_side, gs.holdout_every)
    cache = read_json(out / d.GS_CACHE) if (out / d.GS_CACHE).is_file() else {}
    same = cache.get("key") == key and ckpt.is_file()
    reused = same and cache.get("finished", False) and all((out / rel).is_file() for rel in d.GS_ARTIFACTS.values())
    resume = same and not cache.get("finished", False)
    torch.cuda.reset_peak_memory_stats()
    if reused:
        log.info("reusing the 2DGS training of %s (same inputs and options)", out)
        splats = Splats.from_params(torch.load(ckpt, map_location="cpu")["params"])
        timings["reuse"] = time.perf_counter() - t0
    else:
        if not resume:
            shutil.rmtree(gs_dir, ignore_errors=True)
            gs_dir.mkdir()
        mono = load_mono_depth(out / d.MONO_ARTIFACTS["mono"], views, set(good["image"]))
        init = init_points(urec, views, out / d.FUSION_ARTIFACTS["mono_points"], gs.init_mono_points, seed,
                           gs.max_gaussians)
        prev = cache if resume else {}
        cache = {"key": key, "finished": False, "step": prev.get("step", 0), "init_points": len(init[0]),
                 "mono_views": len(mono), "peak_vram_mb": prev.get("peak_vram_mb", 0)}
        write_json(cache, out / d.GS_CACHE)  # before training: a killed run's checkpoint is then continued
        splats, info = train(views, init, mono, gs, "cuda", out / d.GS_ARTIFACTS["train_log"], seed, ckpt)
        splats.save_ply(out / d.GS_ARTIFACTS["splats"])
        cache |= {"finished": info["finished"], "step": info["step"], "resumed_from": info["resumed_from"],
                  "train_s": info["elapsed_s"],
                  "peak_vram_mb": max(cache["peak_vram_mb"], torch.cuda.max_memory_allocated() / 2**20)}
        write_json(cache, out / d.GS_CACHE)
        if not info["finished"]:
            warnings.append(f"2DGS training at step {info['step']} of {gs.iterations}; rerun dense with --force dense"
                            " to continue")
    timings["train"] = cache["train_s"]

    t0 = time.perf_counter()
    holdout = holdout_metrics(splats, views)
    write_json(holdout, out / d.GS_ARTIFACTS["holdout"])
    timings["holdout"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh, mesh_report = fuse_mesh(splats, views, fusion, t, diag, gs)
    mesh.export(out / d.ARTIFACTS["mesh_raw"])
    shutil.copyfile(out / d.ARTIFACTS["mesh_raw"], out / d.ARTIFACTS["mesh_refined"])
    timings["fuse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    xyz, normals, rgb = splat_points(splats, views, MIN_POINT_OPACITY)
    near = _mesh_distance(mesh, xyz) <= MAX_POINT_DIST_TAU * t
    xyz, normals, rgb = xyz[near], normals[near], rgb[near]
    write_points(out / d.ARTIFACTS["dense_points"], xyz, normals, rgb)
    timings["points"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    made, figure_warnings = _figures(splats, views, urec, out, gs, set(good["image"]))
    artifacts.update(made)
    warnings += figure_warnings
    timings["figures"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh_raw = d._mesh_report(read_mesh(out / d.ARTIFACTS["mesh_raw"]), box)
    dense_stats = d._cloud_stats(xyz, box, diag)
    plots.plot_height_histogram(xyz[:, 2], 0.0, centres[:, 2], out / d.ARTIFACTS["plot_height"])
    plots.plot_trajectory_topdown(centres, xyz, out / d.ARTIFACTS["plot_topdown"], axes="xy")
    timings["outputs"] = time.perf_counter() - t0

    if len(xyz) < d.MIN_DENSE_POINTS:
        warnings.append(f"only {len(xyz)} dense points")
    warnings += d.quality_warnings(mono_report, options.mono, dense_stats, [("raw mesh", mesh_raw)])
    for warning in warnings:
        log.warning(warning)

    peak = max(cache["peak_vram_mb"], torch.cuda.max_memory_allocated() / 2**20)
    gsplat_version = importlib.metadata.version("gsplat")
    gs_report = {
        "iterations": gs.iterations,
        "step": cache["step"],
        "finished": cache["finished"],
        "resumed_from": cache.get("resumed_from"),
        "num_gaussians": len(splats),
        "init_points": cache["init_points"],
        "mono_views": cache["mono_views"],
        "views": len(views),
        "holdout_views": len(holdout["per_view"]),
        "train_s": round(cache["train_s"], 3),
        "holdout_psnr": holdout["psnr"],
        "holdout_ssim": holdout["ssim"],
        "peak_vram_mb": round(peak),
        "seed": seed,
        "reused": reused,  # train_s, init_points, mono_views, resumed_from are then those of the run that trained
        "prep_reused": prep_reused,  # undistortion, import and mono depth of an earlier run (mono timings are its)
        "novel_views": len(list((out / d.GS_FIGURES["gs_novel"]).glob("*.jpg"))),
        "mesh": mesh_report,
        "gsplat_version": gsplat_version,
    }
    report: dict[str, Any] = {
        "input_model": f"{s04_align.DIR_NAME}/{s04_align.ARTIFACTS['model']}",
        "num_images": len(names),
        "units": units,
        "undistorted": cameras,
        "backend": backend,
        "dense_backend": "2dgs",
        "num_dense_points": len(xyz),
        "mono": mono_report,
        "fusion": {"enabled": False},  # no mono gap filler: the 2DGS mesh is the fused surface (gs.mesh)
        "sparse_bbox": {"min": box[0].tolist(), "max": box[1].tolist()},
        "diag": diag,
        "dense": dense_stats,
        "mesh_raw": mesh_raw,
        "refine": {"enabled": False, "ran": False, **mesh_raw},
        "gs": gs_report,
        "hardware": hardware,
        "peak_vram_mb": gs_report["peak_vram_mb"],
        "disk_mb": {"undistorted": d._disk_mb(undistorted_dir), "gs": d._disk_mb(gs_dir)},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / d.ARTIFACTS["report"])
    log.info(
        "2DGS: %d Gaussians (%s, %.1f s), held-out PSNR %.2f; mesh %d faces, largest component %.1f%%",
        len(splats), "reused" if reused else "trained", cache["train_s"], holdout["psnr"], mesh_raw["faces"],
        100 * mesh_raw["largest_component_frac"],
    )
    return StageOutput(
        artifacts=artifacts,
        metrics={
            "num_dense_points": len(xyz),
            "mesh_raw_faces": mesh_raw["faces"],
            "mesh_refined_faces": mesh_raw["faces"],
            "inside_sparse_bbox_frac": dense_stats["inside_sparse_bbox_frac"],
            "largest_component_frac": mesh_raw["largest_component_frac"],
            "depth_source": mono_report.get("source"),
            "mono_views_fitted": mono_report.get("fitted"),
            "densify_s": 0.0,
            "refine_s": 0.0,
            "train_s": round(cache["train_s"], 1),
            "holdout_psnr": round(holdout["psnr"], 3) if np.isfinite(holdout["psnr"]) else None,
            "peak_vram_mb": report["peak_vram_mb"],
            "backend": backend,
            "backend_dense": "2dgs",
        },
        warnings=list(warnings),
        backends={"train": "cuda", **({"mono": mono_report["backend"]} if mono_report["enabled"] else {})},
        tools={
            "gsplat": gsplat_version, "torch": torch.__version__, "OpenMVS": runner.version(INTERFACE_COLMAP),
            "pycolmap": pycolmap.__version__,
        },
    )


def _figures(
    splats: Any, views: list[Any], urec: pycolmap.Reconstruction, out: Path, gs: Any, good: set[str]
) -> tuple[dict[str, str], list[str]]:
    """The 2DGS figures (`d.GS_FIGURES`): training curve, held-out views (with one ground truth | render pair),
    2DGS vs mono depth, and the novel-view frames, video and strip. A figure that fails is a warning.
    Returns the artifacts made and the warnings."""
    from recon.dense.gs_render import novel_path, render, render_path
    from recon.dense.gs_train import load_mono_depth

    files, warnings = d.GS_FIGURES, []
    path = lambda key: out / files[key]  # noqa: E731

    def attempt(name: str, make: Any) -> None:
        try:
            make()
        except Exception as error:  # a figure never fails the stage
            warnings.append(f"2DGS figure {name} skipped: {type(error).__name__}: {error}")

    def numpy(r: dict[str, Any]) -> dict[str, np.ndarray]:
        return {k: v.float().cpu().numpy() for k, v in r.items()}

    attempt("gs_training", lambda: plots.plot_gs_training(pd.read_csv(out / d.GS_ARTIFACTS["train_log"]),
                                                          path("plot_gs_training")))
    held = [v for v in views if v.holdout]
    pool, count = (held, FIGURE_VIEWS[1]) if held else (views, FIGURE_VIEWS[0])
    picks = [pool[i] for i in sorted(set(np.linspace(0, len(pool) - 1, min(count, len(pool))).round().astype(int)))]
    renders = {}

    def holdout() -> None:
        renders.update({v.name: numpy(render(splats, v.K, v.viewmat, v.size, gs.depth)) for v in picks})
        rows = [{"name": v.name if v.holdout else f"{v.name} (train)", "gt": v.image / 255, **renders[v.name]}
                for v in picks]
        plots.plot_gs_holdout(rows, path("plot_gs_holdout"))
        first = rows[0]
        gap = np.ones((first["gt"].shape[0], 8, 3))
        pair = np.concatenate([first["gt"], gap, first["rgb"]], axis=1)
        cv2.imwrite(str(path("gs_holdout_pair")), (pair[..., ::-1] * 255).round().astype(np.uint8))

    attempt("gs_holdout", holdout)

    def depth_vs_mono() -> None:
        mono = load_mono_depth(out / d.MONO_ARTIFACTS["mono"], picks, good)
        rows = [{"name": v.name, **renders[v.name], "mono": mono[v.name][0], "mono_valid": mono[v.name][1]}
                for v in picks if v.name in mono and v.name in renders]
        if rows:
            plots.plot_gs_depth_vs_mono(rows, path("plot_gs_depth_vs_mono"))

    if (out / d.MONO_ARTIFACTS["mono"]).is_dir():
        attempt("gs_depth_vs_mono", depth_vs_mono)

    def novel() -> None:
        shutil.rmtree(path("gs_novel"), ignore_errors=True)
        cams, size = novel_path(urec, gs.novel_views, gs.render_long_side)
        if not cams:
            return
        frames = render_path(splats, cams, size, path("gs_novel"))
        write_mp4((cv2.imread(str(f))[..., ::-1] for f in frames), path("gs_novel_video"), VIDEO_FPS)
        tiles = []
        for i in np.linspace(0, len(cams) - 1, min(STRIP_FRAMES, len(cams))).round().astype(int):
            r = numpy(render(splats, *cams[i], size))
            tiles.append((r["rgb"], np.clip((r["normal"] + 1) / 2, 0, 1)))
        gap = np.ones((size[1], 4, 3))
        rows = [np.concatenate([x for tile in tiles for x in (tile[k], gap)][:-1], axis=1) for k in (0, 1)]
        strip = np.concatenate([rows[0], np.ones((4, rows[0].shape[1], 3)), rows[1]])  # RGB over normals
        cv2.imwrite(str(path("gs_novel_strip")), (strip[..., ::-1] * 255).round().astype(np.uint8))

    attempt("gs_novel", novel)
    return {k: rel for k, rel in files.items() if (out / rel).exists()}, warnings
