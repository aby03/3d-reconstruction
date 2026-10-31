"""Step 3: features, matching and mapping of every sequence's keyframes into one COLMAP model in `03_sfm/`."""

import contextlib
import logging
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.io.colmap import camera_intrinsics, link_image_root, parse_image_name, write_model
from recon.io.tables import read_csv, write_json
from recon.pipeline.device import Backend
from recon.pipeline.run_dir import stage_dir
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s02_keyframes
from recon.report import plots
from recon.sfm import SfmOptions
from recon.sfm.features import extract_features
from recon.sfm.mapper import run_global_mapper
from recon.sfm.matching import match_features, num_verified_pairs
from recon.sfm.quality import camera_centres, model_stats, per_image_stats, reprojection_errors, track_lengths

log = logging.getLogger(__name__)

STAGE = "sfm"
DIR_NAME = "03_sfm"
IMAGE_ROOT = "images"

ARTIFACTS = {
    "database": "database.db",
    "model": "sparse/raw",
    "points": "sparse_points.ply",
    "intrinsics": "intrinsics.json",
    "report": "reports/sfm_report.json",
    "plot_track_len": "reports/plots/track_len_hist.png",
    "plot_reproj": "reports/plots/reproj_hist.png",
    "plot_trajectory": "reports/plots/trajectory_topdown.png",
}


@contextlib.contextmanager
def _colmap_log(log_file: Path | None) -> Iterator[None]:
    """Send COLMAP's glog output to `log_file` instead of the console (errors still reach stderr)."""
    if log_file is None:
        yield
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    glog = pycolmap.logging
    saved = glog.logtostderr, glog.alsologtostderr, glog.stderrthreshold
    glog.logtostderr, glog.alsologtostderr, glog.stderrthreshold = False, False, 2
    glog.set_log_destination(glog.Level.INFO, f"{log_file}.")  # glog appends a timestamp
    for level in (glog.Level.WARNING, glog.Level.ERROR, glog.Level.FATAL):
        glog.set_log_destination(level, "")
    try:
        yield
    finally:
        glog.set_log_destination(glog.Level.INFO, "")
        glog.logtostderr, glog.alsologtostderr, glog.stderrthreshold = saved


def _keyframe_images(sequence_dirs: dict[str, Path]) -> tuple[list[str], dict[str, Path]]:
    """Every keyframe's image name (sorted) and each sequence's keyframe image dir."""
    names, image_dirs = [], {}
    for sequence_id, sequence_dir in sequence_dirs.items():
        keyframes = stage_dir(sequence_dir, s02_keyframes.DIR_NAME)
        table = read_csv(keyframes / s02_keyframes.ARTIFACTS["keyframes"], s02_keyframes.KEYFRAMES_COLUMNS)
        names += table["image_name"].tolist()
        image_dirs[sequence_id] = keyframes / s02_keyframes.ARTIFACTS["images"]
    return sorted(names), image_dirs


def _camera_keys(rec: pycolmap.Reconstruction) -> dict[int, str]:
    """camera_id -> `<sequence_id>/<camera_id>`, from the image folder each camera was made for."""
    keys = {}
    for image in rec.images.values():
        sequence_id, camera_id, _ = parse_image_name(image.name)
        keys.setdefault(image.camera_id, f"{sequence_id}/{camera_id}")
    return keys


def _write_plots(out: Path, rec: pycolmap.Reconstruction) -> None:
    tracks = track_lengths(rec)
    plots.plot_histogram(
        tracks,
        out / ARTIFACTS["plot_track_len"],
        "track length",
        "images per point",
        bins=np.arange(1.5, max(tracks.max(initial=2), 2) + 1.5),
    )
    errors = reprojection_errors(rec)
    plots.plot_histogram(
        errors.clip(max=4.0), out / ARTIFACTS["plot_reproj"], "reprojection error (clipped at 4 px)", "px", bins=80
    )
    centres = np.array(list(camera_centres(rec).values()))
    points = np.array([p.xyz for p in rec.points3D.values()])
    plots.plot_trajectory_topdown(centres, points, out / ARTIFACTS["plot_trajectory"])


def run_sfm(
    sequence_dirs: dict[str, Path],
    out: Path,
    options: SfmOptions | None = None,
    backend: Backend = "cuda",
    gpu_index: int = 0,
    seed: int = 0,
    log_file: Path | None = None,
) -> StageOutput:
    """Run step 3 on the keyframes of `sequence_dirs` (sequence_id -> sequence dir), writing into the stage dir `out`."""
    options = options or SfmOptions()
    use_gpu = backend == "cuda"
    database = out / ARTIFACTS["database"]
    timings = {}

    names, image_dirs = _keyframe_images(sequence_dirs)
    image_root = link_image_root(out / IMAGE_ROOT, image_dirs)
    log.warning("no intrinsics prior: the focal length starts from COLMAP's guess and is refined (calibrate for better)")

    with _colmap_log(log_file):
        t0 = time.perf_counter()
        extract_features(database, image_root, names, options.features, options.initial_camera_model, use_gpu, gpu_index)
        timings["features"] = time.perf_counter() - t0
        log.info("extracted features of %d images in %.1f s", len(names), timings["features"])

        t0 = time.perf_counter()
        strategy = match_features(database, len(names), options.matching, use_gpu, gpu_index)
        num_pairs = num_verified_pairs(database)
        timings["matching"] = time.perf_counter() - t0
        log.info("%s matching: %d verified pairs in %.1f s", strategy, num_pairs, timings["matching"])

        t0 = time.perf_counter()
        models_dir = out / "sparse" / "_global"
        rec = run_global_mapper(database, image_root, models_dir, seed, use_gpu, gpu_index)
        num_models = len([p for p in models_dir.iterdir() if p.is_dir()]) if models_dir.is_dir() else 0
        timings["mapping"] = time.perf_counter() - t0
    if rec is None:
        raise RuntimeError("global mapper produced no model")
    log.info("global mapper: %d of %d images registered in %.1f s", rec.num_reg_images(), len(names), timings["mapping"])

    t0 = time.perf_counter()
    rec.extract_colors_for_all_images(image_root)
    write_model(rec, out / ARTIFACTS["model"])
    shutil.rmtree(models_dir)
    rec.export_PLY(out / ARTIFACTS["points"])
    keys = _camera_keys(rec)
    intrinsics = {keys[cid]: camera_intrinsics(cam) for cid, cam in sorted(rec.cameras.items()) if cid in keys}
    write_json(intrinsics, out / ARTIFACTS["intrinsics"])
    _write_plots(out, rec)
    timings["outputs"] = time.perf_counter() - t0

    stats = model_stats(rec, names)
    report: dict[str, Any] = {
        **{k: v for k, v in stats.items() if k != "unregistered_images"},
        "num_models": num_models,
        "num_verified_pairs": num_pairs,
        "matching_strategy": strategy,
        "mapper_used": "global",
        "gravity_prior_used": False,
        "pair_filter_removed": 0,
        "focal_px": {key: intrinsics[key]["fx"] for key in intrinsics},
        "unregistered_images": stats["unregistered_images"],
        "per_image": per_image_stats(rec),
        "backend": backend,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    write_json(report, out / ARTIFACTS["report"])
    metrics = ["num_registered", "registered_ratio", "num_points3D", "mean_reproj_px", "median_track_len"]
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={k: report[k] for k in metrics},
        backends={"features": backend, "matching": backend, "mapper": backend},
    )


@register_stage
class SfmStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s02_keyframes.STAGE,)
    config_section = "sfm"

    def hash_inputs(self) -> dict[str, Any]:
        return {"device": self.ctx.config.device, "seed": self.ctx.config.run.seed}

    def run(self) -> StageOutput:
        sequence_dirs = {sid: self.ctx.sequence_dir(sid) for sid in self.ctx.sequence_ids}
        return run_sfm(
            sequence_dirs,
            self.out,
            self.options,
            self.ctx.backend("sfm"),
            self.ctx.config.device.gpu_index,
            self.ctx.config.run.seed,
            self.ctx.run_dir / "logs" / STAGE / "colmap.log",
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
