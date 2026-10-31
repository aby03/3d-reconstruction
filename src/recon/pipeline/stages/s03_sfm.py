"""Step 3: features, matching and mapping of every sequence's keyframes into one COLMAP model in `03_sfm/`."""

import contextlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycolmap

from recon.io.colmap import camera_intrinsics, link_image_root, parse_image_name, write_model
from recon.io.tables import read_csv, read_json, write_csv, write_json
from recon.pipeline.device import Backend
from recon.pipeline.run_dir import read_manifest, stage_dir
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s01_ingest, s02_keyframes
from recon.report import plots
from recon.sfm import SfmOptions
from recon.sfm.features import extract_features
from recon.sfm.mapper import map_with_fallback
from recon.sfm.matching import match_features, num_verified_pairs, resolve_strategy, resolve_vocab_tree
from recon.sfm.priors import (
    PAIR_CHECKS,
    PAIR_FILTER_COLUMNS,
    GyroRotations,
    filter_pairs,
    pair_rotation_errors,
    write_gravity_priors,
)
from recon.sfm.refine import refine, sanity_checks
from recon.sfm.quality import (
    camera_centres,
    gravity_spread,
    model_stats,
    per_image_stats,
    reprojection_errors,
    track_lengths,
)

log = logging.getLogger(__name__)

STAGE = "sfm"
DIR_NAME = "03_sfm"
IMAGE_ROOT = "images"

ARTIFACTS = {
    "database": "database.db",
    "model_raw": "sparse/raw",
    "model": "sparse/refined",
    "model_txt": "sparse/refined_txt",
    "points": "sparse_points.ply",
    "intrinsics": "intrinsics.json",
    "report": "reports/sfm_report.json",
    "pair_filter": "reports/pair_filter.csv",
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


@dataclass(slots=True)
class SceneInputs:
    """What SfM reads from every sequence: keyframes, their gravity priors and the gyro rotations."""

    images: pd.DataFrame  # image_name, sequence_id, idx, t_ns; sorted by name
    image_dirs: dict[str, Path]  # sequence_id -> keyframe image dir
    gravity: pd.DataFrame  # gravity_cam.csv rows of every sequence
    gyro: dict[str, GyroRotations]  # sequences that have gyro rotations

    @property
    def names(self) -> list[str]:
        return self.images["image_name"].tolist()


def _read_inputs(sequence_dirs: dict[str, Path]) -> SceneInputs:
    images, gravity, image_dirs, gyro = [], [], {}, {}
    for sequence_id, sequence_dir in sequence_dirs.items():
        keyframes = stage_dir(sequence_dir, s02_keyframes.DIR_NAME)
        table = read_csv(keyframes / s02_keyframes.ARTIFACTS["keyframes"], s02_keyframes.KEYFRAMES_COLUMNS)
        images.append(table[["image_name", "idx", "t_ns"]].assign(sequence_id=sequence_id))
        image_dirs[sequence_id] = keyframes / s02_keyframes.ARTIFACTS["images"]
        gravity.append(read_csv(keyframes / s02_keyframes.ARTIFACTS["gravity_priors"], s02_keyframes.GRAVITY_COLUMNS))
        ingest = stage_dir(sequence_dir, s01_ingest.DIR_NAME)
        if "frame_rotations" in read_manifest(ingest).artifacts:
            R_cum = np.load(ingest / s01_ingest.ARTIFACTS["frame_rotations"])["R_cum"]
            info = read_json(ingest / s01_ingest.ARTIFACTS["sequence"])
            R_cam_imu = np.asarray(info["extrinsics"][s02_keyframes.CAMERA_ID]["R_cam_imu"])
            gyro[sequence_id] = GyroRotations(R_cum, R_cam_imu)
    table = pd.concat(images, ignore_index=True).sort_values("image_name", ignore_index=True)
    return SceneInputs(table, image_dirs, pd.concat(gravity, ignore_index=True), gyro)


def _gravity_vectors(gravity: pd.DataFrame, min_conf: float) -> dict[str, np.ndarray]:
    rows = gravity.dropna(subset=["gx", "gy", "gz", "conf"])
    rows = rows[rows["conf"] >= min_conf]
    vectors = rows[["gx", "gy", "gz"]].to_numpy(dtype=float)
    return dict(zip(rows["image_name"], vectors))


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
    """Run step 3 on the keyframes of `sequence_dirs` (sequence_id -> sequence dir), writing into the stage dir."""
    options = options or SfmOptions()
    priors = options.priors
    use_gpu = backend == "cuda"
    database = out / ARTIFACTS["database"]
    timings, warnings = {}, []

    inputs = _read_inputs(sequence_dirs)
    names = inputs.names
    image_root = link_image_root(out / IMAGE_ROOT, inputs.image_dirs)
    gravity = _gravity_vectors(inputs.gravity, priors.gravity_min_conf)
    log.warning("no intrinsics prior: the focal length starts from COLMAP's guess and is refined (calibrate instead)")

    with _colmap_log(log_file):
        t0 = time.perf_counter()
        camera_model = options.initial_camera_model
        extract_features(database, image_root, names, options.features, camera_model, use_gpu, gpu_index)
        num_gravity_priors = 0
        if priors.use_gravity_prior:
            num_gravity_priors = write_gravity_priors(database, inputs.gravity, priors.gravity_min_conf)
        timings["features"] = time.perf_counter() - t0
        log.info(
            "extracted features of %d images in %.1f s, %d gravity priors",
            len(names),
            timings["features"],
            num_gravity_priors,
        )

        t0 = time.perf_counter()
        strategy = resolve_strategy(len(names), options.matching)
        vocab_tree = None
        if strategy == "sequential" and options.matching.loop_detection:
            vocab_tree = resolve_vocab_tree(options.matching.vocab_tree_path)
            if vocab_tree is None:
                warnings.append("no vocab tree: loop detection is off")
        strategy = match_features(database, len(names), options.matching, use_gpu, gpu_index, vocab_tree)
        num_pairs = num_verified_pairs(database)
        timings["matching"] = time.perf_counter() - t0
        log.info("%s matching: %d verified pairs in %.1f s", strategy, num_pairs, timings["matching"])

        t0 = time.perf_counter()
        pair_filter = priors.imu_pair_filter
        errors = pair_rotation_errors(database, inputs.images, inputs.gyro, gravity, pair_filter)
        if pair_filter.enabled and (errors["check"] == "none").all() and len(errors):
            warnings.append("IMU pair filter skipped: no gyro rotations and no gravity priors")
            log.warning(warnings[-1])
        errors, disagree_frac = filter_pairs(database, errors, pair_filter)
        if disagree_frac is not None:
            warnings.append(
                f"pair filter disabled: {disagree_frac:.0%} of pairs disagree with the IMU"
                " (check R_cam_imu / time offset)"
            )
            log.warning(warnings[-1])
        write_csv(errors, out / ARTIFACTS["pair_filter"], PAIR_FILTER_COLUMNS)
        num_removed = int(errors["removed"].sum())
        timings["pair_filter"] = time.perf_counter() - t0
        log.info("IMU pair filter: %d of %d pairs removed", num_removed, len(errors))

        t0 = time.perf_counter()
        use_gravity = priors.use_gravity_prior and num_gravity_priors > 0
        work_dir = out / "sparse" / "_mapping"
        raw, mapper_used, attempts = map_with_fallback(
            database, image_root, work_dir, len(names), options.mapper, seed, use_gpu, gpu_index, use_gravity
        )
        timings["mapping"] = time.perf_counter() - t0
        if raw is None:
            raise RuntimeError(f"no mapper produced a model: {[a['error'] for a in attempts]}")
        if len(attempts) > 1:
            warnings.append(
                f"mapper fallback: {', '.join(a['mapper'] for a in attempts)} tried, {mapper_used} kept"
                f" ({raw.num_reg_images()} of {len(names)} images registered)"
            )
            log.warning(warnings[-1])
        write_model(raw, out / ARTIFACTS["model_raw"])
        raw_stats = {"num_registered": raw.num_reg_images(), "mean_reproj_px": raw.compute_mean_reprojection_error()}

        t0 = time.perf_counter()
        if options.refine.enabled:
            rec, refine_steps, dropped = refine(raw, database, image_root, options.refine, use_gpu, gpu_index)
        else:
            rec, refine_steps, dropped = raw, [], []
        timings["refine"] = time.perf_counter() - t0
    if dropped:
        log.info("deregistered %d images with too few observations: %s", len(dropped), ", ".join(dropped))
    sanity_warnings = sanity_checks(rec)
    warnings += sanity_warnings
    for warning in sanity_warnings:
        log.warning(warning)
    log.info(
        "refined model: %d images, %d points, %.3f px (raw %.3f px) in %.1f s",
        rec.num_reg_images(),
        rec.num_points3D(),
        rec.compute_mean_reprojection_error(),
        raw_stats["mean_reproj_px"],
        timings["refine"],
    )

    t0 = time.perf_counter()
    rec.extract_colors_for_all_images(image_root)
    write_model(rec, out / ARTIFACTS["model"])
    write_model(rec, out / ARTIFACTS["model_txt"], text=True)
    rec.export_PLY(out / ARTIFACTS["points"])
    keys = _camera_keys(rec)
    intrinsics = {keys[cid]: camera_intrinsics(cam) for cid, cam in sorted(rec.cameras.items()) if cid in keys}
    write_json(intrinsics, out / ARTIFACTS["intrinsics"])
    _write_plots(out, rec)
    timings["outputs"] = time.perf_counter() - t0

    stats = model_stats(rec, names)
    report: dict[str, Any] = {
        **{k: v for k, v in stats.items() if k != "unregistered_images"},
        "num_models": next(a["num_models"] for a in attempts if a["mapper"] == mapper_used),
        "num_verified_pairs": num_pairs,
        "matching_strategy": strategy,
        "loop_detection": {"enabled": vocab_tree is not None, "vocab_tree": vocab_tree},
        "mapper_used": mapper_used,
        "mapper_attempts": attempts,
        "num_gravity_priors": num_gravity_priors,
        "gravity_prior_used": use_gravity,
        "gravity_spread_deg": gravity_spread(rec, gravity),
        "pair_filter_removed": num_removed,
        "pair_filter_checked": {check: int((errors["check"] == check).sum()) for check in PAIR_CHECKS},
        "pair_filter_disabled": disagree_frac is not None,
        "focal_px": {key: intrinsics[key]["fx"] for key in intrinsics},
        "distortion": {key: intrinsics[key]["dist"] for key in intrinsics},
        "raw": raw_stats,
        "refine": {
            "enabled": options.refine.enabled,
            "steps": refine_steps,
            "dropped_images": dropped,
            "sanity_warnings": sanity_warnings,
        },
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
        warnings=warnings,
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
