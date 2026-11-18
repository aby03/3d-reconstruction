"""Step 3: features, matching and mapping of every sequence's keyframes into one COLMAP model in `03_sfm/`."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.io.colmap import camera_intrinsics, camera_keys, colmap_log, link_image_root, write_model
from recon.io.tables import write_csv, write_json
from recon.pipeline.device import Backend
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s02_keyframes
from recon.pipeline.stages.common import gravity_vectors, read_keyframe_inputs
from recon.report import plots
from recon.sfm import SfmOptions
from recon.sfm.continuity import analyze as analyze_continuity, summary as continuity_summary
from recon.sfm.features import LEARNED_TYPES, extract_features
from recon.sfm.imu_checks import check as check_imu
from recon.sfm.learned import FEATURES_DIR, check_installed, extract_learned, match_learned
from recon.sfm.mapper import map_with_fallback
from recon.sfm.matching import candidate_pairs, match_features, num_verified_pairs, resolve_strategy
from recon.sfm.priors import PAIR_FILTER_COLUMNS, imu_pair_filter, write_gravity_priors
from recon.sfm.quality import camera_centres, gravity_spread, model_stats, per_image_stats, reprojection_errors
from recon.sfm.quality import track_lengths
from recon.sfm.refine import refine, sanity_checks
from recon.sfm.repair import remap_on_jumps, repair_model

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


def _write_raw(out: Path, raw: pycolmap.Reconstruction) -> dict[str, Any]:
    write_model(raw, out / ARTIFACTS["model_raw"])
    return {"num_registered": raw.num_reg_images(), "mean_reproj_px": raw.compute_mean_reprojection_error()}


def _write_plots(out: Path, rec: pycolmap.Reconstruction) -> None:
    tracks = track_lengths(rec)
    bins = np.arange(1.5, max(tracks.max(initial=2), 2) + 1.5)
    plots.plot_histogram(tracks, out / ARTIFACTS["plot_track_len"], "track length", "images per point", bins=bins)
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
    priors, features = options.priors, options.features
    use_gpu = backend == "cuda"
    database = out / ARTIFACTS["database"]
    timings, warnings = {}, []

    inputs = read_keyframe_inputs(sequence_dirs)
    names = inputs.names
    image_root = link_image_root(out / IMAGE_ROOT, inputs.image_dirs)
    gravity = gravity_vectors(inputs.gravity, priors.gravity_min_conf)
    learned = features.type in LEARNED_TYPES
    if learned:
        check_installed()
    log.warning("no intrinsics prior: the focal length starts from COLMAP's guess and is refined (calibrate instead)")

    with colmap_log(log_file):
        t0 = time.perf_counter()
        camera_model = options.initial_camera_model
        if learned:
            extract_learned(database, image_root, names, features, out / FEATURES_DIR, camera_model, use_gpu, gpu_index)
        else:
            extract_features(database, image_root, names, features, camera_model, use_gpu, gpu_index)
        num_gravity_priors = 0
        if priors.use_gravity_prior:
            num_gravity_priors = write_gravity_priors(database, inputs.gravity, priors.gravity_min_conf)
        timings["features"] = time.perf_counter() - t0
        log.info("extracted features of %d images in %.1f s, %d gravity priors", len(names), timings["features"],
                 num_gravity_priors)

        t0 = time.perf_counter()
        if learned:
            strategy, loops = resolve_strategy(len(names), options.matching), False
            pairs = candidate_pairs(names, strategy, options.matching, inputs.images)
            match_learned(database, pairs, features, options.matching, out / FEATURES_DIR,
                          out / "matching" / "pairs.txt", use_gpu, gpu_index)
        else:
            strategy, loops = match_features(
                database, len(names), options.matching, features, use_gpu, gpu_index, inputs.images, out / "matching"
            )
        num_pairs = num_verified_pairs(database)
        timings["matching"] = time.perf_counter() - t0
        log.info("%s matching: %d verified pairs in %.1f s", strategy, num_pairs, timings["matching"])

        t0 = time.perf_counter()
        errors, pair_report = imu_pair_filter(
            database, inputs.images, inputs.gyro, gravity, priors.imu_pair_filter, warnings
        )
        write_csv(errors, out / ARTIFACTS["pair_filter"], PAIR_FILTER_COLUMNS)
        timings["pair_filter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        use_gravity = priors.use_gravity_prior and num_gravity_priors > 0
        raw, mapper_used, attempts = map_with_fallback(
            database, image_root, out / "sparse" / "_mapping", len(names), options.mapper, seed, use_gpu, gpu_index,
            use_gravity,
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
        raw_stats = _write_raw(out, raw)

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
    continuity: dict[str, Any] = {"enabled": False}
    if options.continuity.enabled:
        continuity = analyze_continuity(rec, inputs.images, inputs.gyro, options.continuity)
        log.info("pose continuity: %s", continuity_summary(continuity))
        if continuity["num_flagged"]:
            warnings.append(
                f"pose continuity: {continuity['num_flagged']} image(s) out of step with their time neighbours"
                f" ({continuity['num_flagged_strong']} by two or more checks)"
            )
            log.warning(warnings[-1])
    remap_report, remapped = remap_on_jumps(
        rec, continuity, mapper_used, database, image_root, out / "sparse" / "_mapping_incremental", inputs.images,
        inputs.gyro, options, seed, use_gpu, gpu_index, use_gravity, log_file,
    )
    if remapped:
        raw, mapper_used, attempts, rec, continuity, refine_steps, dropped = remapped
        raw_stats = _write_raw(out, raw)
    motion, heading = check_imu(rec, inputs.images, inputs.sensor_dirs, inputs.gyro, gravity, continuity, options)
    repair_report, rec, continuity = repair_model(
        rec, continuity, database, inputs.images, inputs.gyro, options, use_gpu, gpu_index
    )
    log.info("refined model: %d images, %d points, %.3f px (raw %.3f px) in %.1f s", rec.num_reg_images(),
             rec.num_points3D(), rec.compute_mean_reprojection_error(), raw_stats["mean_reproj_px"], timings["refine"])

    t0 = time.perf_counter()
    rec.extract_colors_for_all_images(image_root)
    write_model(rec, out / ARTIFACTS["model"])
    write_model(rec, out / ARTIFACTS["model_txt"], text=True)
    rec.export_PLY(out / ARTIFACTS["points"])
    keys = camera_keys(rec)
    intrinsics = {keys[cid]: camera_intrinsics(cam) for cid, cam in sorted(rec.cameras.items()) if cid in keys}
    write_json(intrinsics, out / ARTIFACTS["intrinsics"])
    _write_plots(out, rec)
    timings["outputs"] = time.perf_counter() - t0

    stats = model_stats(rec, names)
    report: dict[str, Any] = {
        **{k: v for k, v in stats.items() if k != "unregistered_images"},
        "num_models": next(a["num_models"] for a in attempts if a["mapper"] == mapper_used),
        "num_verified_pairs": num_pairs, "feature_type": features.type, "matcher": features.matcher,
        "matching_strategy": strategy, "loop_detection": {"enabled": loops},
        "mapper_used": mapper_used, "mapper_attempts": attempts,
        "num_gravity_priors": num_gravity_priors, "gravity_prior_used": use_gravity,
        "gravity_spread_deg": gravity_spread(rec, gravity),
        **pair_report,
        "focal_px": {key: intrinsics[key]["fx"] for key in intrinsics},
        "distortion": {key: intrinsics[key]["dist"] for key in intrinsics},
        "raw": raw_stats,
        "refine": {"enabled": options.refine.enabled, "steps": refine_steps, "dropped_images": dropped,
                   "sanity_warnings": sanity_warnings},
        "continuity": continuity, "remap": remap_report, "motion": motion, "heading": heading, "repair": repair_report,
        "unregistered_images": stats["unregistered_images"],
        "per_image": per_image_stats(rec),
        "backend": backend,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    write_json(report, out / ARTIFACTS["report"])
    metrics = ["feature_type", "matcher", "num_registered", "registered_ratio", "num_points3D", "mean_reproj_px", "median_track_len"]
    backends = {"features": backend, "matching": backend, "mapper": backend}
    return StageOutput(dict(ARTIFACTS), {k: report[k] for k in metrics}, warnings, backends)


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
