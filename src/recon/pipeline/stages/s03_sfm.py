"""Step 3: features, matching and mapping of every sequence's keyframes into one COLMAP model in `03_sfm/`."""

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycolmap

from recon.io.colmap import camera_intrinsics, camera_keys, colmap_log, link_image_root, write_model
from recon.io.tables import write_csv, write_json
from recon.pipeline.device import Backend
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s02_keyframes
from recon.pipeline.stages.common import gravity_vectors, read_keyframe_inputs
from recon.report import plots
from recon.sfm import SfmOptions
from recon.sfm.features import extract_features
from recon.sfm.mapper import map_with_fallback
from recon.sfm.matching import match_features, num_verified_pairs, resolve_strategy, resolve_vocab_tree
from recon.sfm.priors import (
    PAIR_CHECKS,
    PAIR_FILTER_COLUMNS,
    filter_pairs,
    pair_rotation_errors,
    write_gravity_priors,
)
from recon.sfm.continuity import analyze as analyze_continuity, summary as continuity_summary
from recon.sfm.imu_checks import analyze_heading, analyze_motion, down_in_model
from recon.sfm.refine import polish, refine, sanity_checks
from recon.sfm.repair import repair
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


def _read_sensors(inputs, kind: str) -> dict[str, pd.DataFrame]:
    """The ingested `kind` table (`t_ns, x, y, z`) of every sequence that has one."""
    return {k: pd.read_csv(d / f"{kind}.csv") for k, d in inputs.sensor_dirs.items() if (d / f"{kind}.csv").exists()}


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

    inputs = read_keyframe_inputs(sequence_dirs)
    names = inputs.names
    image_root = link_image_root(out / IMAGE_ROOT, inputs.image_dirs)
    gravity = gravity_vectors(inputs.gravity, priors.gravity_min_conf)
    log.warning("no intrinsics prior: the focal length starts from COLMAP's guess and is refined (calibrate instead)")

    with colmap_log(log_file):
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
        if strategy in ("time", "sequential") and options.matching.loop_detection:
            vocab_tree = resolve_vocab_tree(options.matching.vocab_tree_path)
            if vocab_tree is None:
                warnings.append("no vocab tree: loop detection is off")
        strategy = match_features(
            database, len(names), options.matching, use_gpu, gpu_index, vocab_tree, inputs.images, out / "matching"
        )
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
    remap_report: dict[str, Any] = {"attempted": False, "kept": False}
    num_jumps = sum(len(v.get("jumps", [])) for v in continuity.get("sequences", {}).values())
    if options.mapper.retry_on_jumps and num_jumps and mapper_used == "global":
        # a path in pieces is typical of the global mapper on stretches with little translation: map incrementally
        remap_report.update(attempted=True, jumps_before=num_jumps)
        with colmap_log(log_file):
            raw2, used2, attempts2 = map_with_fallback(
                database, image_root, out / "sparse" / "_mapping_incremental", len(names),
                replace(options.mapper, backend="incremental", fallback=False), seed, use_gpu, gpu_index, use_gravity,
            )
            candidate = refine(raw2, database, image_root, options.refine, use_gpu, gpu_index) if raw2 else None
        if candidate is not None:
            rec2, steps2, dropped2 = candidate
            after = analyze_continuity(rec2, inputs.images, inputs.gyro, options.continuity)
            jumps_after = sum(len(v.get("jumps", [])) for v in after["sequences"].values())
            keep = jumps_after < num_jumps and rec2.num_reg_images() >= 0.95 * rec.num_reg_images()
            remap_report.update(
                jumps_after=jumps_after, flagged_before=continuity["num_flagged"], flagged_after=after["num_flagged"],
                registered_before=rec.num_reg_images(), registered_after=rec2.num_reg_images(), kept=keep,
            )
            log.info(
                "path in pieces (%d jumps): incremental mapping gives %d jumps, %d -> %d flagged, registered %d -> %d, %s",
                num_jumps, jumps_after, continuity["num_flagged"], after["num_flagged"], rec.num_reg_images(),
                rec2.num_reg_images(), "kept" if keep else "discarded",
            )
            if keep:
                rec, continuity, refine_steps, dropped = rec2, after, steps2, dropped2
                raw, mapper_used, attempts = raw2, used2, attempts2
                raw_stats = {"num_registered": raw.num_reg_images(), "mean_reproj_px": raw.compute_mean_reprojection_error()}
                write_model(raw, out / ARTIFACTS["model_raw"])
    motion: dict[str, Any] = {"enabled": False}
    heading: dict[str, Any] = {"enabled": False}
    if options.motion.enabled or options.heading.enabled:
        accel, mag = _read_sensors(inputs, "accel"), _read_sensors(inputs, "magnetic")
        R_cam_imu = {k: v.R_cam_imu for k, v in inputs.gyro.items()}
        if options.motion.enabled:
            motion = analyze_motion(rec, inputs.images, accel, R_cam_imu, down_in_model(rec, gravity), options.motion)
            for name, signals in motion["flagged"].items():  # a second opinion on the path, the same as a continuity signal
                continuity["flagged"].setdefault(name, [])
                continuity["flagged"][name] += [x for x in signals if x not in continuity["flagged"][name]]
            continuity["num_flagged"] = len(continuity["flagged"])
            continuity["num_flagged_strong"] = sum(len(v) >= 2 for v in continuity["flagged"].values())
            if motion.get("scale"):
                log.info(
                    "accelerometer windows: %d, scale spread %.2f (log), %d image(s) flagged",
                    motion["windows"], motion["scale"]["log_spread"], len(motion["flagged"]),
                )
        if options.heading.enabled:
            heading = analyze_heading(rec, inputs.images, mag, R_cam_imu, options.heading)
            if heading.get("checked"):
                log.info("magnetometer: %d keyframes, %d outlier(s)", heading["checked"], len(heading["outliers"]))
    repair_report: dict[str, Any] = {"enabled": options.repair.enabled, "kept": False, "records": []}
    if options.repair.enabled and continuity.get("num_flagged_strong"):
        candidate = pycolmap.Reconstruction(rec)
        repair_report["records"] = repair(candidate, database, continuity["flagged"], options.repair)
        applied = [r["image"] for r in repair_report["records"] if r["applied"]]
        if applied:
            polish(candidate, options.refine, use_gpu, gpu_index)
            after = analyze_continuity(candidate, inputs.images, inputs.gyro, options.continuity)
            before_err, after_err = rec.compute_mean_reprojection_error(), candidate.compute_mean_reprojection_error()
            better = after["num_flagged"] < continuity["num_flagged"] and after["num_flagged_strong"] <= continuity["num_flagged_strong"]
            keep = better and after_err <= 1.1 * before_err and candidate.num_reg_images() >= rec.num_reg_images()
            repair_report.update(
                kept=keep, flagged_before=continuity["num_flagged"], flagged_after=after["num_flagged"],
                reproj_before=before_err, reproj_after=after_err,
            )
            log.info(
                "repaired %d image(s): flagged %d -> %d, reprojection %.3f -> %.3f px, %s",
                len(applied), continuity["num_flagged"], after["num_flagged"], before_err, after_err,
                "kept" if keep else "discarded",
            )
            if keep:
                rec, continuity = candidate, after
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
    keys = camera_keys(rec)
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
        "continuity": continuity,
        "remap": remap_report,
        "motion": motion,
        "heading": heading,
        "repair": repair_report,
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
