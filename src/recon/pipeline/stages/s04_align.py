"""Step 4: move the refined SfM model into the world frame W (Z up, gravity −Z, floor z = 0, X/Y along the walls)
in `04_alignment/`, and check the camera–IMU calibration against the SfM poses."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.align import AlignOptions
from recon.align.apply import align_model, apply_transform, tum_lines
from recon.align.cam_imu import SequenceImu, diagnose
from recon.core.types import SensorKind
from recon.io.colmap import write_model
from recon.io.tables import read_json, write_json
from recon.pipeline.run_dir import read_manifest, stage_dir
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s01_ingest, s02_keyframes, s03_sfm
from recon.pipeline.stages.common import (
    KeyframeInputs,
    gravity_vectors,
    read_keyframe_inputs,
    read_sensor_stream,
)
from recon.report import plots
from recon.sfm.quality import camera_centres

log = logging.getLogger(__name__)

STAGE = "align"
DIR_NAME = "04_alignment"
UNITS = "arbitrary"
SUGGESTION_NOTE = (
    "report only, never applied: a time offset is applied by re-running ingest -> align with"
    " ingest.camera_time_offset_ns; R_cam_imu can't be set from the config yet"
)

ARTIFACTS = {
    "model": "sparse_aligned",
    "model_txt": "sparse_aligned_txt",
    "points": "sparse_points_aligned.ply",
    "transform": "transform.json",
    "trajectory": "trajectory_tum",
    "report": "reports/alignment_report.json",
    "plot_gravity": "reports/plots/gravity_residuals.png",
    "plot_height": "reports/plots/height_hist.png",
    "plot_trajectory": "reports/plots/trajectory_topdown.png",
    "plot_cameras_3d": "reports/plots/cameras_3d.png",
    "plot_path_topdown": "reports/plots/path_topdown.png",
    "plot_yaw": "reports/plots/yaw_hist.png",
}


def _deg(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _imu_inputs(sequence_dirs: dict[str, Path], inputs: KeyframeInputs) -> tuple[dict[str, SequenceImu], list[str]]:
    """Per sequence with a gyro stream: its keyframes, gyro, and the R_cam_imu and time offset ingest used."""
    sequences, warnings = {}, []
    for sequence_id, sequence_dir in sequence_dirs.items():
        ingest = stage_dir(sequence_dir, s01_ingest.DIR_NAME)
        if f"sensors/{SensorKind.GYRO}" not in read_manifest(ingest).artifacts:
            warnings.append(f"camera-IMU check of {sequence_id} skipped: no gyro")
            continue
        extrinsics = read_json(ingest / s01_ingest.ARTIFACTS["sequence"])["extrinsics"][s02_keyframes.CAMERA_ID]
        sequences[sequence_id] = SequenceImu(
            keyframes=inputs.images.loc[inputs.images["sequence_id"] == sequence_id, ["image_name", "t_ns"]],
            gyro=read_sensor_stream(sequence_dir, SensorKind.GYRO),
            R_cam_imu=np.asarray(extrinsics["R_cam_imu"], dtype=float),
            time_offset_ns=int(extrinsics["time_offset_ns"]),
        )
    return sequences, warnings


def _write_plots(
    out: Path, rec: pycolmap.Reconstruction, residuals: np.ndarray, floor_detected: bool, manhattan: dict[str, Any]
) -> None:
    plots.plot_histogram(
        residuals, out / ARTIFACTS["plot_gravity"], "gravity residual (prior vs world down)", "degrees", bins=50
    )
    centres = np.array(list(camera_centres(rec).values()))
    points = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    plots.plot_height_histogram(
        points[:, 2], 0.0 if floor_detected else None, centres[:, 2], out / ARTIFACTS["plot_height"]
    )
    plots.plot_trajectory_topdown(centres, points, out / ARTIFACTS["plot_trajectory"], axes="xy")
    images = sorted((rec.images[i] for i in rec.reg_image_ids()), key=lambda im: im.name)
    forward = np.array([im.cam_from_world().rotation.matrix()[2] for im in images])  # camera z axis in the world
    plots.plot_cameras_3d(
        np.array([im.projection_center() for im in images]), forward, [f"camera {im.camera_id}" for im in images], points,
        out / ARTIFACTS["plot_cameras_3d"],
    )
    plots.plot_path_topdown(
        np.array([im.projection_center() for im in images]), forward, [f"camera {im.camera_id}" for im in images], points,
        out / ARTIFACTS["plot_path_topdown"],
    )
    histogram = np.asarray(manhattan["histogram"] or np.zeros(90))
    plots.plot_yaw_histogram(histogram, manhattan["yaw_deg"], manhattan["applied"], out / ARTIFACTS["plot_yaw"])


def run_align(
    sfm_dir: Path, sequence_dirs: dict[str, Path], out: Path, options: AlignOptions | None = None, seed: int = 0
) -> StageOutput:
    """Run step 4 on the refined model of the sfm stage dir `sfm_dir`, with the keyframes of `sequence_dirs`
    (sequence_id -> sequence dir), writing into the stage dir `out`."""
    options = options or AlignOptions()
    timings = {}

    t0 = time.perf_counter()
    rec = pycolmap.Reconstruction(sfm_dir / s03_sfm.ARTIFACTS["model"])
    sfm_report = read_json(sfm_dir / s03_sfm.ARTIFACTS["report"])
    inputs = read_keyframe_inputs(sequence_dirs)
    gravity = gravity_vectors(inputs.gravity, 0.0)
    conf = dict(zip(inputs.gravity["image_name"], inputs.gravity["conf"].astype(float)))
    t_ns = dict(zip(inputs.images["image_name"], inputs.images["t_ns"].astype(int)))
    timings["load"] = time.perf_counter() - t0

    W_from_sfm, steps, diagnostics = align_model(rec, gravity, conf, options, seed)
    timings.update(diagnostics["timings"])
    warnings = diagnostics["warnings"]

    t0 = time.perf_counter()
    cam_imu: dict[str, Any] = {}
    if options.cam_imu.enabled:
        sequences, imu_warnings = _imu_inputs(sequence_dirs, inputs)
        cam_imu, diagnose_warnings = diagnose(rec, sequences, options.cam_imu)
        warnings += imu_warnings + diagnose_warnings
    timings["cam_imu"] = time.perf_counter() - t0
    for warning in warnings:
        log.warning(warning)

    t0 = time.perf_counter()
    aligned = apply_transform(rec, W_from_sfm)
    write_model(aligned, out / ARTIFACTS["model"])
    write_model(aligned, out / ARTIFACTS["model_txt"], text=True)
    aligned.export_PLY(out / ARTIFACTS["points"])
    transform = {
        "W_from_sfm": W_from_sfm.to_dict(),
        "steps": steps,
        "units": UNITS,
        "up_axis": "+Z",
        "gravity": [0.0, 0.0, -1.0],
    }
    write_json(transform, out / ARTIFACTS["transform"])
    trajectory = out / ARTIFACTS["trajectory"]
    trajectory.mkdir(parents=True, exist_ok=True)
    for sequence_id, lines in tum_lines(aligned, t_ns).items():
        (trajectory / f"{sequence_id}.txt").write_text("".join(f"{line}\n" for line in lines))
    _write_plots(
        out, aligned, diagnostics["residuals_deg"], diagnostics["floor"]["detected"], diagnostics["manhattan"]
    )
    timings["write"] = time.perf_counter() - t0

    points = np.array([p.xyz for p in aligned.points3D.values()]).reshape(-1, 3)
    lo, hi = np.percentile(points, [2, 98], axis=0)
    report: dict[str, Any] = {
        "input_model": f"{s03_sfm.DIR_NAME}/{s03_sfm.ARTIFACTS['model']}",
        "num_registered": aligned.num_reg_images(),
        "mean_reproj_px": aligned.compute_mean_reprojection_error(),
        "sfm": {k: sfm_report.get(k) for k in ("mapper_used", "gravity_prior_used", "gravity_spread_deg")},
        "gravity": diagnostics["gravity"],
        "floor": diagnostics["floor"],
        "manhattan": {k: v for k, v in diagnostics["manhattan"].items() if k != "histogram"},
        "cam_imu": {sid: {k: v for k, v in r.items() if k != "suggestion"} for sid, r in cam_imu.items()},
        "suggested_config": {sid: r["suggestion"] for sid, r in cam_imu.items() if r.get("suggestion")},
        "suggested_config_note": SUGGESTION_NOTE,
        "W_from_sfm": transform["W_from_sfm"],
        "extent": {"min": lo.tolist(), "max": hi.tolist()},
        "units": UNITS,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / ARTIFACTS["report"])
    gravity_report, floor_report = report["gravity"], report["floor"]
    log.info(
        "aligned %d images: %s gravity (residual median %s deg), floor %s",
        report["num_registered"],
        gravity_report["method"],
        _deg(gravity_report["residual_median_deg"]),
        f"tilt {_deg(floor_report['tilt_to_gravity_deg'])} deg" if floor_report["detected"] else "not found",
    )
    manhattan = report["manhattan"]
    log.info(
        "Manhattan yaw %s deg, confidence %s: %s",
        _deg(manhattan["rotation_deg"]),
        _deg(manhattan["confidence"]),
        "applied" if manhattan["applied"] else "not applied",
    )
    for sequence_id, r in report["cam_imu"].items():
        if "error" not in r:
            log.info(
                "%s: R_cam_imu %s deg from nominal (%d pairs, well conditioned: %s), time offset %+.0f ms",
                sequence_id,
                _deg(r["angle_to_nominal_deg"]),
                r["num_pairs"],
                r["well_conditioned"],
                r["time_offset_ms"],
            )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "gravity_residual_median_deg": gravity_report["residual_median_deg"],
            "gravity_method": gravity_report["method"],
            "floor_detected": floor_report["detected"],
            "floor_tilt_deg": floor_report["tilt_to_gravity_deg"],
            "num_registered": report["num_registered"],
            "yaw_applied": report["manhattan"]["applied"],
            "yaw_confidence": report["manhattan"]["confidence"],
        },
        warnings=list(warnings),
    )


@register_stage
class AlignStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s03_sfm.STAGE,)
    config_section = "align"

    def hash_inputs(self) -> dict[str, Any]:
        return {"seed": self.ctx.config.run.seed}

    def run(self) -> StageOutput:
        sequence_dirs = {sid: self.ctx.sequence_dir(sid) for sid in self.ctx.sequence_ids}
        return run_align(
            self.ctx.stage_dir(s03_sfm.STAGE), sequence_dirs, self.out, self.options, self.ctx.config.run.seed
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
