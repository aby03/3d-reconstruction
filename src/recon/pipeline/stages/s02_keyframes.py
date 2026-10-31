"""Step 2: select keyframes, extract them at full resolution and write their IMU priors into `02_keyframes/`."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recon.capture.base import BaseCaptureLoader
from recon.core.types import SensorKind
from recon.io.tables import read_csv, read_json, write_csv, write_json
from recon.keyframes.metrics import REJECT_REASONS, SCORE_COLUMNS, analysis_scale, analyze_frames, score_frames
from recon.keyframes.priors import gravity_priors, relative_rotation_priors
from recon.keyframes.select import KeyframeOptions, Selection, select_uniform
from recon.report import plots
from recon.pipeline.run_dir import read_manifest, stage_dir
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s01_ingest
from recon.pipeline.stages.common import open_loader, read_sensor_stream
from recon.sensors.sync import COLUMNS

log = logging.getLogger(__name__)

STAGE = "keyframes"
DIR_NAME = "02_keyframes"
CAMERA_ID = "cam0"
KEYFRAMES_COLUMNS = ["idx", "t_ns", "image_name", "quality", "sharp_rel", "omega_norm", "trigger", "gap_s"]
GRAVITY_COLUMNS = ["image_name", "gx", "gy", "gz", "conf"]
ROTATION_COLUMNS = ["image_i", "image_j", "qx", "qy", "qz", "qw", "dt_s"]

ARTIFACTS = {
    "images": "images",
    "keyframes": "keyframes.csv",
    "frame_scores": "frame_scores.csv",
    "gravity_priors": "priors/gravity_cam.csv",
    "relative_rotations": "priors/relative_rotations.csv",
    "selection_report": "reports/selection_report.json",
    "plot_scores": "reports/plots/scores_timeline.png",
    "plot_keyframes": "reports/plots/keyframe_strip.jpg",
}


def image_name(sequence_id: str, idx: int, camera_id: str = CAMERA_ID) -> str:
    """Scene-wide key of a keyframe image; the file is `images/<camera_id>/<idx:06d>.jpg` in this stage dir."""
    return f"{sequence_id}/{camera_id}/{idx:06d}.jpg"


def _keyframes_table(scores: pd.DataFrame, selection: Selection, names: dict[int, str]) -> pd.DataFrame:
    indices = selection.indices
    rows = scores.set_index("idx").loc[indices]
    t_ns = rows["t_ns"].to_numpy()
    gap_s = np.full(len(indices), np.nan)
    gap_s[1:] = np.diff(t_ns) * 1e-9
    return pd.DataFrame(
        {
            "idx": indices,
            "t_ns": t_ns,
            "image_name": [names[i] for i in indices],
            "quality": rows["q"].to_numpy(),
            "sharp_rel": rows["sharp_rel"].to_numpy(),
            "omega_norm": rows["omega_norm"].to_numpy(),
            "trigger": selection.triggers,
            "gap_s": gap_s,
        }
    )


def _analyze(loader: BaseCaptureLoader, frames: pd.DataFrame, options: KeyframeOptions) -> pd.DataFrame:
    """Pass A: quality metrics of every frame at analysis resolution, joined with omega_norm and scored."""
    camera = loader.load().cameras[CAMERA_ID]
    scale = analysis_scale(camera.upright_size, options.analysis_long_side)
    metrics = analyze_frames(loader.iter_frames(CAMERA_ID, scale=scale))
    t_ns, omega_norm = frames["t_ns"].to_numpy(), frames["omega_norm"].to_numpy()
    return score_frames(metrics, t_ns, omega_norm, options.reject, options.weights)


def _write_plots(out: Path, scores: pd.DataFrame, keyframes: pd.DataFrame, options: KeyframeOptions) -> None:
    t0 = scores["t_ns"].iloc[0]
    plots.plot_scores_timeline(
        (scores["t_ns"].to_numpy() - t0) * 1e-9,
        {name: scores[name].to_numpy() for name in ("sharp_rel", "omega_norm", "feat_coverage", "q")},
        (keyframes["t_ns"].to_numpy() - t0) * 1e-9,
        out / ARTIFACTS["plot_scores"],
        thresholds={
            "sharp_rel": options.reject.sharp_rel_min,
            "omega_norm": options.reject.omega_max,
            "feat_coverage": options.reject.feat_coverage_min,
        },
    )
    images = out / ARTIFACTS["images"]
    sheet = [
        (f"{idx} {trigger}", images / name.split("/", 1)[1])
        for idx, trigger, name in keyframes[["idx", "trigger", "image_name"]].itertuples(index=False)
    ]
    plots.contact_sheet(sheet, out / ARTIFACTS["plot_keyframes"])


def run_keyframes(sequence_dir: Path, out: Path, options: KeyframeOptions | None = None) -> StageOutput:
    """Run step 2 on the ingested sequence in `sequence_dir`, writing into the existing stage dir `out`."""
    options = options or KeyframeOptions()
    ingest = stage_dir(sequence_dir, s01_ingest.DIR_NAME)
    t_start = time.perf_counter()

    info = read_json(ingest / s01_ingest.ARTIFACTS["sequence"])
    sync_info = read_json(ingest / s01_ingest.ARTIFACTS["sync_report"])
    frames = read_csv(ingest / s01_ingest.ARTIFACTS["frame_sensor_data"], COLUMNS)
    t_ns = frames["t_ns"].to_numpy()
    sequence_id = info["sequence_id"]
    timings = {}

    loader = open_loader(sequence_dir)
    scores = _analyze(loader, frames, options)
    write_csv(scores, out / ARTIFACTS["frame_scores"], SCORE_COLUMNS)
    timings["pass_a"] = time.perf_counter() - t_start
    rejected = scores["rejected_reason"].ne("").to_numpy()
    log.info("analyzed %d frames in %.1f s, %d rejected", len(scores), timings["pass_a"], rejected.sum())

    t0 = time.perf_counter()
    selection = select_uniform(t_ns, options.uniform_fps, scores["q"].to_numpy(), rejected)
    indices = selection.indices
    names = {i: image_name(sequence_id, i) for i in indices}
    timings["select"] = time.perf_counter() - t0
    log.info("selected %d of %d frames at %.3g fps", len(indices), len(frames), options.uniform_fps)

    t0 = time.perf_counter()
    paths = loader.extract_frames(indices, out / ARTIFACTS["images"], CAMERA_ID, options.jpeg_quality)
    if len(paths) != len(indices):
        raise RuntimeError(f"extracted {len(paths)} of {len(indices)} keyframes")
    timings["extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    keyframes = _keyframes_table(scores, selection, names)
    write_csv(keyframes, out / ARTIFACTS["keyframes"], KEYFRAMES_COLUMNS)

    gravity = gravity_priors(frames, indices)
    gravity.insert(0, "image_name", gravity.pop("idx").map(names))
    write_csv(gravity, out / ARTIFACTS["gravity_priors"], GRAVITY_COLUMNS)

    warnings = []
    imu_used = "sensors/gyro" in read_manifest(ingest).artifacts
    if imu_used:
        gyro = read_sensor_stream(sequence_dir, SensorKind.GYRO)
        R_cam_imu = np.asarray(info["extrinsics"][CAMERA_ID]["R_cam_imu"])
        t_query = t_ns + sync_info["camera_time_offset_ns"]  # sensor-clock times, as used in sync
        rotations = relative_rotation_priors(gyro, t_query, indices, R_cam_imu, options.prior_window)
    else:
        warnings.append("no gyro stream: relative rotation priors are empty")
        log.warning(warnings[-1])
        rotations = pd.DataFrame(columns=["idx_i", "idx_j", *ROTATION_COLUMNS[2:]])
    rotations.insert(0, "image_i", rotations.pop("idx_i").map(names))
    rotations.insert(1, "image_j", rotations.pop("idx_j").map(names))
    write_csv(rotations, out / ARTIFACTS["relative_rotations"], ROTATION_COLUMNS)
    timings["priors"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _write_plots(out, scores, keyframes, options)
    timings["plots"] = time.perf_counter() - t0

    gaps = keyframes["gap_s"].dropna()
    reasons = scores["rejected_reason"]
    report = {
        "num_frames": len(frames),
        "num_rejected_by_reason": {r: int((reasons == r).sum()) for r in REJECT_REASONS},
        "num_keyframes": len(indices),
        "method": options.method,
        "triggers_count": {t: selection.triggers.count(t) for t in sorted(set(selection.triggers))},
        "mean_sharp_rel_selected": float(keyframes["sharp_rel"].mean()),
        "mean_sharp_rel_all": float(scores["sharp_rel"].mean()),
        "median_gap_s": float(gaps.median()) if len(gaps) else None,
        "max_gap_s": float(gaps.max()) if len(gaps) else None,
        "imu_used": imu_used,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    write_json(report, out / ARTIFACTS["selection_report"])
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "num_keyframes": report["num_keyframes"],
            "num_rejected": int(rejected.sum()),
            "num_forced": report["triggers_count"].get("forced", 0),
            "median_gap_s": report["median_gap_s"],
            "max_gap_s": report["max_gap_s"],
            "pass_a_s": report["timings"]["pass_a"],
            "extract_s": report["timings"]["extract"],
        },
        warnings=warnings,
    )


@register_stage
class KeyframesStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "sequence"
    requires = (s01_ingest.STAGE,)
    config_section = "keyframes"

    def run(self) -> StageOutput:
        return run_keyframes(self.ctx.sequence_dir(self.sequence_id), self.out, self.options)

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
