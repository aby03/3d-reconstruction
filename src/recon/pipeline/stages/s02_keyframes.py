"""Step 2: select keyframes, extract them at full resolution and write their IMU priors into `02_keyframes/`."""

import logging
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from recon.capture.base import BaseCaptureLoader
from recon.core.types import Frame, SensorKind
from recon.io.tables import read_csv, read_json, write_csv, write_json
from recon.io.video import resize
from recon.keyframes.metrics import (
    REJECT_REASONS,
    SCORE_COLUMNS,
    analysis_scale,
    analyze_frames,
    score_frames,
    to_gray,
)
from recon.keyframes.overlap import KLT_LONG_SIDE
from recon.keyframes.priors import gravity_priors, relative_rotation_priors
from recon.keyframes.select import ADAPTIVE_TRIGGERS, KeyframeOptions, Selection, select_adaptive, select_uniform
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
STILLS_CAMERA_ID = "cam1"  # the reference JPEGs: their own camera, so SfM calibrates its lens separately
KEYFRAMES_COLUMNS = ["idx", "t_ns", "image_name", "quality", "sharp_rel", "omega_norm", "trigger", "gap_s"]
FRAME_SCORES_COLUMNS = [*SCORE_COLUMNS[:-2], "flow_med", "track_ratio", *SCORE_COLUMNS[-2:]]
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


@dataclass(slots=True)
class Backends:
    decode: Literal["nvdec", "pyav"] = "pyav"  # analysis pass decode
    metrics: Literal["torch", "numpy"] = "numpy"
    gpu_index: int = 0

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.gpu_index}"


# per-operation setting -> device preference
_PREFERENCE = {"auto": "auto", "nvdec": "cuda", "torch": "cuda", "pyav": "cpu", "numpy": "cpu"}


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


def _copy_stills(loader: BaseCaptureLoader, sequence_id: str, frames: pd.DataFrame, images: Path) -> dict[int, str]:
    """Copy the capture's reference JPEGs (frame index -> file) to `images/<stills camera>/` unchanged; frame index -> image name."""
    refs = loader.load().reference_images
    folder = images / STILLS_CAMERA_ID
    folder.mkdir(parents=True, exist_ok=True)
    names = {}
    for idx, path in sorted(refs.items()):
        names[idx] = image_name(sequence_id, idx, STILLS_CAMERA_ID)
        shutil.copyfile(path, folder / f"{idx:06d}.jpg")
    return names


def _stills_table(frames: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    idx = np.array(sorted(names), dtype=int)
    return pd.DataFrame(
        {
            "idx": idx,
            "t_ns": frames["t_ns"].to_numpy()[idx],
            "image_name": [names[i] for i in idx],
            "quality": np.nan,
            "sharp_rel": np.nan,
            "omega_norm": np.nan,
            "trigger": "still",
            "gap_s": np.nan,
        }
    )


def _keep_tracking_frames(frames: Iterator[Frame], grays: list[np.ndarray | None]) -> Iterator[Frame]:
    """Pass the frames through, storing each as a gray frame at the KLT tracking resolution in `grays[idx]`."""
    for frame in frames:
        gray = to_gray(frame.image)
        scale = analysis_scale(gray.shape[::-1], KLT_LONG_SIDE)
        grays[frame.idx] = resize(gray, scale)
        yield frame


def _analyze(
    loader: BaseCaptureLoader, frames: pd.DataFrame, options: KeyframeOptions, backends: Backends
) -> tuple[pd.DataFrame, list[np.ndarray | None]]:
    """Pass A: quality metrics of every frame at analysis resolution, joined with omega_norm and scored.

    For adaptive selection also the gray tracking frames (None where decoding failed), else an empty list.
    """
    camera = loader.load().cameras[CAMERA_ID]
    scale = analysis_scale(camera.upright_size, options.analysis_long_side)
    frames_iter = loader.iter_frames(CAMERA_ID, scale=scale)
    grays: list[np.ndarray | None] = []
    if options.method == "adaptive":
        grays = [None] * len(frames)
        frames_iter = _keep_tracking_frames(frames_iter, grays)
    metrics = analyze_frames(frames_iter, backends.metrics, options.metrics_batch, backends.torch_device)
    t_ns, omega_norm = frames["t_ns"].to_numpy(), frames["omega_norm"].to_numpy()
    return score_frames(metrics, t_ns, omega_norm, options.reject, options.weights), grays


def _gyro_angle(ingest: Path, ingest_artifacts: dict[str, str]) -> Callable[[int, int], float]:
    """Rotation angle in degrees between two frames from the chained gyro rotations; NaN without gyro."""
    if "frame_rotations" not in ingest_artifacts:
        return lambda i, j: np.nan
    R_cum = np.load(ingest / s01_ingest.ARTIFACTS["frame_rotations"])["R_cum"]
    valid = ~np.isnan(R_cum).any(axis=1)
    rotations = Rotation.from_quat(np.where(valid[:, None], R_cum, [0.0, 0.0, 0.0, 1.0]))

    def angle(i: int, j: int) -> float:
        return float(np.degrees((rotations[i].inv() * rotations[j]).magnitude())) if valid[i] and valid[j] else np.nan

    return angle


def _select(
    scores: pd.DataFrame, grays: list[np.ndarray | None], angle: Callable[[int, int], float], options: KeyframeOptions
) -> Selection:
    t_ns, q = scores["t_ns"].to_numpy(), scores["q"].to_numpy()
    rejected = scores["rejected_reason"].ne("").to_numpy()
    if options.method == "uniform":
        return select_uniform(t_ns, options.uniform_fps, q, rejected)
    return select_adaptive(t_ns, q, rejected, grays, angle, options)


def _write_plots(out: Path, scores: pd.DataFrame, keyframes: pd.DataFrame, options: KeyframeOptions) -> None:
    t0 = scores["t_ns"].iloc[0]
    names = ["sharp_rel", "omega_norm", "feat_coverage", "q", "flow_med", "track_ratio"]
    plots.plot_scores_timeline(
        (scores["t_ns"].to_numpy() - t0) * 1e-9,
        {name: scores[name].to_numpy() for name in names if scores[name].notna().any()},
        (keyframes["t_ns"].to_numpy() - t0) * 1e-9,
        out / ARTIFACTS["plot_scores"],
        thresholds={
            "sharp_rel": options.reject.sharp_rel_min,
            "omega_norm": options.reject.omega_max,
            "feat_coverage": options.reject.feat_coverage_min,
            "flow_med": options.trigger.flow_frac,
            "track_ratio": options.trigger.track_ratio,
        },
    )
    images = out / ARTIFACTS["images"]
    sheet = [
        (f"{idx} {trigger}", images / name.split("/", 1)[1])
        for idx, trigger, name in keyframes[["idx", "trigger", "image_name"]].itertuples(index=False)
    ]
    plots.contact_sheet(sheet, out / ARTIFACTS["plot_keyframes"])


def run_keyframes(
    sequence_dir: Path, out: Path, options: KeyframeOptions | None = None, backends: Backends | None = None
) -> StageOutput:
    """Run step 2 on the ingested sequence in `sequence_dir`, writing into the existing stage dir `out`."""
    options = options or KeyframeOptions()
    backends = backends or Backends()
    ingest = stage_dir(sequence_dir, s01_ingest.DIR_NAME)
    t_start = time.perf_counter()

    info = read_json(ingest / s01_ingest.ARTIFACTS["sequence"])
    sync_info = read_json(ingest / s01_ingest.ARTIFACTS["sync_report"])
    frames = read_csv(ingest / s01_ingest.ARTIFACTS["frame_sensor_data"], COLUMNS)
    t_ns = frames["t_ns"].to_numpy()
    sequence_id = info["sequence_id"]
    timings = {}

    # the analysis pass decodes every frame (NVDEC pays off); extraction decodes few at full size, PyAV is faster
    loader = open_loader(sequence_dir, backends.decode, backends.gpu_index)
    scores, grays = _analyze(loader, frames, options, backends)
    timings["pass_a"] = time.perf_counter() - t_start
    rejected = scores["rejected_reason"].ne("").to_numpy()
    log.info("analyzed %d frames in %.1f s, %d rejected", len(scores), timings["pass_a"], rejected.sum())

    t0 = time.perf_counter()
    ingest_manifest = read_manifest(ingest)
    selection = _select(scores, grays, _gyro_angle(ingest, ingest_manifest.artifacts), options)
    del grays
    scores["flow_med"] = np.nan if selection.flow_med is None else selection.flow_med
    scores["track_ratio"] = np.nan if selection.track_ratio is None else selection.track_ratio
    write_csv(scores, out / ARTIFACTS["frame_scores"], FRAME_SCORES_COLUMNS)
    indices = selection.indices
    names = {i: image_name(sequence_id, i) for i in indices}
    timings["select"] = time.perf_counter() - t0
    log.info(
        "%s selection: %d of %d frames (threshold scale %.3g after %d tuning passes)",
        options.method,
        len(indices),
        len(frames),
        selection.threshold_scale,
        selection.tuning_iterations,
    )

    t0 = time.perf_counter()
    loader = open_loader(sequence_dir)
    paths = loader.extract_frames(indices, out / ARTIFACTS["images"], CAMERA_ID, options.jpeg_quality)
    if len(paths) != len(indices):
        raise RuntimeError(f"extracted {len(paths)} of {len(indices)} keyframes")
    timings["extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    keyframes = _keyframes_table(scores, selection, names)
    still_names = _copy_stills(loader, sequence_id, frames, out / ARTIFACTS["images"]) if options.stills else {}
    table = pd.concat([keyframes, _stills_table(frames, still_names)], ignore_index=True)
    write_csv(table, out / ARTIFACTS["keyframes"], KEYFRAMES_COLUMNS)

    gravity = gravity_priors(frames, indices)
    gravity.insert(0, "image_name", gravity.pop("idx").map(names))
    if still_names:
        still_gravity = gravity_priors(frames, list(still_names))
        still_gravity.insert(0, "image_name", still_gravity.pop("idx").map(still_names))
        gravity = pd.concat([gravity, still_gravity], ignore_index=True)
    write_csv(gravity, out / ARTIFACTS["gravity_priors"], GRAVITY_COLUMNS)

    warnings = []
    imu_used = "sensors/gyro" in ingest_manifest.artifacts
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
    triggers = ADAPTIVE_TRIGGERS if options.method == "adaptive" else ("uniform", "forced")
    reasons = scores["rejected_reason"]
    report = {
        "num_frames": len(frames),
        "num_rejected_by_reason": {r: int((reasons == r).sum()) for r in REJECT_REASONS},
        "num_keyframes": len(indices),
        "method": options.method,
        "triggers_count": {t: selection.triggers.count(t) for t in triggers},
        "tuning_iterations": selection.tuning_iterations,
        "threshold_scale": selection.threshold_scale,
        "mean_sharp_rel_selected": float(keyframes["sharp_rel"].mean()),
        "mean_sharp_rel_all": float(scores["sharp_rel"].mean()),
        "median_gap_s": float(gaps.median()) if len(gaps) else None,
        "max_gap_s": float(gaps.max()) if len(gaps) else None,
        "imu_used": imu_used,
        "backend": {"decode": backends.decode, "metrics": backends.metrics},
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
        backends=report["backend"],
    )


@register_stage
class KeyframesStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "sequence"
    requires = (s01_ingest.STAGE,)
    config_section = "keyframes"

    def hash_inputs(self) -> dict[str, Any]:
        # backends change the metrics slightly, so a different device choice re-runs the stage
        return {"device": self.ctx.config.device, "decode_backend": self.ctx.config.loader.decode_backend}

    def run(self) -> StageOutput:
        decode = self.ctx.backend("decode", _PREFERENCE[self.ctx.config.loader.decode_backend])
        metrics = self.ctx.backend("metrics", _PREFERENCE[self.options.metrics_backend])
        backends = Backends(
            decode="nvdec" if decode == "cuda" else "pyav",
            metrics="torch" if metrics == "cuda" else "numpy",
            gpu_index=self.ctx.config.device.gpu_index,
        )
        return run_keyframes(self.ctx.sequence_dir(self.sequence_id), self.out, self.options, backends)

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
