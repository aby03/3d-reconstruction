"""Step 2: select keyframes, extract them at full resolution and write their IMU priors into `02_keyframes/`."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recon.core.types import SensorKind
from recon.io.tables import read_csv, read_json, write_csv, write_json
from recon.keyframes.priors import gravity_priors, relative_rotation_priors
from recon.keyframes.select import KeyframeOptions, select_uniform
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
    "gravity_priors": "priors/gravity_cam.csv",
    "relative_rotations": "priors/relative_rotations.csv",
    "selection_report": "reports/selection_report.json",
}


def image_name(sequence_id: str, idx: int, camera_id: str = CAMERA_ID) -> str:
    """Scene-wide key of a keyframe image; the file is `images/<camera_id>/<idx:06d>.jpg` in this stage dir."""
    return f"{sequence_id}/{camera_id}/{idx:06d}.jpg"


def _keyframes_table(frames: pd.DataFrame, indices: list[int], names: dict[int, str], trigger: str) -> pd.DataFrame:
    rows = frames.set_index("idx").loc[indices]
    t_ns = rows["t_ns"].to_numpy()
    gap_s = np.full(len(indices), np.nan)
    gap_s[1:] = np.diff(t_ns) * 1e-9
    return pd.DataFrame(
        {
            "idx": indices,
            "t_ns": t_ns,
            "image_name": [names[i] for i in indices],
            "quality": np.nan,
            "sharp_rel": np.nan,
            "omega_norm": rows["omega_norm"].to_numpy(),
            "trigger": trigger,
            "gap_s": gap_s,
        }
    )


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

    indices = select_uniform(t_ns, options.uniform_fps)
    names = {i: image_name(sequence_id, i) for i in indices}
    timings["select"] = time.perf_counter() - t_start
    log.info("selected %d of %d frames at %.3g fps", len(indices), len(frames), options.uniform_fps)

    t0 = time.perf_counter()
    loader = open_loader(sequence_dir)
    paths = loader.extract_frames(indices, out / ARTIFACTS["images"], CAMERA_ID, options.jpeg_quality)
    if len(paths) != len(indices):
        raise RuntimeError(f"extracted {len(paths)} of {len(indices)} keyframes")
    timings["extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    keyframes = _keyframes_table(frames, indices, names, trigger="uniform")
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

    gaps = keyframes["gap_s"].dropna()
    report = {
        "num_frames": len(frames),
        "num_keyframes": len(indices),
        "method": options.method,
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
            "median_gap_s": report["median_gap_s"],
            "max_gap_s": report["max_gap_s"],
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
