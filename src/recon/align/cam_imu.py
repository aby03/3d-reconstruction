"""Camera–IMU diagnostics per sequence: hand-eye rotation and camera time offset from the SfM poses and the gyro.

Report only: the suggestions are never applied (that means re-running ingest → align with new values).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pycolmap

from recon.core.types import SensorStream
from recon.sensors.calibration import GyroTrack, hand_eye_rotation, rotation_residuals_deg, time_offset

EDGE_MARGIN_MS = 20.0  # keyframes stay this much further inside the gyro range than the offset search needs
MIN_GAIN = 1.2  # an offset is suggested when the cost at 0 is at least this times the best cost
WARN_ANGLE_DEG = 5.0
WARN_OFFSET_MS = 50.0


@dataclass(slots=True)
class CamImuOptions:
    enabled: bool = True
    pair_span: int = 5  # keyframe pairs up to this many apart
    min_rotation_deg: float = 2.0  # hand-eye uses pairs that rotate at least this much
    max_offset_ms: float = 100.0  # time offsets searched in +-this, 1 ms steps


@dataclass(slots=True)
class SequenceImu:
    keyframes: pd.DataFrame  # image_name, t_ns of the sequence's keyframes
    gyro: SensorStream
    R_cam_imu: np.ndarray  # (3, 3) the one ingest used (nominal)
    time_offset_ns: int  # the camera_time_offset_ns ingest used


def keyframe_pairs(
    rec: pycolmap.Reconstruction, keyframes: pd.DataFrame, span: int, t_range_ns: tuple[int, int]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Pairs of registered keyframes up to `span` apart in time order, both inside `t_range_ns`: a table
    (`image_a, image_b, t_a_ns, t_b_ns`) and the SfM rotations cam_b_from_cam_a (N, 3, 3)."""
    images = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    rows = keyframes[keyframes["image_name"].isin(images)].sort_values("t_ns")
    rows = rows[(rows["t_ns"] >= t_range_ns[0]) & (rows["t_ns"] <= t_range_ns[1])]
    names, times = rows["image_name"].tolist(), rows["t_ns"].astype(np.int64).tolist()
    R = {name: images[name].cam_from_world().rotation.matrix() for name in names}
    pairs, R_cam = [], []
    for i in range(len(names)):
        for j in range(i + 1, min(i + span + 1, len(names))):
            pairs.append((names[i], names[j], times[i], times[j]))
            R_cam.append(R[names[j]] @ R[names[i]].T)
    table = pd.DataFrame(pairs, columns=["image_a", "image_b", "t_a_ns", "t_b_ns"])
    return table, np.array(R_cam).reshape(-1, 3, 3)


def _angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R_a @ R_b.T) - 1) / 2, -1.0, 1.0))))


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def diagnose_sequence(rec: pycolmap.Reconstruction, imu: SequenceImu, options: CamImuOptions) -> dict[str, Any]:
    """Hand-eye `R_cam_imu` and time offset of one sequence, as report values (`suggestion` is None unless the
    estimate is well conditioned and the offset has a clear minimum inside the search range)."""
    track = GyroTrack.from_stream(imu.gyro)
    margin = int((options.max_offset_ms + EDGE_MARGIN_MS) * 1e6)
    lo, hi = track.range_ns
    # keyframe times on the sensor clock, as ingest used them
    keyframes = imu.keyframes.assign(t_ns=imu.keyframes["t_ns"].astype(np.int64) + imu.time_offset_ns)
    pairs, R_cam = keyframe_pairs(rec, keyframes, options.pair_span, (lo + margin, hi - margin))
    t_a, t_b = pairs["t_a_ns"].to_numpy(np.int64), pairs["t_b_ns"].to_numpy(np.int64)
    if len(pairs) < 3:
        return {"num_pairs": len(pairs), "error": "fewer than 3 keyframe pairs inside the gyro range"}

    R_dev = track.rotation(t_a, t_b)
    hand_eye = hand_eye_rotation(R_cam, R_dev, options.min_rotation_deg)
    nominal = imu.R_cam_imu
    estimate = hand_eye.R_cam_imu if hand_eye.well_conditioned else None
    offset = time_offset(R_cam, t_a, t_b, track, nominal if estimate is None else estimate, options.max_offset_ms)
    # the hand-eye rotation once more, with the gyro at the best offset
    at_offset = hand_eye_rotation(
        R_cam, track.rotation(t_a + offset.offset_ns, t_b + offset.offset_ns), options.min_rotation_deg
    )
    clear = offset.cost_at_zero >= MIN_GAIN * offset.cost_best and not offset.at_edge
    report = {
        "num_pairs": len(pairs),
        "num_rotating_pairs": hand_eye.num_pairs,
        "R_cam_imu": None if hand_eye.R_cam_imu is None else hand_eye.R_cam_imu.tolist(),
        "angle_to_nominal_deg": None if hand_eye.R_cam_imu is None else _angle_deg(hand_eye.R_cam_imu, nominal),
        "residual_median_deg": {
            "nominal": _median(rotation_residuals_deg(R_cam, R_dev, nominal)),
            "estimate": None if estimate is None else _median(rotation_residuals_deg(R_cam, R_dev, estimate)),
        },
        "axis_singular_values": hand_eye.axis_singular_values,
        "well_conditioned": hand_eye.well_conditioned,
        "time_offset_ms": offset.offset_ns * 1e-6,
        "cost_at_zero": offset.cost_at_zero,
        "cost_best": offset.cost_best,
        "sharpness": offset.sharpness,
        "at_edge": offset.at_edge,
        "corr_peak_ms": offset.corr_peak_ns * 1e-6,
        "at_offset": {
            "R_cam_imu": None if at_offset.R_cam_imu is None else at_offset.R_cam_imu.tolist(),
            "angle_to_nominal_deg": None if at_offset.R_cam_imu is None else _angle_deg(at_offset.R_cam_imu, nominal),
        },
        "suggestion": None,
    }
    if hand_eye.well_conditioned and clear:
        R_suggested = at_offset.R_cam_imu if at_offset.R_cam_imu is not None else hand_eye.R_cam_imu
        report["suggestion"] = {
            "ingest.camera_time_offset_ns": int(imu.time_offset_ns + offset.offset_ns),
            "R_cam_imu": R_suggested.tolist(),
        }
    return report


def diagnose(
    rec: pycolmap.Reconstruction, sequences: dict[str, SequenceImu], options: CamImuOptions
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """`diagnose_sequence` for every sequence, and the warnings (far from nominal, large offset, failures)."""
    reports, warnings = {}, []
    for sequence_id, imu in sequences.items():
        report = reports[sequence_id] = diagnose_sequence(rec, imu, options)
        if "error" in report:
            warnings.append(f"camera-IMU check of {sequence_id} skipped: {report['error']}")
            continue
        angle = report["angle_to_nominal_deg"]
        if angle is not None and angle > WARN_ANGLE_DEG:
            warnings.append(f"{sequence_id}: R_cam_imu estimate {angle:.1f}° from nominal (check device orientation)")
        if abs(report["time_offset_ms"]) > WARN_OFFSET_MS:
            warnings.append(f"{sequence_id}: camera time offset estimate {report['time_offset_ms']:+.0f} ms")
    return reports, warnings
