"""Sample sensor streams at frame timestamps."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recon.core.types import CaptureSequence, FrameSensorData, SensorKind, SensorStream

PREFIX = {SensorKind.ACCEL: "accel", SensorKind.GYRO: "gyro", SensorKind.MAG: "mag"}
AXES = ("x", "y", "z")


def sensor_columns(kind: SensorKind) -> list[str]:
    return [f"{PREFIX[kind]}_{a}" for a in AXES] + [f"{PREFIX[kind]}_dt_ms"]


COLUMNS = ["idx", "t_ns", "t_s", "valid", *(c for kind in SensorKind for c in sensor_columns(kind)), "omega_norm"]


@dataclass(slots=True)
class SyncOptions:
    camera_time_offset_ns: int = 0  # added to frame timestamps to get the sensor-clock query time
    max_gap_ms: dict[SensorKind, float] = field(
        default_factory=lambda: {SensorKind.ACCEL: 20.0, SensorKind.GYRO: 20.0, SensorKind.MAG: 50.0}
    )
    required: tuple[SensorKind, ...] = (SensorKind.ACCEL, SensorKind.GYRO)


def interpolate_stream(
    stream: SensorStream, t_query_ns: np.ndarray, max_gap_ms: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate each axis at `t_query_ns`.

    Returns (values (N, D), Δt to the nearest sample in ms, valid). A query is invalid outside the
    stream's time range (its values are NaN) or when its bracketing samples are more than `max_gap_ms` apart.
    """
    t_query_ns = np.asarray(t_query_ns, dtype=np.int64)
    n_q, t = len(t_query_ns), stream.t_ns
    values = np.full((n_q, stream.values.shape[1]), np.nan)
    dt_ms = np.full(n_q, np.nan)
    if len(t) < 2 or n_q == 0:
        return values, dt_ms, np.zeros(n_q, dtype=bool)

    # Subtract in int64 first so float64 only ever holds small relative times.
    t0 = t_query_ns[0]
    ts = (t - t0) * 1e-9
    tq = (t_query_ns - t0) * 1e-9
    for d in range(values.shape[1]):
        values[:, d] = np.interp(tq, ts, stream.values[:, d])

    right = np.clip(np.searchsorted(t, t_query_ns, side="right"), 1, len(t) - 1)
    left = right - 1
    dt_ms = np.minimum(np.abs(t_query_ns - t[left]), np.abs(t[right] - t_query_ns)) * 1e-6
    in_range = (t_query_ns >= t[0]) & (t_query_ns <= t[-1])
    gap_ok = ((t[right] - t[left]) * 1e-6 <= max_gap_ms) | (dt_ms == 0)
    values[~in_range] = np.nan
    return values, dt_ms, in_range & gap_ok


def synchronize(
    sequence: CaptureSequence, opts: SyncOptions | None = None, camera_id: str = "cam0"
) -> FrameSensorData:
    """One row per frame: each sensor interpolated at the frame time, validity, and ‖ω‖."""
    opts = opts or SyncOptions()
    t_frame_ns = sequence.cameras[camera_id].frame_t_ns
    t_query_ns = t_frame_ns + opts.camera_time_offset_ns
    n = len(t_frame_ns)

    df = pd.DataFrame(
        {"idx": np.arange(n), "t_ns": t_frame_ns, "t_s": (t_frame_ns - t_frame_ns[0]) * 1e-9, "valid": True}
    )
    report = {"camera_time_offset_ns": opts.camera_time_offset_ns, "num_frames": n, "sensors": {}, "warnings": []}
    for kind in SensorKind:
        cols = sensor_columns(kind)
        stream = sequence.sensors.get(kind)
        if stream is None:
            df[cols] = np.nan
            valid = np.zeros(n, dtype=bool)
            report["warnings"].append(f"no {kind} stream")
        else:
            values, dt_ms, valid = interpolate_stream(stream, t_query_ns, opts.max_gap_ms[kind])
            df[cols[:3]] = values
            df[cols[3]] = dt_ms
            report["sensors"][str(kind)] = {
                "valid_ratio": float(valid.mean()) if n else 0.0,
                "max_dt_ms": float(np.nanmax(dt_ms)) if n else float("nan"),
                "coverage_start_ms": float(t_query_ns[0] - stream.t_ns[0]) * 1e-6,
                "coverage_end_ms": float(stream.t_ns[-1] - t_query_ns[-1]) * 1e-6,
            }
        if kind in opts.required:
            df["valid"] &= valid

    df["omega_norm"] = np.linalg.norm(df[sensor_columns(SensorKind.GYRO)[:3]].to_numpy(), axis=1)
    report["frames_valid_ratio"] = float(df["valid"].mean()) if n else 0.0
    return FrameSensorData(frames=df[COLUMNS], report=report)
