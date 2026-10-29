from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class SensorKind(StrEnum):
    ACCEL = "accel"
    GYRO = "gyro"
    MAG = "magnetic"


@dataclass(slots=True)
class SensorStream:
    """Timestamped samples of one sensor. Timestamps are int64 ns on the capture clock."""

    kind: SensorKind
    t_ns: np.ndarray  # (N,) int64, strictly increasing
    values: np.ndarray  # (N, D) float64
    units: str
    frame: str  # coordinate frame of `values`, e.g. "android_device"
    source_file: Path | None = None

    @property
    def rate_hz(self) -> float:
        if len(self.t_ns) < 2:
            return 0.0
        return 1e9 / float(np.median(np.diff(self.t_ns)))


@dataclass(slots=True)
class VideoStream:
    """One video of a capture: stream properties plus the PTS and capture-clock timestamp of every frame."""

    camera_id: str
    path: Path
    codec: str
    stored_size: tuple[int, int]  # (W, H) as encoded
    upright_size: tuple[int, int]  # (W, H) after applying rotation_deg
    rotation_deg: int  # clockwise rotation that makes stored frames upright
    num_frames: int
    nominal_fps: float
    pts_s: np.ndarray  # (num_frames,) float64 container PTS in output order
    frame_t_ns: np.ndarray  # (num_frames,) int64 capture-clock timestamp per frame


@dataclass(slots=True)
class CaptureSequence:
    """Everything a loader knows about one recording, without decoded pixels."""

    sequence_id: str
    source_type: str
    root: Path
    cameras: dict[str, VideoStream]
    sensors: dict[SensorKind, SensorStream]
    reference_images: dict[int, Path] = field(default_factory=dict)  # frame idx -> jpg
    metadata: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
