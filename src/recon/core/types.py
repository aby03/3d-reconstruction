from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

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
