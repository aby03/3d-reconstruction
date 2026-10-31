"""Keyframe selection from per-frame timestamps."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class KeyframeOptions:
    method: Literal["uniform"] = "uniform"
    uniform_fps: float = 3.0
    jpeg_quality: int = 95
    prior_window: int = 3  # relative-rotation priors between keyframes up to this many apart


def select_uniform(t_ns: np.ndarray, fps: float) -> list[int]:
    """For each tick at t0 + k/fps, the frame closest in time (works with variable frame rate); no duplicates."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len(t_ns) < 2:
        return list(range(len(t_ns)))
    t = (np.asarray(t_ns, dtype=np.int64) - t_ns[0]) * 1e-9
    ticks = np.arange(int(np.floor(t[-1] * fps + 1e-9)) + 1) / fps
    right = np.clip(np.searchsorted(t, ticks), 1, len(t) - 1)
    nearest = np.where(ticks - t[right - 1] <= t[right] - ticks, right - 1, right)
    return sorted(set(nearest.tolist()))
