"""Keyframe selection from per-frame timestamps and quality scores."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from recon.keyframes.metrics import RejectOptions, WeightOptions


@dataclass(slots=True)
class KeyframeOptions:
    method: Literal["uniform"] = "uniform"
    uniform_fps: float = 3.0
    analysis_long_side: int = 960  # quality metrics are computed on frames scaled to this long side
    metrics_backend: Literal["auto", "torch", "numpy"] = "auto"  # torch: filter metrics on the GPU
    metrics_batch: int = 32  # frames per GPU batch
    reject: RejectOptions = field(default_factory=RejectOptions)
    weights: WeightOptions = field(default_factory=WeightOptions)
    jpeg_quality: int = 95
    prior_window: int = 3  # relative-rotation priors between keyframes up to this many apart


@dataclass(slots=True)
class Selection:
    indices: list[int]
    triggers: list[str]  # why each keyframe was chosen


def _best(candidates: np.ndarray, q: np.ndarray, rejected: np.ndarray) -> tuple[int, bool]:
    """The max-q candidate that is not rejected, else the max-q one; and whether it had to be forced."""
    ok = candidates[~rejected[candidates]]
    pool = ok if len(ok) else candidates
    return int(pool[np.argmax(np.nan_to_num(q[pool], nan=-np.inf))]), not len(ok)


def select_uniform(
    t_ns: np.ndarray, fps: float, q: np.ndarray | None = None, rejected: np.ndarray | None = None
) -> Selection:
    """One keyframe per tick at t0 + k/fps (works with variable frame rate).

    Without scores, the frame closest to each tick. With scores, the max-`q` frame that is not `rejected` within
    half a tick of it; if all of them are rejected, the max-`q` one with trigger "forced".
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    n = len(t_ns)
    if n < 2:
        return Selection(list(range(n)), ["uniform"] * n)
    t = (np.asarray(t_ns, dtype=np.int64) - t_ns[0]) * 1e-9
    ticks = np.arange(int(np.floor(t[-1] * fps + 1e-9)) + 1) / fps
    if q is None:
        right = np.clip(np.searchsorted(t, ticks), 1, n - 1)
        nearest = np.where(ticks - t[right - 1] <= t[right] - ticks, right - 1, right)
        indices = sorted(set(nearest.tolist()))
        return Selection(indices, ["uniform"] * len(indices))

    rejected = np.zeros(n, dtype=bool) if rejected is None else np.asarray(rejected, dtype=bool)
    half = 0.5 / fps
    lo = np.searchsorted(t, ticks - half, side="left")
    hi = np.searchsorted(t, ticks + half, side="left")
    selection = Selection([], [])
    for a, b in zip(lo, hi):
        if b > a:
            idx, forced = _best(np.arange(a, b), q, rejected)
            selection.indices.append(idx)
            selection.triggers.append("forced" if forced else "uniform")
    return selection
