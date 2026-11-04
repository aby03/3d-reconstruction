"""Keyframe selection from per-frame timestamps and quality scores: uniform in time, or adaptive to image overlap."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from recon.keyframes.metrics import RejectOptions, WeightOptions
from recon.keyframes.overlap import KeyframeTracker

log = logging.getLogger(__name__)

FIRST_WINDOW_S = 0.5  # the first keyframe is the best frame of this opening span
MAX_TUNING_ITERATIONS = 8
ADAPTIVE_TRIGGERS = ("first", "flow", "track", "rotation", "forced")


@dataclass(slots=True)
class TriggerOptions:
    flow_frac: float = 0.04  # median KLT displacement since the keyframe, in image diagonals
    track_ratio: float = 0.70  # fraction of the keyframe's corners still tracked
    rot_deg: float = 8.0  # gyro rotation since the keyframe


@dataclass(slots=True)
class KeyframeOptions:
    method: Literal["adaptive", "uniform"] = "adaptive"
    uniform_fps: float = 3.0  # uniform: one keyframe per 1/uniform_fps s
    analysis_long_side: int = 960  # quality metrics are computed on frames scaled to this long side
    metrics_backend: Literal["auto", "torch", "numpy"] = "auto"  # torch: filter metrics on the GPU
    metrics_batch: int = 32  # frames per GPU batch
    reject: RejectOptions = field(default_factory=RejectOptions)
    weights: WeightOptions = field(default_factory=WeightOptions)
    min_per_s: float = 2.5  # adaptive: trigger thresholds are scaled until the count is within these rates
    max_per_s: float = 7.0
    trigger: TriggerOptions = field(default_factory=TriggerOptions)
    window_s: float = 0.25  # candidates come from [t_trigger - window_s, t_trigger + window_s / 2]
    min_track_ratio: float = 0.50  # a candidate must still track this fraction of the keyframe's corners
    min_flow_frac: float = 0.01  # and have moved at least this much (no near-duplicates)
    max_gap_s: float = 1.0  # a keyframe is forced when nothing triggered for this long
    jpeg_quality: int = 95
    prior_window: int = 3  # relative-rotation priors between keyframes up to this many apart
    stills: bool = True  # the capture's full-quality reference JPEGs join the keyframes as a second camera


@dataclass(slots=True)
class Selection:
    indices: list[int]
    triggers: list[str]  # why each keyframe was chosen
    flow_med: np.ndarray | None = None  # adaptive: per frame, relative to the keyframe before it
    track_ratio: np.ndarray | None = None
    threshold_scale: float = 1.0  # adaptive: factor applied to the trigger thresholds
    tuning_iterations: int = 0


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


def _qualifies(
    candidates: np.ndarray, rejected: np.ndarray, flow: np.ndarray, ratio: np.ndarray, options: KeyframeOptions
) -> np.ndarray:
    ok = ~rejected[candidates] & (ratio[candidates] >= options.min_track_ratio)
    return candidates[ok & (flow[candidates] >= options.min_flow_frac)]


def _greedy(
    t: np.ndarray,
    q: np.ndarray,
    rejected: np.ndarray,
    tracker: KeyframeTracker,
    angle_deg: Callable[[int, int], float],
    options: KeyframeOptions,
    scale: float,
) -> Selection:
    """One causal pass: advance from the keyframe until a trigger fires, then take the best frame around it."""
    n = len(t)
    trig = options.trigger
    flow_thr, rot_thr = scale * trig.flow_frac, scale * trig.rot_deg
    ratio_thr = 1.0 - min(0.99, scale * (1.0 - trig.track_ratio))
    w = options.window_s
    flow, ratio = np.full(n, np.nan), np.full(n, np.nan)

    key, _ = _best(np.flatnonzero(t <= t[0] + FIRST_WINDOW_S), q, rejected)
    selection = Selection([key], ["first"], flow, ratio, scale)
    pending: tuple[int, str] | None = None  # (frame where the trigger fired, trigger)

    def take(idx: int, trigger: str) -> int:
        selection.indices.append(idx)
        selection.triggers.append(trigger)
        tracker.seed(idx)
        return idx

    tracker.seed(key)
    f = key + 1
    while f < n:
        flow[f], ratio[f] = tracker.step(f)
        if t[f] - t[key] > options.max_gap_s:
            span = np.arange(key + 1, f)
            span = span[t[span] >= t[key] + options.max_gap_s - w]
            if not len(span):
                span = np.array([max(f - 1, key + 1)])
            if len(ok := _qualifies(span, rejected, flow, ratio, options)):
                span = ok
            key, pending = take(_best(span, q, rejected)[0], "forced"), None
            f = key + 1
            continue
        if pending is None:
            if flow[f] >= flow_thr:
                pending = (f, "flow")
            elif ratio[f] <= ratio_thr:
                pending = (f, "track")
            elif angle_deg(key, f) >= rot_thr:
                pending = (f, "rotation")
        if pending is not None and (t[f] >= t[pending[0]] + w / 2 or f == n - 1):
            j, trigger = pending
            span = np.arange(key + 1, f + 1)
            candidates = _qualifies(span[t[span] >= t[j] - w], rejected, flow, ratio, options)
            if len(candidates):
                best = candidates[np.argmax(np.nan_to_num(q[candidates], nan=-np.inf))]
                key, pending = take(int(best), trigger), None
                f = key + 1
                continue
            pending = (min(j + 1, f), trigger)  # nothing usable yet: slide the window forward
        f += 1
    return selection


def select_adaptive(
    t_ns: np.ndarray,
    q: np.ndarray,
    rejected: np.ndarray,
    grays: Sequence[np.ndarray | None],
    angle_deg: Callable[[int, int], float],
    options: KeyframeOptions,
) -> Selection:
    """Greedy keyframes triggered by image overlap (KLT flow, lost tracks) or gyro rotation since the last one.

    The trigger thresholds are scaled by bisection (at most 8 more passes) until the count is within
    [min_per_s, max_per_s] × duration; the selection closest to that range is returned.
    `grays` are the tracking-resolution frames; `angle_deg(i, j)` is the rotation between frames i and j (NaN
    without gyro).
    """
    t = (np.asarray(t_ns, dtype=np.int64) - t_ns[0]) * 1e-9
    rejected = np.asarray(rejected, dtype=bool)
    lo_count, hi_count = options.min_per_s * t[-1], options.max_per_s * t[-1]
    tracker = KeyframeTracker(grays)

    def miss(sel: Selection) -> float:
        return max(lo_count - len(sel.indices), len(sel.indices) - hi_count, 0.0)

    scale, too_many, too_few = 1.0, None, None  # largest scale with too many keyframes, smallest with too few
    best = None
    for iteration in range(MAX_TUNING_ITERATIONS + 1):
        sel = _greedy(t, q, rejected, tracker, angle_deg, options, scale)
        log.debug("threshold scale %.3g: %d keyframes", scale, len(sel.indices))
        if best is None or miss(sel) < miss(best):
            best = sel
        if not miss(sel):
            break
        if len(sel.indices) > hi_count:
            too_many = scale
            scale = scale * 2 if too_few is None else np.sqrt(scale * too_few)
        else:
            too_few = scale
            scale = scale / 2 if too_many is None else np.sqrt(scale * too_many)
    best.tuning_iterations = iteration
    return best
