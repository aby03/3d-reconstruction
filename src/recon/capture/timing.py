import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

DROP_FACTOR = 1.5
MIN_DELTA_CORR = 0.95
MAX_RESIDUAL_MS = 1.0


@dataclass(slots=True)
class FrameTimingCheck:
    num_pts: int
    num_timestamps: int
    num_frames: int  # frames kept: the shorter of the two when a mismatch is allowed
    delta_corr: float
    drops_pts: list[int]  # frame indices that follow a gap > 1.5x the median frame period
    drops_t: list[int]
    residual_max_ms: float
    residual_std_ms: float

    @property
    def ok(self) -> bool:
        return (
            self.num_pts == self.num_timestamps
            and self.delta_corr > MIN_DELTA_CORR
            and self.drops_pts == self.drops_t
            and self.residual_max_ms < MAX_RESIDUAL_MS
        )


def _drops(t: np.ndarray) -> list[int]:
    dt = np.diff(t)
    return (np.flatnonzero(dt > DROP_FACTOR * np.median(dt)) + 1).tolist()


def check_frame_timing(pts_s: np.ndarray, frame_t_ns: np.ndarray, allow_mismatch: bool = False) -> FrameTimingCheck:
    """Check that decoded frame i belongs to timestamp i: same count, same frame gaps, linear relation."""
    n_pts, n_t = len(pts_s), len(frame_t_ns)
    if n_pts != n_t:
        msg = f"video has {n_pts} frames but there are {n_t} frame timestamps"
        if not allow_mismatch:
            raise ValueError(msg)
        log.warning("%s, truncating both to %d", msg, min(n_pts, n_t))
    n = min(n_pts, n_t)
    pts_s = pts_s[:n] - pts_s[0]
    t_s = (frame_t_ns[:n] - frame_t_ns[0]) * 1e-9

    residual = t_s - np.polyval(np.polyfit(pts_s, t_s, 1), pts_s)
    return FrameTimingCheck(
        num_pts=n_pts,
        num_timestamps=n_t,
        num_frames=n,
        delta_corr=float(np.corrcoef(np.diff(pts_s), np.diff(t_s))[0, 1]),
        drops_pts=_drops(pts_s),
        drops_t=_drops(t_s),
        residual_max_ms=float(np.abs(residual).max()) * 1e3,
        residual_std_ms=float(residual.std()) * 1e3,
    )
