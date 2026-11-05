import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

DROP_FACTOR = 1.5
MIN_DELTA_CORR = 0.95
MAX_RESIDUAL_MS = 1.0
MAX_EXTRA_TIMESTAMPS = 5  # more timestamps than frames by up to this many are matched by video time


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
    count_matched: bool = True  # False: the counts differed and both were truncated
    t_offset: int = 0  # timestamps skipped at the start so that frame i gets timestamp t_offset + i

    @property
    def ok(self) -> bool:
        return (
            self.count_matched
            and self.delta_corr > MIN_DELTA_CORR
            and self.drops_pts == self.drops_t
            and self.residual_max_ms < MAX_RESIDUAL_MS
        )


def _drops(t: np.ndarray) -> list[int]:
    dt = np.diff(t)
    return (np.flatnonzero(dt > DROP_FACTOR * np.median(dt)) + 1).tolist()


def _residual(pts_s: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    t_s = (t_ns - t_ns[0]) * 1e-9
    return t_s - np.polyval(np.polyfit(pts_s - pts_s[0], t_s, 1), pts_s - pts_s[0])


def check_frame_timing(
    pts_s: np.ndarray, frame_t_ns: np.ndarray, allow_mismatch: bool = False, max_extra: int = MAX_EXTRA_TIMESTAMPS
) -> FrameTimingCheck:
    """Check that decoded frame i belongs to timestamp t_offset + i: same frame gaps, linear relation. With more
    timestamps than frames (at most `max_extra` more), the start offset with the best linear fit is taken and
    accepted when that fit is within MAX_RESIDUAL_MS; any other count mismatch needs `allow_mismatch` (then both
    are truncated at the end to the shorter)."""
    n_pts, n_t = len(pts_s), len(frame_t_ns)
    offset, count_matched = 0, True
    if n_pts != n_t:
        msg = f"video has {n_pts} frames but there are {n_t} frame timestamps"
        matched = False
        if 0 < n_t - n_pts <= max_extra:
            fits = [float(np.abs(_residual(pts_s, frame_t_ns[k : k + n_pts])).max()) for k in range(n_t - n_pts + 1)]
            offset = int(np.flatnonzero(np.array(fits) <= min(fits) + 5e-5)[0])  # ties: the earliest start
            matched = fits[offset] * 1e3 < MAX_RESIDUAL_MS
            if matched:
                log.warning("%s: frame i takes timestamp %d + i (fit %.3f ms)", msg, offset, fits[offset] * 1e3)
            else:
                offset = 0
        if not matched:
            if not allow_mismatch:
                raise ValueError(msg)
            log.warning("%s, truncating both to %d", msg, min(n_pts, n_t))
            count_matched = False
    n = min(n_pts, n_t)
    pts_s = pts_s[:n] - pts_s[0]
    t_s = (frame_t_ns[offset : offset + n] - frame_t_ns[offset]) * 1e-9

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
        count_matched=count_matched,
        t_offset=offset,
    )
