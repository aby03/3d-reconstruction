"""Room extent from the dense point cloud: the span of each axis where the points are dense, so thin trails of
stray surface (a view through a doorway, a badly placed camera) fall outside it."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RoiOptions:
    enabled: bool = True
    bin_frac: float = 0.005  # histogram bin size, as a fraction of the diagonal
    min_density_frac: float = 0.15  # a bin is dense above this fraction of the typical bin count
    margin_frac: float = 0.01  # padding around the dense span, as a fraction of the diagonal


def room_bounds(points: np.ndarray, diag: float, options: RoiOptions) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned (min, max) of the dense span of `points` per axis: the run of histogram bins around the
    fullest bin whose count is at least `min_density_frac` of the 75th percentile of the counts of the bins holding points."""
    lo, hi = np.empty(3), np.empty(3)
    step = options.bin_frac * diag
    for axis in range(3):
        v = points[:, axis]
        edges = np.arange(v.min(), v.max() + step, step)
        counts = np.histogram(v, bins=edges)[0]
        dense = counts >= options.min_density_frac * np.percentile(counts[counts > 0], 75)
        peak = int(counts.argmax())
        first = peak
        while first > 0 and dense[first - 1]:
            first -= 1
        last = peak
        while last < len(counts) - 1 and dense[last + 1]:
            last += 1
        pad = options.margin_frac * diag
        lo[axis], hi[axis] = edges[first] - pad, edges[last + 1] + pad
    return lo, hi
