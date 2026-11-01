"""Floor plane of a roughly gravity-aligned point cloud: the lowest big z-histogram peak below the cameras, then RANSAC.

Units are arbitrary, so every distance threshold is a fraction of the scene extent.
"""

from dataclasses import dataclass

import numpy as np
import pycolmap

from recon.core.geometry import angle_deg, fit_plane

UP = np.array([0.0, 0.0, 1.0])
PEAK_HALFWIDTH_BINS = 2  # histogram mass of a peak: within this many bins of it
BAND_HALFWIDTH_BINS = 3  # RANSAC candidates: points within this many bins of the peak


@dataclass(slots=True)
class FloorOptions:
    enabled: bool = True
    min_inlier_frac: float = 0.05  # the floor peak holds at least this fraction of the points
    max_normal_deg: float = 10.0  # floor plane normal within this of up
    cameras_above_frac: float = 0.9  # the floor lies below at least this fraction of the cameras
    max_tilt_correction_deg: float = 2.0  # rotate the floor normal onto up when it is within this of gravity
    fallback_max_deg: float = 15.0  # camera_up: the floor normal may be this far from the image-down estimate
    bin_frac: float = 0.01  # histogram bin, as a fraction of the scene extent
    ransac_iterations: int = 1000
    max_reproj_px: float = 1.5  # floor candidates: points with a smaller error
    min_track_len: int = 3  # and a longer track


@dataclass(slots=True)
class FloorResult:
    normal: np.ndarray  # (3,) unit, oriented up (+Z)
    d: float  # plane normal . x + d = 0
    z: float  # median z of the inliers
    inliers: np.ndarray  # indices into the points
    inlier_frac: float  # inliers / all points
    band_frac: float  # points in the RANSAC band / all points
    rms: float  # point-to-plane rms of the inliers


def floor_points(rec: pycolmap.Reconstruction, options: FloorOptions) -> np.ndarray:
    """(N, 3) points with a track of at least `min_track_len` images and an error below `max_reproj_px`."""
    xyz = [
        p.xyz
        for p in rec.points3D.values()
        if p.track.length() >= options.min_track_len and p.error < options.max_reproj_px
    ]
    return np.array(xyz).reshape(-1, 3)


def scene_extent(points: np.ndarray) -> float:
    """Largest side of the 2nd–98th percentile box of `points`."""
    lo, hi = np.percentile(points, [2, 98], axis=0)
    return float((hi - lo).max())


def _lowest_peak(z: np.ndarray, bin_size: float, z_max: float, min_count: float) -> float | None:
    """Centre of the lowest local maximum of the smoothed z histogram that lies below `z_max` and holds at least
    `min_count` points within `PEAK_HALFWIDTH_BINS`."""
    lo, hi = np.percentile(z, [0.5, 99.5])
    edges = np.arange(lo - 3 * bin_size, hi + 4 * bin_size, bin_size)
    counts, _ = np.histogram(z, edges)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    centres = 0.5 * (edges[:-1] + edges[1:])
    k = PEAK_HALFWIDTH_BINS
    for i in range(1, len(smooth) - 1):
        if centres[i] >= z_max:
            break
        is_peak = smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]
        if is_peak and counts[max(i - k, 0) : i + k + 1].sum() >= min_count:
            return float(centres[i])
    return None


def _ransac_plane(
    points: np.ndarray, max_normal_deg: float, threshold: float, iterations: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Inlier mask of the 3-point plane with the most points within `threshold`, among normals near +Z."""
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm * np.sign(n[2] or 1.0)
        if angle_deg(n, UP) > max_normal_deg:
            continue
        inliers = np.abs((points - a) @ n) < threshold
        if (count := int(inliers.sum())) > best_count:
            best, best_count = inliers, count
    return best


def find_floor(
    points: np.ndarray,
    camera_z: np.ndarray,
    options: FloorOptions,
    seed: int = 0,
    max_normal_deg: float | None = None,
) -> FloorResult | None:
    """The floor plane of `points` (N, 3) in a frame whose +Z is roughly up, with camera centres at heights
    `camera_z`; None when no histogram peak or plane qualifies. `max_normal_deg` overrides the option."""
    max_normal_deg = options.max_normal_deg if max_normal_deg is None else max_normal_deg
    if len(points) < 3 or not len(camera_z):
        return None
    bin_size = options.bin_frac * scene_extent(points)
    z = points[:, 2]
    z_max = float(np.percentile(camera_z, 100 * (1 - options.cameras_above_frac)))
    peak = _lowest_peak(z, bin_size, z_max, options.min_inlier_frac * len(points))
    if peak is None:
        return None
    band = np.flatnonzero(np.abs(z - peak) <= BAND_HALFWIDTH_BINS * bin_size)
    if len(band) < 3:
        return None
    threshold = 0.5 * bin_size
    rng = np.random.default_rng(seed)
    mask = _ransac_plane(points[band], max_normal_deg, threshold, options.ransac_iterations, rng)
    if mask is None:
        return None
    n, d = fit_plane(points[band[mask]])
    if n[2] < 0:
        n, d = -n, -d
    if angle_deg(n, UP) > max_normal_deg:
        return None
    inliers = band[np.abs(points[band] @ n + d) < threshold]
    distances = points[inliers] @ n + d
    return FloorResult(
        normal=n,
        d=d,
        z=float(np.median(points[inliers, 2])),
        inliers=inliers,
        inlier_frac=len(inliers) / len(points),
        band_frac=len(band) / len(points),
        rms=float(np.sqrt(np.mean(distances**2))),
    )
