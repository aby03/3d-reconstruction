"""Manhattan yaw: the rotation about the up axis that puts X/Y along the dominant walls, from sparse-point normals."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

HIST_BINS = 90  # 1° bins over [0, 90)
SMOOTHING = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


@dataclass(slots=True)
class ManhattanOptions:
    enabled: bool = True
    min_confidence: float = 0.25  # mass of the peak bin +- peak_halfwidth_deg bins; below: yaw not applied
    knn: int = 30  # neighbours for the point normals
    min_planarity: float = 5.0  # second / smallest eigenvalue of the neighbourhood
    max_normal_z: float = 0.2  # wall-like normals are this close to horizontal
    peak_halfwidth_deg: int = 2  # in 1° bins
    extent_tie: float = 1.1  # 90° ambiguity: X along the longer side, unless the sides are this close


@dataclass(slots=True)
class YawResult:
    yaw_deg: float  # wall direction mod 90°, in (−45, 45]
    rotation_deg: float  # direction that becomes +X: yaw_deg, or yaw_deg ∓ 90 when that side is longer
    confidence: float  # fraction of the wall-like normals in the peak bin and peak_halfwidth_deg bins either side
    num_normals: int  # wall-like normals
    extents: list[float]  # p2–p98 extent of the wall-like points along yaw_deg and yaw_deg + 90
    histogram: np.ndarray  # (90,) counts of the normal directions mod 90°, 1° bins


def point_normals(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal (N, 3) of each point from the PCA of its `k` nearest neighbours, and the planarity λ₂/λ₃
    (middle over smallest eigenvalue; large for points on a plane)."""
    k = min(k, len(points))
    _, idx = cKDTree(points).query(points, k=k)
    neighbours = points[idx]
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    planarity = eigvals[:, 1] / np.maximum(eigvals[:, 0], np.finfo(float).tiny)
    return eigvecs[:, :, 0], planarity


def _wrap45(deg: float) -> float:
    """`deg` mod 90 mapped to (−45, 45]."""
    deg = deg % 90.0
    return deg - 90.0 if deg > 45.0 else deg


def _extent(xy: np.ndarray, deg: float) -> float:
    direction = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    lo, hi = np.percentile(xy @ direction, [2, 98])
    return float(hi - lo)


def estimate_yaw(points: np.ndarray, options: ManhattanOptions) -> YawResult | None:
    """Dominant wall direction of `points` (N, 3) in a Z-up frame; None without wall-like normals."""
    if len(points) < 3:
        return None
    normals, planarity = point_normals(points, options.knn)
    wall = (planarity > options.min_planarity) & (np.abs(normals[:, 2]) < options.max_normal_z)
    if not wall.any():
        return None
    angles = np.degrees(np.arctan2(normals[wall, 1], normals[wall, 0])) % 90.0
    histogram = np.bincount(np.minimum(angles.astype(int), HIST_BINS - 1), minlength=HIST_BINS)
    half = len(SMOOTHING) // 2
    padded = np.concatenate([histogram[-half:], histogram, histogram[:half]])
    smooth = np.convolve(padded, SMOOTHING, mode="valid")
    peak = int(np.argmax(smooth))
    window = np.arange(peak - options.peak_halfwidth_deg, peak + options.peak_halfwidth_deg + 1) % HIST_BINS
    confidence = float(histogram[window].sum() / wall.sum())
    yaw = _wrap45(peak + 0.5)
    xy = points[wall, :2]
    other = yaw - 90.0 if yaw > 0 else yaw + 90.0
    extents = [_extent(xy, yaw), _extent(xy, other)]
    longer = max(extents) / max(min(extents), 1e-12)
    rotation = other if extents[1] > extents[0] and longer >= options.extent_tie else yaw
    return YawResult(yaw, rotation, confidence, int(wall.sum()), extents, histogram)
