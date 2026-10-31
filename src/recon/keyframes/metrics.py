"""Per-frame image quality metrics for keyframe selection: blur, exposure and texture, on gray analysis frames."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from recon.core.types import Frame

METRIC_COLUMNS = ["sharpness", "tenengrad", "brightness", "clipped", "feat_coverage"]
SCORE_COLUMNS = [
    "idx",
    "t_ns",
    "sharpness",
    "sharp_rel",
    "tenengrad",
    "brightness",
    "clipped",
    "feat_coverage",
    "omega_norm",
    "q",
    "rejected_reason",
]
REJECT_REASONS = ["decode_error", "sharp_rel", "clipped", "feat_coverage", "omega_norm"]
CENTER_CROP = 0.8  # sharpness uses the central 80% of the frame
TENENGRAD_THRESHOLD = 50.0  # Sobel magnitudes below this count as noise
DARK, BRIGHT = 5, 250  # gray levels counted as clipped
FAST_THRESHOLD = 20
GRID = 8
MIN_CORNERS_PER_CELL = 5


@dataclass(slots=True)
class RejectOptions:
    sharp_rel_min: float = 0.55
    clipped_max: float = 0.25
    feat_coverage_min: float = 0.15
    omega_max: float = 1.5  # rad/s


@dataclass(slots=True)
class WeightOptions:
    sharp: float = 1.0
    features: float = 0.5
    omega: float = 0.7
    clipped: float = 1.0


def analysis_scale(size: tuple[int, int], long_side: int) -> float:
    """Scale that brings an image of `size` (W, H) to `long_side` on its longer side; never upscales."""
    return min(1.0, long_side / max(size))


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _center_crop(image: np.ndarray, frac: float = CENTER_CROP) -> np.ndarray:
    h, w = image.shape[:2]
    dy, dx = round(h * (1 - frac) / 2), round(w * (1 - frac) / 2)
    return image[dy : h - dy, dx : w - dx]


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 3×3 Laplacian over the central crop."""
    _, std = cv2.meanStdDev(_center_crop(cv2.Laplacian(gray, cv2.CV_32F)))
    return float(std[0, 0] ** 2)


def tenengrad(gray: np.ndarray, threshold: float = TENENGRAD_THRESHOLD) -> float:
    """Mean of the squared Sobel magnitude, with magnitudes below `threshold` set to zero."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    g2 = cv2.add(cv2.multiply(gx, gx), cv2.multiply(gy, gy))
    _, g2 = cv2.threshold(g2, threshold * threshold - 1e-3, 0, cv2.THRESH_TOZERO)
    return float(cv2.mean(g2)[0])


def exposure(gray: np.ndarray) -> tuple[float, float]:
    """Mean gray level and the fraction of pixels darker than 5 or brighter than 250."""
    return float(gray.mean()), float(((gray < DARK) | (gray > BRIGHT)).mean())


def feat_coverage(gray: np.ndarray) -> float:
    """Fraction of cells of an 8×8 grid holding at least 5 FAST corners."""
    corners = cv2.FastFeatureDetector_create(FAST_THRESHOLD).detect(gray)
    if not corners:
        return 0.0
    h, w = gray.shape[:2]
    xy = np.array([kp.pt for kp in corners])
    cx = np.minimum((xy[:, 0] * GRID / w).astype(int), GRID - 1)
    cy = np.minimum((xy[:, 1] * GRID / h).astype(int), GRID - 1)
    counts = np.bincount(cy * GRID + cx, minlength=GRID * GRID)
    return float((counts >= MIN_CORNERS_PER_CELL).mean())


def gray_metrics(gray: np.ndarray) -> list[float]:
    """The METRIC_COLUMNS values of one gray frame."""
    return [sharpness(gray), tenengrad(gray), *exposure(gray), feat_coverage(gray)]


def analyze_frames(frames: Iterable[Frame]) -> pd.DataFrame:
    """idx plus METRIC_COLUMNS for every frame yielded (frames should already be at analysis resolution)."""
    rows = [[frame.idx, *gray_metrics(to_gray(frame.image))] for frame in frames]
    return pd.DataFrame(rows, columns=["idx", *METRIC_COLUMNS]).astype({"idx": np.int64})


def rolling_median(values: np.ndarray, t_ns: np.ndarray, window_s: float) -> np.ndarray:
    """Median of the finite values within ±window_s/2 of each sample's time."""
    t = (np.asarray(t_ns, dtype=np.int64) - t_ns[0]) * 1e-9
    lo = np.searchsorted(t, t - window_s / 2, side="left")
    hi = np.searchsorted(t, t + window_s / 2, side="right")
    return np.array([np.nanmedian(values[a:b]) if np.isfinite(values[a:b]).any() else np.nan for a, b in zip(lo, hi)])


def zscore(values: np.ndarray) -> np.ndarray:
    std = np.nanstd(values)
    return (values - np.nanmean(values)) / std if std > 0 else np.zeros_like(values)


def first_reason(conditions: Sequence[tuple[str, np.ndarray]], n: int) -> np.ndarray:
    """Per frame, the name of the first condition that holds, or "" if none does."""
    reason = np.full(n, "", dtype=object)
    for name, mask in reversed(conditions):
        reason[np.asarray(mask, dtype=bool)] = name
    return reason


def score_frames(
    metrics: pd.DataFrame,
    t_ns: np.ndarray,
    omega_norm: np.ndarray,
    reject: RejectOptions,
    weights: WeightOptions,
    sharp_window_s: float = 2.0,
) -> pd.DataFrame:
    """SCORE_COLUMNS for every frame: `metrics` rows (by idx) joined to the frame times; missing rows are decode errors.

    A NaN `omega_norm` (no gyro) drops the motion terms.
    """
    n = len(t_ns)
    df = pd.DataFrame({"idx": np.arange(n), "t_ns": np.asarray(t_ns, dtype=np.int64)})
    df = df.merge(metrics, on="idx", how="left")
    sharp = df["sharpness"].to_numpy()
    df["sharp_rel"] = sharp / rolling_median(sharp, df["t_ns"].to_numpy(), sharp_window_s)
    omega = np.nan_to_num(np.asarray(omega_norm, dtype=np.float64), nan=0.0)
    df["omega_norm"] = omega_norm

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rel = np.log(df["sharp_rel"].to_numpy())
    log_rel[~np.isfinite(log_rel)] = np.nan
    df["q"] = (
        weights.sharp * zscore(log_rel)
        + weights.features * df["feat_coverage"]
        - weights.omega * omega
        - weights.clipped * np.maximum(0.0, df["clipped"] - 0.05)
    )
    decode_error = df["sharpness"].isna().to_numpy()
    df.loc[decode_error, "q"] = -np.inf
    conditions = [
        ("decode_error", decode_error),
        ("sharp_rel", df["sharp_rel"] < reject.sharp_rel_min),
        ("clipped", df["clipped"] > reject.clipped_max),
        ("feat_coverage", df["feat_coverage"] < reject.feat_coverage_min),
        ("omega_norm", omega > reject.omega_max),
    ]
    df["rejected_reason"] = first_reason(conditions, n)
    return df[SCORE_COLUMNS]
