"""Monocular depth per keyframe: Depth Anything V2 small (ONNX, onnxruntime), with a per-view affine fit of its
relative inverse depth to the SfM points the view sees."""

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

# a fixed revision of the Apache-2.0 ONNX export of Depth Anything V2 small
MODEL_REVISION = "4472b7362082ad9968fee890ca0f1e5aca36b93d"
MODEL_URL = (
    f"https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/{MODEL_REVISION}/onnx/model.onnx"
)
MODEL_SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
MODEL_FILE = "depth_anything_v2_small.onnx"
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14  # the ViT patch size: the input sides are multiples of it
MIN_DISPARITY = 1e-6


@dataclass(slots=True)
class MonoOptions:
    enabled: bool = True
    input_long_side: int = 700  # the image is resized so its long side is this (rounded to a multiple of 14)
    cache_dir: str | None = None  # None: ~/.cache/recon (or $XDG_CACHE_HOME/recon)
    min_points: int = 30  # SfM points a view needs to fit its scale and shift
    ransac_iters: int = 200
    inlier_rel: float = 0.1  # an SfM point is an inlier below this relative error of its depth
    min_inliers: int = 20
    min_inlier_ratio: float = 0.5
    max_median_residual: float = 0.05  # a fit above this median relative residual is a warning
    edge_rel: float = 0.05  # depth jumps above this fraction of the depth between neighbour pixels have no confidence


@dataclass(slots=True)
class ViewFit:
    """Affine fit `1/z = scale * d + shift` of the model output `d` to the SfM depths `z` of one view."""

    scale: float
    shift: float
    points: int
    inliers: int
    inlier_ratio: float
    median_rel_residual: float  # of the inliers, relative to their SfM depth
    median_rel_residual_all: float  # of every point
    fitted: bool
    reason: str | None = None  # why the view was skipped


def cache_dir(options: MonoOptions) -> Path:
    if options.cache_dir:
        return Path(options.cache_dir)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "recon"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_model(options: MonoOptions) -> tuple[Path, dict[str, Any]]:
    """The ONNX model file, downloaded once into the cache directory and checked against its SHA-256, and the
    record of where it came from."""
    path = cache_dir(options) / MODEL_FILE
    if not path.is_file() or _sha256(path) != MODEL_SHA256:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("downloading %s to %s", MODEL_URL, path)
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if (digest := _sha256(tmp)) != MODEL_SHA256:
            tmp.unlink()
            raise RuntimeError(f"{MODEL_URL}: SHA-256 {digest}, expected {MODEL_SHA256}")
        tmp.replace(path)
    return path, {"url": MODEL_URL, "sha256": MODEL_SHA256, "license": "Apache-2.0", "path": str(path)}


class MonoDepth:
    """Depth Anything V2 small on onnxruntime: CUDA when `use_cuda` and available, else the CPU."""

    def __init__(self, model: Path, use_cuda: bool, options: MonoOptions) -> None:
        wanted = ["CUDAExecutionProvider"] if use_cuda else []
        providers = [p for p in wanted if p in ort.get_available_providers()] + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model), providers=providers)
        self.provider = self.session.get_providers()[0]
        self.options = options

    def input_size(self, width: int, height: int) -> tuple[int, int]:
        """(width, height) of the network input for an image: long side `input_long_side`, both sides multiples
        of 14."""
        scale = self.options.input_long_side / max(width, height)
        return (max(PATCH, round(width * scale / PATCH) * PATCH), max(PATCH, round(height * scale / PATCH) * PATCH))

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth (larger = nearer, float32 (h, w) at the network input size) of a BGR image."""
        h, w = image_bgr.shape[:2]
        size = self.input_size(w, h)
        rgb = cv2.cvtColor(cv2.resize(image_bgr, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1)[None]
        return self.session.run(None, {"pixel_values": x})[0][0].astype(np.float32)


def fit_view(
    disparity: np.ndarray, uv: np.ndarray, depth: np.ndarray, options: MonoOptions, seed: int = 0
) -> ViewFit:
    """Fit `1/z = scale * d + shift` of the model output `disparity` (h, w) at the pixels `uv` (N, 2; in the
    coordinates of `disparity`) to the SfM depths `depth` (N,): RANSAC on pairs, then least squares on the
    inliers. Inliers have a relative depth error below `options.inlier_rel` and a positive fitted depth.
    The view is skipped (`fitted=False`, `reason`) with too few points, inliers or a low inlier ratio."""
    n = len(depth)
    skip = ViewFit(1.0, 0.0, n, 0, 0.0, float("nan"), float("nan"), False)
    if n < options.min_points:
        skip.reason = f"{n} points (< {options.min_points})"
        return skip
    h, w = disparity.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    d, inv = disparity[y, x].astype(float), 1.0 / depth

    def residual(scale: float, shift: float) -> np.ndarray:
        fitted = scale * d + shift
        err = np.full(n, np.inf)
        ok = fitted > MIN_DISPARITY
        err[ok] = np.abs(1.0 / fitted[ok] - depth[ok]) / depth[ok]
        return err

    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(options.ransac_iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(d[i] - d[j]) < 1e-6:
            continue
        scale = (inv[i] - inv[j]) / (d[i] - d[j])
        shift = inv[i] - scale * d[i]
        count = int((residual(scale, shift) < options.inlier_rel).sum())
        if count > best_count:
            best, best_count = (scale, shift), count
    if best is None:
        skip.reason = "no model from the point pairs"
        return skip
    scale, shift = best
    for _ in range(3):  # least squares in inverse depth on the inliers, then re-select them
        inliers = residual(scale, shift) < options.inlier_rel
        if inliers.sum() < 2:
            break
        A = np.stack([d[inliers], np.ones(int(inliers.sum()))], axis=1)
        scale, shift = np.linalg.lstsq(A, inv[inliers], rcond=None)[0]
    err = residual(scale, shift)
    inliers = err < options.inlier_rel
    ratio = float(inliers.mean())
    fit = ViewFit(
        float(scale), float(shift), n, int(inliers.sum()), ratio,
        float(np.median(err[inliers])) if inliers.any() else float("nan"),
        float(np.median(err[np.isfinite(err)])) if np.isfinite(err).any() else float("nan"),
        True,
    )
    if fit.inliers < options.min_inliers:
        fit.fitted, fit.reason = False, f"{fit.inliers} inliers (< {options.min_inliers})"
    elif ratio < options.min_inlier_ratio:
        fit.fitted, fit.reason = False, f"inlier ratio {ratio:.2f} (< {options.min_inlier_ratio})"
    elif scale <= 0:
        fit.fitted, fit.reason = False, "negative scale"
    return fit


def fitted_depth(disparity: np.ndarray, fit: ViewFit, options: MonoOptions) -> tuple[np.ndarray, np.ndarray]:
    """Metric-consistent (SfM units) depth (float32, 0 where the fit is not positive) and a uint8 confidence
    mask (1 away from depth jumps, 0 on them and where there is no depth) from a fitted view."""
    inv = fit.scale * disparity + fit.shift
    depth = np.where(inv > MIN_DISPARITY, 1.0 / np.maximum(inv, MIN_DISPARITY), 0.0).astype(np.float32)
    ok = depth > 0
    gy, gx = np.gradient(depth)
    jump = np.hypot(gx, gy) > options.edge_rel * np.maximum(depth, 1e-9)
    return depth, (ok & ~jump).astype(np.uint8)
