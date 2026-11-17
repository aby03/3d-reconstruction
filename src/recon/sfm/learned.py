"""DISK and SuperPoint features matched with LightGlue (torch, the `lightglue` package), written into a COLMAP
database so geometric verification and mapping run as for SIFT."""

import gc
import importlib.util
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import torch
from PIL import Image

from recon.sfm.features import FeatureOptions, extract_features
from recon.sfm.matching import MatchingOptions

log = logging.getLogger(__name__)

FEATURES_DIR = "learned_features"


class LearnedFeaturesError(RuntimeError):
    """The `lightglue` package is not installed."""


def check_installed() -> None:
    """Raises `LearnedFeaturesError` unless the `lightglue` package can be imported."""
    if importlib.util.find_spec("lightglue") is None:
        raise LearnedFeaturesError(
            "sfm.features.type disk / superpoint needs the lightglue package (run with `pixi run -e lightglue ...`)"
        )


def torch_device(use_gpu: bool, gpu_index: int = 0) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda", gpu_index)
    if use_gpu:
        log.warning("no CUDA device for the learned features: running on the CPU")
    return torch.device("cpu")


def resized_size(width: int, height: int, max_image_size: int) -> tuple[int, int]:
    """Size the image is scaled to for extraction: its long side at most `max_image_size`, aspect kept."""
    scale = min(1.0, max_image_size / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def colmap_keypoints(keypoints: np.ndarray, size: tuple[int, int], resized: tuple[int, int]) -> np.ndarray:
    """Pixel-centre-at-integer keypoints `(x, y)` of the resized image as COLMAP full-resolution coordinates, where
    the centre of the first pixel is (0.5, 0.5): (k + 0.5) * scale."""
    scale = np.array([size[0] / resized[0], size[1] / resized[1]], dtype=np.float64)
    return ((keypoints.astype(np.float64) + 0.5) * scale).astype(np.float32)


def _extractor(options: FeatureOptions, device: torch.device) -> Any:
    from lightglue import DISK, SuperPoint

    cls = DISK if options.type == "disk" else SuperPoint
    return cls(max_num_keypoints=options.max_num_features).eval().to(device)


def _load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)


def _feature_file(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name.replace('/', '__')}.npz"


def extract_learned(
    database: Path,
    image_root: Path,
    image_names: Sequence[str],
    options: FeatureOptions,
    cache_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    use_gpu: bool = True,
    gpu_index: int = 0,
) -> None:
    """DISK / SuperPoint keypoints of `image_names` into `database` (full-resolution COLMAP coordinates), the
    descriptors into `cache_dir` for matching. The images and cameras are created by a one-keypoint SIFT pass."""
    check_installed()
    cache_dir.mkdir(parents=True, exist_ok=True)
    stub = replace(options, type="sift", max_num_features=1, max_image_size=32, domain_size_pooling=False)
    extract_features(database, image_root, image_names, stub, camera_model, use_gpu=False)
    device = torch_device(use_gpu, gpu_index)
    extractor = _extractor(options, device)
    db = pycolmap.Database.open(database)
    try:
        db.clear_keypoints()
        db.clear_descriptors()
        for name in image_names:
            image = db.read_image_with_name(name)
            camera = db.read_camera(image.camera_id)
            size = (int(camera.width), int(camera.height))
            resized = resized_size(*size, options.max_image_size)
            with torch.no_grad():
                feats = extractor.extract(_load_image(image_root / name, resized).to(device), resize=None)
            keypoints = feats["keypoints"][0].cpu().numpy()
            descriptors = feats["descriptors"][0].cpu().numpy().astype(np.float16)
            np.savez(_feature_file(cache_dir, name), keypoints=keypoints, descriptors=descriptors, size=resized)
            db.write_keypoints(image.image_id, colmap_keypoints(keypoints, size, resized))
            del feats
    finally:
        db.close()
        del extractor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _load_features(cache_dir: Path, name: str, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(_feature_file(cache_dir, name)) as data:
        return {
            "keypoints": torch.from_numpy(data["keypoints"]).float()[None].to(device),
            "descriptors": torch.from_numpy(data["descriptors"]).float()[None].to(device),
            "image_size": torch.from_numpy(data["size"]).float()[None].to(device),
        }


def match_learned(
    database: Path,
    pairs: Sequence[tuple[str, str]],
    options: FeatureOptions,
    matching: MatchingOptions,
    cache_dir: Path,
    pair_file: Path,
    use_gpu: bool = True,
    gpu_index: int = 0,
) -> None:
    """Match `pairs` of image names with LightGlue into `database`, then verify them geometrically (COLMAP)."""
    check_installed()
    from lightglue import LightGlue

    device = torch_device(use_gpu, gpu_index)
    matcher = LightGlue(features=options.type).eval().to(device)
    db = pycolmap.Database.open(database)
    kept: list[tuple[str, str]] = []
    try:
        ids = {image.name: image.image_id for image in db.read_all_images()}
        cached: dict[str, dict[str, torch.Tensor]] = {}
        for a, b in sorted(pairs, key=lambda p: (ids[p[0]], ids[p[1]])):
            if ids[a] > ids[b]:
                a, b = b, a
            if a not in cached:
                cached = {a: _load_features(cache_dir, a, device)}  # pairs come grouped by their first image
            with torch.no_grad():
                out = matcher({"image0": cached[a], "image1": _load_features(cache_dir, b, device)})
            idx0 = out["matches0"][0]
            valid = (idx0 > -1).nonzero().squeeze(1)
            if len(valid) < matching.min_num_inliers:
                continue
            matches = np.stack([valid.cpu().numpy(), idx0[valid].cpu().numpy()], axis=1).astype(np.uint32)
            db.write_matches(ids[a], ids[b], matches[: matching.max_num_matches])
            kept.append((a, b))
    finally:
        db.close()
        del matcher
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pair_file.parent.mkdir(parents=True, exist_ok=True)
    pair_file.write_text("".join(f"{a} {b}\n" for a, b in kept))
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = matching.min_num_inliers
    verification.compute_relative_pose = True
    pycolmap.verify_matches(database, pair_file, verification)
