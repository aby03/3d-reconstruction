"""Feature matching and two-view geometric verification of the images in a COLMAP database."""

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap

from recon.sfm.features import device

log = logging.getLogger(__name__)

# COLMAP's SIFT vocabulary tree as `url;file name;sha256`, the model-path form COLMAP itself understands
SIFT_VOCAB_TREE = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin;"
    "vocab_tree_faiss_flickr100K_words256K.bin;"
    "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"
)


@dataclass(slots=True)
class MatchingOptions:
    strategy: Literal["auto", "sequential", "exhaustive"] = "auto"
    exhaustive_max_images: int = 200  # auto: exhaustive up to this many images, else sequential
    overlap: int = 20  # sequential: neighbours matched in name order
    quadratic_overlap: bool = True  # sequential: also neighbours 2^k apart
    min_num_inliers: int = 15
    max_num_matches: int = 32768
    loop_detection: bool = True  # sequential: also match each loop_detection_period-th image with similar ones
    loop_detection_period: int = 10
    loop_detection_num_images: int = 50
    vocab_tree_path: str | None = None  # None: COLMAP's SIFT tree, downloaded once into ~/.cache/colmap/


def resolve_strategy(num_images: int, options: MatchingOptions) -> str:
    if options.strategy != "auto":
        return options.strategy
    return "exhaustive" if num_images <= options.exhaustive_max_images else "sequential"


def _cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "colmap"


def _download(url: str, path: Path, sha256: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha256:
            raise OSError(f"sha256 {digest} != {sha256}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def resolve_vocab_tree(path: str | None = None, spec: str = SIFT_VOCAB_TREE) -> Path | None:
    """The vocabulary tree file: `path` if given, else `spec` (`url;name;sha256`) from COLMAP's cache, downloaded
    there once if missing. None (with a warning) when it can't be found or downloaded."""
    if path is not None:
        if Path(path).is_file():
            return Path(path)
        log.warning("vocab tree %s not found: loop detection is off", path)
        return None
    url, name, sha256 = spec.split(";")
    cached = _cache_dir() / f"{sha256}-{name}"  # the name COLMAP gives its own downloads
    if not cached.is_file():
        log.info("downloading the vocab tree %s to %s", url, cached)
        try:
            _download(url, cached, sha256)
        except OSError as e:
            log.warning("vocab tree download failed (%s): loop detection is off", e)
            return None
    return cached


def match_features(
    database: Path,
    num_images: int,
    options: MatchingOptions,
    use_gpu: bool = True,
    gpu_index: int = 0,
    vocab_tree: Path | None = None,
) -> str:
    """Match and verify image pairs in `database`; returns the strategy used. Relative poses are always computed.

    With `vocab_tree`, sequential matching also detects loops (`options.loop_detection*`).
    """
    strategy = resolve_strategy(num_images, options)
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.gpu_index = str(gpu_index)
    matching.max_num_matches = options.max_num_matches
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = options.min_num_inliers
    verification.compute_relative_pose = True
    if strategy == "exhaustive":
        pycolmap.match_exhaustive(
            database, matching_options=matching, verification_options=verification, device=device(use_gpu)
        )
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = options.overlap
        pairing.quadratic_overlap = options.quadratic_overlap
        if vocab_tree is not None:
            pairing.loop_detection = True
            pairing.loop_detection_period = options.loop_detection_period
            pairing.loop_detection_num_images = options.loop_detection_num_images
            pairing.vocab_tree_path = vocab_tree
        pycolmap.match_sequential(
            database,
            matching_options=matching,
            pairing_options=pairing,
            verification_options=verification,
            device=device(use_gpu),
        )
    return strategy


def num_verified_pairs(database: Path) -> int:
    """Pairs with a two-view geometry (`Database.num_verified_image_pairs` counts every matched pair)."""
    db = pycolmap.Database.open(database)
    try:
        return len(db.read_two_view_geometries()[0])
    finally:
        db.close()
