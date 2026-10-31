"""Feature matching and two-view geometric verification of the images in a COLMAP database."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap

from recon.sfm.features import device


@dataclass(slots=True)
class MatchingOptions:
    strategy: Literal["auto", "sequential", "exhaustive"] = "auto"
    exhaustive_max_images: int = 200  # auto: exhaustive up to this many images, else sequential
    overlap: int = 20  # sequential: neighbours matched in name order
    quadratic_overlap: bool = True  # sequential: also neighbours 2^k apart
    min_num_inliers: int = 15
    max_num_matches: int = 32768


def resolve_strategy(num_images: int, options: MatchingOptions) -> str:
    if options.strategy != "auto":
        return options.strategy
    return "exhaustive" if num_images <= options.exhaustive_max_images else "sequential"


def match_features(
    database: Path, num_images: int, options: MatchingOptions, use_gpu: bool = True, gpu_index: int = 0
) -> str:
    """Match and verify image pairs in `database`; returns the strategy used. Relative poses are always computed."""
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
