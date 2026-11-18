"""Feature matching and two-view geometric verification of the images in a COLMAP database."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pycolmap

from recon.sfm.features import FeatureOptions, colmap_types, device

log = logging.getLogger(__name__)

# COLMAP's SIFT vocabulary tree as `url;file name;sha256`, the model-path form COLMAP itself understands
SIFT_VOCAB_TREE = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin;"
    "vocab_tree_faiss_flickr100K_words256K.bin;"
    "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"
)


@dataclass(slots=True)
class MatchingOptions:
    strategy: Literal["auto", "time", "exhaustive"] = "auto"
    exhaustive_max_images: int = 200  # auto: exhaustive up to this many images, else time
    overlap: int = 20  # time: neighbours matched by capture time
    quadratic_overlap: bool = True  # time: also neighbours 2^k apart
    min_num_inliers: int = 15
    max_num_matches: int = 32768
    loop_detection: bool = True  # time (sift only): also match each loop_detection_period-th image with similar ones (time: and every image of the non-path cameras)
    loop_detection_period: int = 10
    loop_detection_num_images: int = 50
    vocab_tree_path: str | None = None  # None: COLMAP's SIFT tree, downloaded by COLMAP once; a path or `url;name;sha256` string


def resolve_strategy(num_images: int, options: MatchingOptions) -> str:
    if options.strategy != "auto":
        return options.strategy
    return "exhaustive" if num_images <= options.exhaustive_max_images else "time"


def time_order(images: pd.DataFrame) -> list[str]:
    """Image names ordered by sequence, then capture time (all cameras of a sequence interleaved by time; ties
    keep the table order)."""
    ordered = images.reset_index(drop=True).sort_values(["sequence_id", "t_ns"], kind="stable")
    return ordered["image_name"].tolist()


def time_pairs(names: list[str], overlap: int, quadratic: bool) -> list[tuple[str, str]]:
    """Pairs of images that are neighbours in `names` (time order): each with the next `overlap` ones, and with
    the ones 2^k further on when `quadratic`."""
    offsets = set(range(1, overlap + 1))
    if quadratic:
        offsets |= {2**k for k in range(len(names).bit_length()) if 2**k < len(names)}
    return sorted({(names[i], names[i + o]) for i in range(len(names)) for o in offsets if i + o < len(names)})


def candidate_pairs(
    names: list[str], strategy: str, options: MatchingOptions, images: pd.DataFrame | None = None
) -> list[tuple[str, str]]:
    """The image pairs the learned matcher matches: all pairs for `exhaustive`, else (`images` needs
    `image_name, sequence_id, t_ns`; without it name order) time neighbours."""
    if strategy == "exhaustive":
        ordered = sorted(names)
        return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]]
    order = time_order(images) if images is not None else sorted(names)
    return time_pairs(order, options.overlap, options.quadratic_overlap)


def _query_images(images: pd.DataFrame, period: int) -> list[str]:
    """Images that look for loop closures: every `period`-th image of the camera with most images in a sequence, and
    every image of the other cameras (stills are few and each is worth anchoring)."""
    names = []
    for _, seq in images.groupby("sequence_id", sort=True):
        group = seq["image_name"].map(lambda n: n.rsplit("/", 2)[1])
        main = group.value_counts().index[0]
        ordered = seq.assign(group=group).sort_values("t_ns", kind="stable")
        path = ordered[ordered["group"] == main]["image_name"].tolist()
        names += path[::period] + ordered[ordered["group"] != main]["image_name"].tolist()
    return names


def match_features(
    database: Path,
    num_images: int,
    options: MatchingOptions,
    features: FeatureOptions,
    use_gpu: bool = True,
    gpu_index: int = 0,
    images: pd.DataFrame | None = None,
    work_dir: Path | None = None,
) -> tuple[str, bool]:
    """Match and verify image pairs in `database` with COLMAP's matcher for `features`; returns the strategy used
    and whether loop detection ran. Relative poses are always computed.

    `time` needs `images` (`image_name, sequence_id, t_ns`) and a `work_dir` for its pair lists; without them it
    runs as `exhaustive`. For `sift` features `time` also detects loops (`options.loop_detection*`) with COLMAP's
    vocab tree, which is built on SIFT descriptors; offline (no vocab tree) it runs without.
    """
    strategy = resolve_strategy(num_images, options)
    if strategy == "time" and (images is None or work_dir is None):
        strategy = "exhaustive"
    matching = pycolmap.FeatureMatchingOptions()
    matching.type = colmap_types(features)[1]
    matching.use_gpu = use_gpu
    matching.gpu_index = str(gpu_index)
    matching.max_num_matches = options.max_num_matches
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = options.min_num_inliers
    verification.compute_relative_pose = True
    loops = False
    if strategy == "exhaustive":
        pycolmap.match_exhaustive(
            database, matching_options=matching, verification_options=verification, device=device(use_gpu)
        )
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        pairs = time_pairs(time_order(images), options.overlap, options.quadratic_overlap)
        pair_file = work_dir / "time_pairs.txt"
        pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
        imported = pycolmap.ImportedPairingOptions()
        imported.match_list_path = str(pair_file)
        pycolmap.match_image_pairs(
            database, matching_options=matching, pairing_options=imported, verification_options=verification,
            device=device(use_gpu),
        )
        if options.loop_detection and features.type == "sift":
            query_file = work_dir / "loop_queries.txt"
            query_file.write_text("".join(f"{n}\n" for n in _query_images(images, options.loop_detection_period)))
            tree = pycolmap.VocabTreePairingOptions()
            tree.vocab_tree_path = options.vocab_tree_path or SIFT_VOCAB_TREE
            tree.match_list_path = str(query_file)
            tree.num_images = options.loop_detection_num_images
            try:
                pycolmap.match_vocabtree(
                    database, matching_options=matching, pairing_options=tree, verification_options=verification,
                    device=device(use_gpu),
                )
                loops = True
            except Exception as e:  # offline: COLMAP can't fetch the vocab tree
                log.warning("loop detection is off, vocab tree unavailable: %s", e)
    return strategy, loops


def num_verified_pairs(database: Path) -> int:
    """Pairs with a two-view geometry (`Database.num_verified_image_pairs` counts every matched pair)."""
    db = pycolmap.Database.open(database)
    try:
        return len(db.read_two_view_geometries()[0])
    finally:
        db.close()
