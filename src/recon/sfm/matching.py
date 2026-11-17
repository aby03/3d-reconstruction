"""Feature matching and two-view geometric verification of the images in a COLMAP database."""

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
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
    strategy: Literal["auto", "time", "sequential", "exhaustive"] = "auto"
    exhaustive_max_images: int = 200  # auto: exhaustive up to this many images, else time
    overlap: int = 20  # time / sequential: neighbours matched (by capture time / in name order)
    quadratic_overlap: bool = True  # time / sequential: also neighbours 2^k apart
    min_num_inliers: int = 15
    max_num_matches: int = 32768
    loop_detection: bool = True  # time / sequential: also match each loop_detection_period-th image with similar ones (time: and every image of the non-path cameras)
    loop_detection_period: int = 10
    loop_detection_num_images: int = 50
    vocab_tree_path: str | None = None  # None: COLMAP's SIFT tree, downloaded once into ~/.cache/colmap/


def resolve_strategy(num_images: int, options: MatchingOptions) -> str:
    if options.strategy != "auto":
        return options.strategy
    return "exhaustive" if num_images <= options.exhaustive_max_images else "time"


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


def sequential_pairs(names: list[str], overlap: int, quadratic: bool) -> list[tuple[str, str]]:
    """Pairs of images that are neighbours in name order (COLMAP's sequential matching without loop detection)."""
    return time_pairs(sorted(names), overlap, quadratic)


def candidate_pairs(
    names: list[str], strategy: str, options: MatchingOptions, images: pd.DataFrame | None = None
) -> list[tuple[str, str]]:
    """The image pairs `strategy` (as from `resolve_strategy`) matches: all pairs, time neighbours (`images` needs
    `image_name, sequence_id, t_ns`; without it name-order neighbours) or name-order neighbours."""
    if strategy == "exhaustive":
        ordered = sorted(names)
        return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]]
    if strategy == "time" and images is not None:
        return time_pairs(time_order(images), options.overlap, options.quadratic_overlap)
    return sequential_pairs(names, options.overlap, options.quadratic_overlap)


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
    use_gpu: bool = True,
    gpu_index: int = 0,
    vocab_tree: Path | None = None,
    images: pd.DataFrame | None = None,
    work_dir: Path | None = None,
    feature_type: str = "sift",
) -> str:
    """Match and verify image pairs in `database`; returns the strategy used. Relative poses are always computed.

    `time` needs `images` (`image_name, sequence_id, t_ns`) and a `work_dir` for its pair lists; without them it
    runs as `sequential`. With `vocab_tree`, `time` and `sequential` also detect loops (`options.loop_detection*`).
    `feature_type` `aliked` matches with COLMAP's ALIKED LightGlue (no loop detection: the vocab tree is SIFT-only).
    """
    if feature_type == "aliked":
        vocab_tree = None
    strategy = resolve_strategy(num_images, options)
    if strategy == "time" and (images is None or work_dir is None):
        strategy = "sequential"
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.gpu_index = str(gpu_index)
    matching.max_num_matches = options.max_num_matches
    if feature_type == "aliked":
        matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = options.min_num_inliers
    verification.compute_relative_pose = True
    if strategy == "exhaustive":
        pycolmap.match_exhaustive(
            database, matching_options=matching, verification_options=verification, device=device(use_gpu)
        )
    elif strategy == "time":
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
        if vocab_tree is not None:
            query_file = work_dir / "loop_queries.txt"
            query_file.write_text("".join(f"{n}\n" for n in _query_images(images, options.loop_detection_period)))
            tree = pycolmap.VocabTreePairingOptions()
            tree.vocab_tree_path = str(vocab_tree)
            tree.match_list_path = str(query_file)
            tree.num_images = options.loop_detection_num_images
            pycolmap.match_vocabtree(
                database, matching_options=matching, pairing_options=tree, verification_options=verification,
                device=device(use_gpu),
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
