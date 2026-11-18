"""Feature extraction into a COLMAP database: one camera per image folder (`<sequence_id>/<camera_id>/`)."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap

log = logging.getLogger(__name__)

FEATURE_TYPES = ("sift", "aliked", "disk", "superpoint")
LEARNED_TYPES = ("disk", "superpoint")  # extracted and matched with torch + LightGlue (`recon.sfm.learned`)
COLMAP_LIGHTGLUE_MAX_FEATURES = 4096  # 8192 features abort COLMAP's LightGlue on a 6 GB GPU

_MATCHER_TYPES = {
    "sift": {
        "bruteforce": pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE,
        "lightglue": pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE,
    },
    "aliked": {
        "bruteforce": pycolmap.FeatureMatcherType.ALIKED_BRUTEFORCE,
        "lightglue": pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE,
    },
}
_EXTRACTOR_TYPES = {
    ("sift", None): pycolmap.FeatureExtractorType.SIFT,
    ("aliked", "n16rot"): pycolmap.FeatureExtractorType.ALIKED_N16ROT,
    ("aliked", "n32"): pycolmap.FeatureExtractorType.ALIKED_N32,
}


@dataclass(slots=True)
class FeatureOptions:
    type: Literal["sift", "aliked", "disk", "superpoint"] = "sift"
    aliked_model: Literal["n16rot", "n32"] = "n16rot"  # aliked: COLMAP's ONNX network (n16rot is rotation robust)
    matcher: Literal["bruteforce", "lightglue"] = "bruteforce"  # disk / superpoint need lightglue
    max_num_features: int = 8192  # capped for COLMAP's LightGlue
    max_image_size: int = 1920  # long side the images are scaled down to for extraction
    domain_size_pooling: bool = False  # true forces COLMAP's CPU SIFT

    def __post_init__(self) -> None:
        if self.type not in FEATURE_TYPES:
            raise ValueError(f"sfm.features.type must be one of {', '.join(FEATURE_TYPES)}, not {self.type!r}")
        if self.aliked_model not in ("n16rot", "n32"):
            raise ValueError(f"sfm.features.aliked_model must be n16rot or n32, not {self.aliked_model!r}")
        if self.matcher not in ("bruteforce", "lightglue"):
            raise ValueError(f"sfm.features.matcher must be bruteforce or lightglue, not {self.matcher!r}")
        if self.type in LEARNED_TYPES and self.matcher != "lightglue":
            raise ValueError(f"sfm.features.type {self.type} needs sfm.features.matcher lightglue, not {self.matcher!r}")


def colmap_types(options: FeatureOptions) -> tuple[pycolmap.FeatureExtractorType, pycolmap.FeatureMatcherType]:
    """COLMAP's extractor and matcher enums of a `sift` / `aliked` option set."""
    model = options.aliked_model if options.type == "aliked" else None
    return _EXTRACTOR_TYPES[options.type, model], _MATCHER_TYPES[options.type][options.matcher]


def colmap_max_features(options: FeatureOptions) -> int:
    """`max_num_features`, capped for COLMAP's LightGlue."""
    if options.matcher == "lightglue" and options.max_num_features > COLMAP_LIGHTGLUE_MAX_FEATURES:
        log.warning(
            "max_num_features %d capped to %d for COLMAP's LightGlue",
            options.max_num_features,
            COLMAP_LIGHTGLUE_MAX_FEATURES,
        )
        return COLMAP_LIGHTGLUE_MAX_FEATURES
    return options.max_num_features


def device(use_gpu: bool) -> pycolmap.Device:
    return pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu


def extract_features(
    database: Path,
    image_root: Path,
    image_names: Sequence[str],
    options: FeatureOptions,
    camera_model: str = "SIMPLE_RADIAL",
    use_gpu: bool = True,
    gpu_index: int = 0,
) -> None:
    """Extract features of `image_names` (relative to `image_root`) into `database`. `disk` and `superpoint` are
    extracted by `recon.sfm.learned.extract_learned` instead."""
    if options.type in LEARNED_TYPES:
        raise ValueError(f"{options.type} features are extracted by recon.sfm.learned")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = options.max_image_size
    extraction.use_gpu = use_gpu
    extraction.gpu_index = str(gpu_index)
    extraction.type = colmap_types(options)[0]
    max_features = colmap_max_features(options)
    if options.type == "aliked":
        extraction.aliked.max_num_features = max_features
    else:
        extraction.sift.max_num_features = max_features
        extraction.sift.domain_size_pooling = options.domain_size_pooling
    pycolmap.extract_features(
        database,
        image_root,
        image_names=list(image_names),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        reader_options=reader,
        extraction_options=extraction,
        device=device(use_gpu),
    )
