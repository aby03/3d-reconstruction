"""Feature extraction into a COLMAP database: one camera per image folder (`<sequence_id>/<camera_id>/`)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap


FEATURE_TYPES = ("sift", "aliked", "disk", "superpoint")
LEARNED_TYPES = ("disk", "superpoint")  # extracted and matched with torch + LightGlue (`recon.sfm.learned`)


@dataclass(slots=True)
class FeatureOptions:
    type: Literal["sift", "aliked", "disk", "superpoint"] = "sift"
    aliked_model: Literal["n16rot", "n32"] = "n16rot"  # aliked: COLMAP's ONNX network (n16rot is rotation robust)
    max_num_features: int = 8192
    max_image_size: int = 1920  # long side the images are scaled down to for extraction
    domain_size_pooling: bool = False  # true forces COLMAP's CPU SIFT

    def __post_init__(self) -> None:
        if self.type not in FEATURE_TYPES:
            raise ValueError(f"sfm.features.type must be one of {', '.join(FEATURE_TYPES)}, not {self.type!r}")
        if self.aliked_model not in ("n16rot", "n32"):
            raise ValueError(f"sfm.features.aliked_model must be n16rot or n32, not {self.aliked_model!r}")


def matcher_name(options: FeatureOptions) -> str:
    """The matcher that goes with the feature type, as reported."""
    return "sift_bruteforce" if options.type == "sift" else f"{options.type}_lightglue"


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
    if options.type == "aliked":
        extraction.type = getattr(pycolmap.FeatureExtractorType, f"ALIKED_{options.aliked_model.upper()}")
        extraction.aliked.max_num_features = options.max_num_features
    else:
        extraction.type = pycolmap.FeatureExtractorType.SIFT
        extraction.sift.max_num_features = options.max_num_features
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
