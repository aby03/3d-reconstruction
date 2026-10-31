"""Feature extraction into a COLMAP database: one camera per image folder (`<sequence_id>/<camera_id>/`)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap


@dataclass(slots=True)
class FeatureOptions:
    type: Literal["sift"] = "sift"
    max_num_features: int = 8192
    max_image_size: int = 1920  # long side the images are scaled down to for extraction
    domain_size_pooling: bool = False  # true forces COLMAP's CPU SIFT


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
    """Extract features of `image_names` (relative to `image_root`) into `database`."""
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.type = pycolmap.FeatureExtractorType.SIFT
    extraction.max_image_size = options.max_image_size
    extraction.use_gpu = use_gpu
    extraction.gpu_index = str(gpu_index)
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
