"""Pose estimation and triangulation from a matched COLMAP database."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap


@dataclass(slots=True)
class MapperOptions:
    backend: Literal["global"] = "global"


def _global_options(seed: int, use_gpu: bool, gpu_index: int, use_gravity: bool) -> pycolmap.GlobalPipelineOptions:
    options = pycolmap.GlobalPipelineOptions()
    gpu = str(gpu_index)
    mapper = options.mapper
    options.random_seed = mapper.random_seed = seed
    mapper.rotation_averaging.random_seed = mapper.global_positioning.random_seed = seed
    mapper.global_positioning.use_gpu = use_gpu
    mapper.global_positioning.gpu_index = gpu
    mapper.bundle_adjustment.ceres.use_gpu = use_gpu
    mapper.bundle_adjustment.ceres.gpu_index = gpu
    mapper.ba_gpu_index = gpu
    mapper.bundle_adjustment.refine_focal_length = True
    mapper.bundle_adjustment.refine_extra_params = True
    mapper.bundle_adjustment.refine_principal_point = False
    # gravity pose priors (down in the camera frame) align the model's +y with gravity
    mapper.rotation_averaging.use_gravity = use_gravity
    return options


def largest_model(models: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction | None:
    """The model with the most registered images, ties broken by the number of points."""
    if not models:
        return None
    return max(models.values(), key=lambda rec: (rec.num_reg_images(), rec.num_points3D()))


def run_global_mapper(
    database: Path,
    image_root: Path,
    out_dir: Path,
    seed: int = 0,
    use_gpu: bool = True,
    gpu_index: int = 0,
    use_gravity: bool = False,
) -> pycolmap.Reconstruction | None:
    """Global SfM (rotation averaging, global positioning, BA); the largest model, None when there is none.

    `use_gravity` constrains rotation averaging with the gravity pose priors in the database.
    COLMAP writes every model it finds into `out_dir/<k>/`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    options = _global_options(seed, use_gpu, gpu_index, use_gravity)
    models = pycolmap.global_mapping(database, image_root, out_dir, options)
    return largest_model(models)
