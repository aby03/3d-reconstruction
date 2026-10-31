"""Pose estimation and triangulation from a matched COLMAP database."""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pycolmap

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MapperOptions:
    backend: Literal["global", "incremental"] = "global"
    fallback: bool = True  # retry when too few images register: global without gravity, then incremental
    min_registered_ratio: float = 0.8  # an attempt registering fewer than this fraction of the images falls back


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


def _incremental_options(seed: int, use_gpu: bool, gpu_index: int) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = options.triangulation.random_seed = seed
    options.ba_refine_focal_length = True
    options.ba_refine_extra_params = True
    options.ba_refine_principal_point = False
    options.ba_use_gpu = use_gpu
    options.ba_gpu_index = str(gpu_index)
    return options


def run_incremental_mapper(
    database: Path, image_root: Path, out_dir: Path, seed: int = 0, use_gpu: bool = True, gpu_index: int = 0
) -> pycolmap.Reconstruction | None:
    """Incremental SfM; the largest model, None when there is none. Models go to `out_dir/<k>/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    options = _incremental_options(seed, use_gpu, gpu_index)
    return largest_model(pycolmap.incremental_mapping(database, image_root, out_dir, options))


def map_with_fallback(
    database: Path,
    image_root: Path,
    work_dir: Path,
    num_images: int,
    options: MapperOptions,
    seed: int = 0,
    use_gpu: bool = True,
    gpu_index: int = 0,
    use_gravity: bool = False,
) -> tuple[pycolmap.Reconstruction | None, str | None, list[dict[str, Any]]]:
    """Map with `options.backend`; while fewer than `min_registered_ratio` of the images register (or the mapper
    fails), retry: global without gravity (if gravity was on), then incremental. Keeps the first good attempt, else
    the best by (registered, -mean reprojection). Returns the model, the mapper that made it and every attempt.

    Each attempt writes its models into `work_dir/<mapper>/`; `work_dir` is removed afterwards.
    """
    chain = ["global", "global_no_gravity", "incremental"] if options.backend == "global" else ["incremental"]
    if not use_gravity:
        chain = [m for m in chain if m != "global_no_gravity"]
    if not options.fallback:
        chain = chain[:1]
    best: tuple[tuple, pycolmap.Reconstruction | None, str | None] = ((-1, 0.0), None, None)
    attempts = []
    for mapper in chain:
        out = work_dir / mapper
        t0 = time.perf_counter()
        attempt: dict[str, Any] = {"mapper": mapper, "num_registered": 0, "registered_ratio": 0.0}
        attempt["mean_reproj_px"] = None
        try:
            if mapper == "incremental":
                rec = run_incremental_mapper(database, image_root, out, seed, use_gpu, gpu_index)
            else:
                gravity = use_gravity and mapper == "global"
                rec = run_global_mapper(database, image_root, out, seed, use_gpu, gpu_index, gravity)
            attempt["error"] = None if rec is not None else "no model"
        except Exception as e:  # noqa: BLE001  (a failed mapper is one attempt; the next one may work)
            rec, attempt["error"] = None, f"{type(e).__name__}: {e}"
        if rec is not None:
            attempt["num_registered"] = rec.num_reg_images()
            attempt["registered_ratio"] = rec.num_reg_images() / num_images
            attempt["mean_reproj_px"] = rec.compute_mean_reprojection_error()
        attempt["num_models"] = sum(p.is_dir() for p in out.iterdir()) if out.is_dir() else 0
        attempt["duration_s"] = round(time.perf_counter() - t0, 3)
        attempts.append(attempt)
        log.info(
            "%s mapper: %d of %d images registered in %.1f s%s",
            mapper,
            attempt["num_registered"],
            num_images,
            attempt["duration_s"],
            f" ({attempt['error']})" if attempt["error"] else "",
        )
        if rec is not None and (key := (rec.num_reg_images(), -attempt["mean_reproj_px"])) > best[0]:
            best = (key, rec, mapper)
        if attempt["registered_ratio"] >= options.min_registered_ratio:
            break
    shutil.rmtree(work_dir, ignore_errors=True)
    return best[1], best[2], attempts
