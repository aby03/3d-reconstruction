"""Undistortion of a COLMAP model and its images into PINHOLE cameras (COLMAP workspace layout)."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pycolmap

log = logging.getLogger(__name__)


@dataclass(slots=True)
class UndistortOptions:
    max_image_size: int = -1  # -1: full resolution


def undistort(
    model_dir: Path, image_root: Path, image_names: Sequence[str], out: Path, options: UndistortOptions | None = None
) -> pycolmap.Reconstruction:
    """Undistort the model in `model_dir` and the images `image_names` (relative to `image_root`) into `out`:
    `images/<name>` (resampled copies, names unchanged), `sparse/` (PINHOLE) and `stereo/`. Returns the new model."""
    options = options or UndistortOptions()
    num_registered = pycolmap.Reconstruction(model_dir).num_reg_images()
    pycolmap.undistort_images(
        output_path=out,
        input_path=model_dir,
        image_path=image_root,
        image_names=list(image_names),
        output_type="COLMAP",
        copy_policy=pycolmap.FileCopyType.copy,
        undistort_options=pycolmap.UndistortCameraOptions(max_image_size=options.max_image_size),
    )
    rec = pycolmap.Reconstruction(out / "sparse")
    if models := {cam.model.name for cam in rec.cameras.values()} - {"PINHOLE"}:
        raise RuntimeError(f"undistorted cameras are not PINHOLE: {', '.join(sorted(models))}")
    if rec.num_reg_images() != num_registered:
        raise RuntimeError(f"undistorted {rec.num_reg_images()} of {num_registered} registered images")
    log.info("undistorted %d images into %s", rec.num_reg_images(), out)
    return rec
