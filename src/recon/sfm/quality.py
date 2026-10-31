"""Quality numbers of a COLMAP model: registration, tracks, reprojection errors, camera centres."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pycolmap


def _image_errors(rec: pycolmap.Reconstruction, image: pycolmap.Image) -> np.ndarray:
    """Reprojection error in pixels of every observation in `image`."""
    observed = [p for p in image.points2D if p.has_point3D()]
    if not observed:
        return np.zeros(0)
    xy = np.array([p.xy for p in observed])
    xyz = np.array([rec.points3D[p.point3D_id].xyz for p in observed])
    cam = xyz @ image.cam_from_world().rotation.matrix().T + image.cam_from_world().translation
    projected = image.camera.img_from_cam(cam, check_cheirality=False)
    return np.linalg.norm(projected - xy, axis=1)


def per_image_errors(rec: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    """Image name -> reprojection errors of its observations, for every registered image."""
    return {rec.images[i].name: _image_errors(rec, rec.images[i]) for i in rec.reg_image_ids()}


def per_image_stats(rec: pycolmap.Reconstruction) -> dict[str, dict[str, Any]]:
    """Image name -> number of observations and their mean reprojection error."""
    return {
        name: {"num_obs": len(errors), "mean_reproj": float(errors.mean()) if len(errors) else None}
        for name, errors in sorted(per_image_errors(rec).items())
    }


def reprojection_errors(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Reprojection error of every observation in pixels."""
    errors = list(per_image_errors(rec).values())
    return np.concatenate(errors) if errors else np.zeros(0)


def track_lengths(rec: pycolmap.Reconstruction) -> np.ndarray:
    return np.array([p.track.length() for p in rec.points3D.values()], dtype=int)


def camera_centres(rec: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    """Image name -> camera centre in the model frame, registered images only, by name."""
    centres = {rec.images[i].name: rec.images[i].projection_center() for i in rec.reg_image_ids()}
    return dict(sorted(centres.items()))


def model_stats(rec: pycolmap.Reconstruction, image_names: Sequence[str]) -> dict[str, Any]:
    """Registration and accuracy of `rec` against the input images `image_names`."""
    registered = {rec.images[i].name for i in rec.reg_image_ids()}
    tracks = track_lengths(rec)
    return {
        "num_images_input": len(image_names),
        "num_registered": len(registered),
        "registered_ratio": len(registered) / len(image_names) if image_names else 0.0,
        "num_points3D": rec.num_points3D(),
        "num_observations": rec.compute_num_observations(),
        "mean_reproj_px": rec.compute_mean_reprojection_error(),
        "median_track_len": float(np.median(tracks)) if len(tracks) else 0.0,
        "mean_track_len": rec.compute_mean_track_length(),
        "unregistered_images": sorted(set(image_names) - registered),
    }
