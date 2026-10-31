"""Model refinement: retriangulation, camera model upgrade, bundle adjustment, outlier filtering, sanity checks."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.io.colmap import parse_image_name
from recon.sfm.quality import camera_centres


@dataclass(slots=True)
class RefineOptions:
    enabled: bool = True
    final_camera_model: str = "OPENCV"
    refine_principal_point: bool = False  # a second BA that also refines cx, cy
    max_reproj_px: float = 2.0  # observations above this are removed
    min_track_len: int = 3  # points seen in fewer images are removed
    min_tri_angle_deg: float = 1.5  # points with a smaller largest triangulation angle are removed
    min_image_obs: int = 15  # images with fewer observations are deregistered first (no constraint on their pose)


# initial model -> final model -> function from the initial params to the final ones
_UPGRADES = {
    ("SIMPLE_RADIAL", "OPENCV"): lambda f, cx, cy, k: [f, f, cx, cy, k, 0.0, 0.0, 0.0],
    ("RADIAL", "OPENCV"): lambda f, cx, cy, k1, k2: [f, f, cx, cy, k1, k2, 0.0, 0.0],
    ("SIMPLE_PINHOLE", "OPENCV"): lambda f, cx, cy: [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
    ("PINHOLE", "OPENCV"): lambda fx, fy, cx, cy: [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0],
}


def upgrade_camera(rec: pycolmap.Reconstruction, model: str) -> None:
    """Switch every camera of `rec` to `model`, keeping its focal, principal point and radial distortion."""
    for camera_id in rec.cameras:
        camera = rec.camera(camera_id)
        if camera.model.name == model:
            continue
        upgrade = _UPGRADES.get((camera.model.name, model))
        if upgrade is None:
            raise ValueError(f"no camera upgrade from {camera.model.name} to {model}")
        params = upgrade(*camera.params)
        camera.model = getattr(pycolmap.CameraModelId, model)
        camera.params = params


def _bundle_adjust(rec: pycolmap.Reconstruction, principal_point: bool, use_gpu: bool, gpu_index: int) -> None:
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = True
    options.refine_extra_params = True
    options.refine_principal_point = principal_point
    options.print_summary = False
    options.ceres.use_gpu = use_gpu
    options.ceres.gpu_index = str(gpu_index)
    pycolmap.bundle_adjustment(rec, options)


def _filter(rec: pycolmap.Reconstruction, options: RefineOptions) -> None:
    observations = pycolmap.ObservationManager(rec)
    observations.filter_points3D_with_large_reprojection_error(options.max_reproj_px, set(rec.point3D_ids()))
    observations.filter_points3D_with_short_tracks(options.min_track_len)
    observations.filter_points3D_with_small_triangulation_angle(options.min_tri_angle_deg, set(rec.point3D_ids()))
    observations.filter_observations_with_negative_depth()


def drop_weak_images(rec: pycolmap.Reconstruction, min_obs: int) -> list[str]:
    """Deregister the images that observe fewer than `min_obs` points; returns their names."""
    weak = [rec.images[i] for i in rec.reg_image_ids() if rec.images[i].num_points3D < min_obs]
    for image in weak:
        rec.deregister_frame(image.frame_id)
    return sorted(image.name for image in weak)


def _step(name: str, rec: pycolmap.Reconstruction) -> dict[str, Any]:
    return {
        "name": name,
        "num_points3D": rec.num_points3D(),
        "num_observations": rec.compute_num_observations(),
        "mean_reproj_px": rec.compute_mean_reprojection_error(),
    }


def refine(
    rec: pycolmap.Reconstruction,
    database: Path,
    image_root: Path,
    options: RefineOptions,
    use_gpu: bool = True,
    gpu_index: int = 0,
) -> tuple[pycolmap.Reconstruction, list[dict[str, Any]], list[str]]:
    """Refine a mapped model; returns the refined model, the point/error stats after each step and the images
    deregistered for having too few observations.

    Retriangulation runs first, with the mapper's camera model: COLMAP checks the cameras against the database.
    Every BA runs on filtered points.
    """
    steps = [_step("input", rec)]
    dropped = drop_weak_images(rec, options.min_image_obs)
    with tempfile.TemporaryDirectory() as tmp:
        rec = pycolmap.triangulate_points(rec, database, image_root, Path(tmp), clear_points=False)
    steps.append(_step("retriangulate", rec))
    _filter(rec, options)  # new points with bad geometry would make the next BA diverge
    steps.append(_step("filter_retriangulated", rec))
    upgrade_camera(rec, options.final_camera_model)
    _bundle_adjust(rec, False, use_gpu, gpu_index)
    steps.append(_step("bundle_adjustment", rec))
    if options.refine_principal_point:
        _bundle_adjust(rec, True, use_gpu, gpu_index)
        steps.append(_step("bundle_adjustment_principal_point", rec))
    _filter(rec, options)
    steps.append(_step("filter", rec))
    _bundle_adjust(rec, options.refine_principal_point, use_gpu, gpu_index)
    steps.append(_step("final_bundle_adjustment", rec))
    return rec, steps, dropped


def sanity_checks(rec: pycolmap.Reconstruction, max_jump: float = 10.0) -> list[str]:
    """Warnings about implausible intrinsics and trajectory jumps (a keyframe step over `max_jump` x the median)."""
    warnings = []
    for camera_id, camera in sorted(rec.cameras.items()):
        size = max(camera.width, camera.height)
        f = camera.mean_focal_length()
        if not 0.5 * size <= f <= 3 * size:
            warnings.append(f"camera {camera_id}: focal {f:.0f} px outside [0.5, 3] x {size}")
        k1 = camera.params[camera.extra_params_idxs()[0]] if camera.extra_params_idxs() else 0.0
        if abs(k1) >= 1:
            warnings.append(f"camera {camera_id}: |k1| = {abs(k1):.2f} >= 1")
        dx = abs(camera.principal_point_x - camera.width / 2) / camera.width
        dy = abs(camera.principal_point_y - camera.height / 2) / camera.height
        if dx > 0.1 or dy > 0.1:
            warnings.append(f"camera {camera_id}: principal point outside the central 20% of the image")
    by_sequence: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, centre in camera_centres(rec).items():
        by_sequence.setdefault(parse_image_name(name)[0], []).append((name, centre))
    for sequence_id, cameras in by_sequence.items():
        if len(cameras) < 3:
            continue
        steps = np.linalg.norm(np.diff([c for _, c in cameras], axis=0), axis=1)
        median = np.median(steps)
        jumps = [cameras[k + 1][0] for k in np.flatnonzero(steps > max_jump * median)]
        if jumps:
            warnings.append(f"{sequence_id}: trajectory jumps over {max_jump:g}x the median step at {', '.join(jumps)}")
    return warnings
