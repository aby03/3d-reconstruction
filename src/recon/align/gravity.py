"""Gravity direction of a COLMAP model from the keyframes' IMU gravity priors, or from the cameras' image-down axis."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pycolmap

from recon.core.geometry import angle_deg

DOWN = np.array([0.0, 0.0, -1.0])  # gravity in the world frame W
IMAGE_DOWN = np.array([0.0, 1.0, 0.0])  # +y of an upright camera image


@dataclass(slots=True)
class GravityOptions:
    method: Literal["auto", "imu", "camera_up"] = "auto"
    huber_deg: float = 3.0  # IRLS Huber threshold on the angle to the mean down direction
    min_conf: float = 0.3  # priors below this confidence are left out
    min_prior_frac: float = 0.2  # auto: imu needs priors on at least this fraction of the registered images
    max_residual_deg: float = 5.0  # warn above this median residual


@dataclass(slots=True)
class GravityResult:
    down: np.ndarray  # (3,) unit down direction in the model frame
    method: Literal["imu", "camera_up"]
    num_priors: int  # registered images with a finite prior
    num_used: int  # of those, with conf >= min_conf
    iterations: int  # IRLS iterations (0 for camera_up)


def _rotations(rec: pycolmap.Reconstruction, names: list[str]) -> np.ndarray:
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    return np.array([by_name[name].cam_from_world().rotation.matrix() for name in names]).reshape(-1, 3, 3)


def down_vectors(rec: pycolmap.Reconstruction, gravity: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Names of the registered images that have a prior in `gravity` (camera-frame down vectors), and their down
    directions in the model frame `R_CSᵀ g` (N, 3), unit length, sorted by name."""
    registered = {rec.images[i].name for i in rec.reg_image_ids()}
    names = sorted(registered & set(gravity))
    g = np.array([gravity[name] for name in names]).reshape(-1, 3)
    d = np.einsum("nji,nj->ni", _rotations(rec, names), g)
    return names, d / np.linalg.norm(d, axis=1, keepdims=True)


def robust_mean_direction(
    d: np.ndarray, weights: np.ndarray, huber_deg: float, max_iter: int = 20
) -> tuple[np.ndarray, int]:
    """Mean unit direction of `d` (N, 3), starting from the weighted mean and reweighted with a Huber loss on the
    angle to the mean (IRLS); returns the mean and the number of iterations until it moved by < 1e-6°."""
    mean = (weights[:, None] * d).sum(axis=0)
    mean /= np.linalg.norm(mean)
    for iteration in range(1, max_iter + 1):
        r = angle_deg(d, mean)
        w = weights * np.minimum(1.0, huber_deg / np.maximum(r, 1e-12))
        new = (w[:, None] * d).sum(axis=0)
        new /= np.linalg.norm(new)
        moved = float(angle_deg(new, mean))
        mean = new
        if moved < 1e-6:
            return mean, iteration
    return mean, max_iter


def camera_down(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Mean image-down direction `R_CSᵀ (0, 1, 0)` of the registered cameras in the model frame: roughly gravity for a
    handheld capture."""
    R = np.array([rec.images[i].cam_from_world().rotation.matrix() for i in rec.reg_image_ids()])
    mean = (R.transpose(0, 2, 1) @ IMAGE_DOWN).mean(axis=0)
    return mean / np.linalg.norm(mean)


def estimate_gravity(
    rec: pycolmap.Reconstruction, gravity: dict[str, np.ndarray], conf: dict[str, float], options: GravityOptions
) -> GravityResult:
    """Down direction of `rec` from the priors (`imu`: conf-weighted, Huber IRLS), or the image-down mean
    (`camera_up`) when asked for or, with `auto`, when fewer than 3 priors or `min_prior_frac` of the registered
    images have a usable one. `imu` without enough priors raises ValueError."""
    names, d = down_vectors(rec, gravity)
    weights = np.array([conf[name] for name in names])
    used = weights >= options.min_conf
    num_used = int(used.sum())
    enough = num_used >= 3 and num_used >= options.min_prior_frac * rec.num_reg_images()
    if options.method == "imu" and not enough:
        raise ValueError(
            f"gravity.method imu needs priors: {num_used} of {rec.num_reg_images()} registered images have one"
            f" with conf >= {options.min_conf}"
        )
    if options.method == "camera_up" or not enough:
        return GravityResult(camera_down(rec), "camera_up", len(names), num_used, 0)
    down, iterations = robust_mean_direction(d[used], weights[used], options.huber_deg)
    return GravityResult(down, "imu", len(names), num_used, iterations)


def residuals_deg(d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Angle in degrees of each rotated down direction `R d_i` to the world down −Z."""
    return angle_deg(np.asarray(d).reshape(-1, 3) @ R.T, DOWN)
