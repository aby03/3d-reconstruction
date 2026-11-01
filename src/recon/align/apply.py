"""World frame W of a COLMAP model: gravity onto −Z, floor at z = 0, origin under the cameras, X/Y along walls."""

import logging
import time
from typing import Any

import numpy as np
import pycolmap

from recon.align import AlignOptions
from recon.align.floor import UP, FloorOptions, FloorResult, find_floor, floor_points
from recon.align.gravity import DOWN, down_vectors, estimate_gravity, residuals_deg
from recon.align.manhattan import ManhattanOptions, estimate_yaw
from recon.core.geometry import R_to_quat, Sim3, angle_deg, rotation_between
from recon.io.colmap import parse_image_name, to_sim3d
from recon.sfm.quality import camera_centres

log = logging.getLogger(__name__)

FALLBACK_PASSES = 3
FALLBACK_CONVERGED_DEG = 0.1
FLOOR_STEP_KEYS = ("detected", "plane", "tilt_to_gravity_deg", "tilt_applied", "inlier_frac", "rms")


def _refine_down_with_floor(
    down: np.ndarray, points: np.ndarray, centres: np.ndarray, options: FloorOptions, seed: int
) -> tuple[np.ndarray, list[float]]:
    """Replace `down` by the floor normal found in its gravity frame, a few times; returns it and the change of each
    pass in degrees (no floor: unchanged, empty list)."""
    changes = []
    for _ in range(FALLBACK_PASSES):
        R = rotation_between(down, DOWN)
        floor = find_floor(points @ R.T, (centres @ R.T)[:, 2], options, seed, options.fallback_max_deg)
        if floor is None:
            break
        new = R.T @ -floor.normal
        changes.append(float(angle_deg(new, down)))
        down = new
        if changes[-1] < FALLBACK_CONVERGED_DEG:
            break
    return down, changes


def _stats(r: np.ndarray, huber_deg: float) -> dict[str, Any]:
    if not len(r):
        return {"residual_median_deg": None, "residual_p90_deg": None, "residual_max_deg": None, "num_over_huber": 0}
    return {
        "residual_median_deg": float(np.median(r)),
        "residual_p90_deg": float(np.percentile(r, 90)),
        "residual_max_deg": float(r.max()),
        "num_over_huber": int((r > huber_deg).sum()),
    }


def _rotation_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _manhattan_step(
    points: np.ndarray, options: ManhattanOptions, warnings: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The `manhattan` step (yaw about Z through the origin, identity when not applied) and its report values."""
    result = estimate_yaw(points, options) if options.enabled else None
    applied = result is not None and result.confidence >= options.min_confidence
    if options.enabled and result is None:
        warnings.append("Manhattan yaw skipped: no wall-like point normals")
    info = {
        "enabled": options.enabled,
        "applied": applied,
        "yaw_deg": None if result is None else result.yaw_deg,
        "rotation_deg": None if result is None else result.rotation_deg,
        "confidence": None if result is None else result.confidence,
        "num_normals": 0 if result is None else result.num_normals,
        "extents": None if result is None else result.extents,
        "histogram": None if result is None else result.histogram.tolist(),
    }
    R = _rotation_z(-result.rotation_deg) if applied else np.eye(3)
    step = {
        "name": "manhattan",
        "R": R.tolist(),
        "t": [0.0, 0.0, 0.0],
        **{k: info[k] for k in ("yaw_deg", "rotation_deg", "confidence", "applied", "num_normals")},
    }
    return step, info


def align_model(
    rec: pycolmap.Reconstruction,
    gravity: dict[str, np.ndarray],
    conf: dict[str, float],
    options: AlignOptions | None = None,
    seed: int = 0,
) -> tuple[Sim3, list[dict[str, Any]], dict[str, Any]]:
    """`W_from_sfm` of `rec`, its steps (applied in list order) and the diagnostics.

    `gravity` holds the camera-frame down vector per image name, `conf` its confidence. Gravity comes from the priors
    (or the image-down direction refined by the floor, see `estimate_gravity`); then the floor plane gives a small
    tilt correction and z = 0, and the median camera x/y the origin. Last, a yaw about Z puts X/Y along the dominant
    walls when their direction is clear (`manhattan`, listed with `applied: false` otherwise).
    """
    options = options or AlignOptions()
    warnings: list[str] = []
    timings = {}
    t0 = time.perf_counter()
    estimate = estimate_gravity(rec, gravity, conf, options.gravity)
    centres = np.array(list(camera_centres(rec).values()))
    points = floor_points(rec, options.floor)
    down = estimate.down
    if estimate.method == "camera_up":
        reason = "no usable gravity priors" if options.gravity.method == "auto" else "gravity.method camera_up"
        warnings.append(f"{reason}: up from the image-down direction and the floor")
        if options.floor.enabled:
            down, changes = _refine_down_with_floor(down, points, centres, options.floor, seed)
            log.info("camera_up: floor refinement passes moved down by %s deg", [round(c, 3) for c in changes])
            if not changes:
                warnings.append(f"camera_up: no floor within {options.floor.fallback_max_deg}° of the image-down axis")
    R_grav = rotation_between(down, DOWN)
    timings["gravity"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    points_g, centres_g = points @ R_grav.T, centres @ R_grav.T
    floor: FloorResult | None = None
    if options.floor.enabled:
        floor = find_floor(points_g, centres_g[:, 2], options.floor, seed)
    R_tilt, tilt, tilt_applied = np.eye(3), None, False
    if floor is None:
        floor_z = float(np.percentile(points_g[:, 2], 2))
        if options.floor.enabled:
            warnings.append("no floor found: z = 0 at the 2nd percentile of the point heights")
    else:
        tilt = float(angle_deg(floor.normal, UP))
        if tilt <= options.floor.max_tilt_correction_deg:
            R_tilt, tilt_applied = rotation_between(floor.normal, UP), True
        else:
            warnings.append(f"floor normal is {tilt:.2f}° from gravity: tilt not corrected (gravity or floor suspect)")
        floor_z = float(np.median((points_g[floor.inliers] @ R_tilt.T)[:, 2]))
    centres_r = centres_g @ R_tilt.T
    t = -np.array([np.median(centres_r[:, 0]), np.median(centres_r[:, 1]), floor_z])
    timings["floor"] = time.perf_counter() - t0

    gravity_step, floor_step = Sim3(R=R_grav), Sim3(R=R_tilt, t=t)
    W_from_sfm = floor_step @ gravity_step
    names, d = down_vectors(rec, gravity)
    used = np.array([conf[name] >= options.gravity.min_conf for name in names], dtype=bool)
    residuals = residuals_deg(d[used], W_from_sfm.R)
    stats = _stats(residuals, options.gravity.huber_deg)
    if stats["residual_median_deg"] is not None and stats["residual_median_deg"] > options.gravity.max_residual_deg:
        warnings.append(
            f"gravity residual median {stats['residual_median_deg']:.2f}° > {options.gravity.max_residual_deg}°"
            " (R_cam_imu, time offset or model suspect)"
        )
    camera_z = centres @ W_from_sfm.R.T[:, 2] + t[2]
    gravity_info = {
        "method": estimate.method,
        "num_priors": estimate.num_priors,
        "num_used": estimate.num_used,
        "iterations": estimate.iterations,
        "down_vs_model_y_deg": float(angle_deg(down, [0.0, 1.0, 0.0])),
    }
    floor_info = {
        "detected": floor is not None,
        "tilt_to_gravity_deg": tilt,
        "tilt_applied": tilt_applied,
        "inlier_frac": None if floor is None else floor.inlier_frac,
        "band_frac": None if floor is None else floor.band_frac,
        "rms": None if floor is None else floor.rms,
        "plane": None if floor is None else [*floor.normal.tolist(), floor.d],
        "cameras_above_frac": float((camera_z > 0).mean()),
        "camera_height_median": float(np.median(camera_z)),
    }
    steps = [
        {
            "name": "gravity",
            "R": R_grav.tolist(),
            "t": [0.0, 0.0, 0.0],
            **{k: gravity_info[k] for k in ("method", "num_priors", "num_used", "iterations")},
            "residual_median_deg": stats["residual_median_deg"],
            "residual_p90_deg": stats["residual_p90_deg"],
        },
        {
            "name": "floor",
            "R": R_tilt.tolist(),
            "t": t.tolist(),
            **{k: floor_info[k] for k in FLOOR_STEP_KEYS},
        },
    ]
    t0 = time.perf_counter()
    points_W = W_from_sfm.apply(np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3))
    manhattan_step, manhattan_info = _manhattan_step(points_W, options.manhattan, warnings)
    steps.append(manhattan_step)
    W_from_sfm = Sim3.from_dict(manhattan_step) @ W_from_sfm
    timings["manhattan"] = time.perf_counter() - t0
    diagnostics = {
        "gravity": {**gravity_info, **stats},
        "floor": floor_info,
        "manhattan": manhattan_info,
        "residuals_deg": residuals,
        "warnings": warnings,
        "timings": timings,
    }
    return W_from_sfm, steps, diagnostics


def apply_transform(rec: pycolmap.Reconstruction, sim3: Sim3) -> pycolmap.Reconstruction:
    """A copy of `rec` moved by `sim3`; its camera centres are checked against `sim3.apply`."""
    before = camera_centres(rec)
    out = pycolmap.Reconstruction(rec)
    out.transform(to_sim3d(sim3))
    after = camera_centres(out)
    expected = sim3.apply(np.array(list(before.values())))
    error = np.abs(np.array(list(after.values())) - expected).max(initial=0.0)
    if list(after) != list(before) or error > 1e-6 * (1.0 + np.abs(expected).max(initial=0.0)):
        raise RuntimeError(f"Reconstruction.transform disagrees with the Sim3 (max centre error {error:.3g})")
    return out


def _seconds(t_ns: int) -> str:
    """`t_ns * 1e-9` with 9 decimals, exactly (a float loses the last digits of capture-clock times)."""
    sign, t_ns = ("-", -t_ns) if t_ns < 0 else ("", t_ns)
    return f"{sign}{t_ns // 10**9}.{t_ns % 10**9:09d}"


def tum_lines(rec: pycolmap.Reconstruction, t_ns: dict[str, int]) -> dict[str, list[str]]:
    """Per sequence, one TUM line `t tx ty tz qx qy qz qw` (camera-to-world, seconds on the capture clock) per
    registered image, sorted by time. `t_ns` maps image names to their capture time."""
    rows: dict[str, list[tuple[int, str]]] = {}
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        sequence_id, _, _ = parse_image_name(image.name)
        R_WC = image.cam_from_world().rotation.matrix().T
        values = [*image.projection_center(), *R_to_quat(R_WC)]
        line = " ".join([_seconds(int(t_ns[image.name])), *(f"{v:.9f}" for v in values)])
        rows.setdefault(sequence_id, []).append((int(t_ns[image.name]), line))
    return {sid: [line for _, line in sorted(lines)] for sid, lines in sorted(rows.items())}
