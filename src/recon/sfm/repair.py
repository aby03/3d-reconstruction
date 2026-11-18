"""Re-register cameras that sit out of place in a mapped model: the pose of a flagged camera is estimated again from
2D-3D correspondences with its registered neighbours (their verified matches and triangulated points), and kept only
when it explains clearly more of those correspondences than the old pose."""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycolmap

from recon.io.colmap import colmap_log
from recon.sfm.continuity import analyze as analyze_continuity
from recon.sfm.mapper import map_with_fallback
from recon.sfm.priors import GyroRotations
from recon.sfm.refine import polish, refine

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RepairOptions:
    enabled: bool = True
    min_signals: int = 2  # a camera is repaired when this many independent continuity checks flag it
    min_inliers: int = 30  # correspondences the new pose must explain
    inlier_px: float = 4.0  # reprojection error below which a correspondence is an inlier
    gain: float = 1.2  # the new pose needs this many times the inliers of the old one


def _correspondences(
    rec: pycolmap.Reconstruction, db: pycolmap.Database, image: pycolmap.Image, skip: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """2D points of `image` and the 3D points they match through the verified pairs with registered images not in
    `skip`; points whose track contains `image` are left out (they were triangulated with its current pose)."""
    keypoints = db.read_keypoints(image.image_id)[:, :2]
    xy, xyz = [], []
    for other_id in rec.reg_image_ids():
        other = rec.images[other_id]
        if other.name in skip or other_id == image.image_id:
            continue
        try:
            geometry = db.read_two_view_geometry(image.image_id, other_id)
        except Exception:  # no pair in the database
            continue
        matches = geometry.inlier_matches
        if matches is None or len(matches) == 0:
            continue
        mine, theirs = matches[:, 0], matches[:, 1]  # columns follow the argument order
        points2D = other.points2D
        for k, l in zip(mine, theirs):
            if k >= len(keypoints) or l >= len(points2D):
                continue
            q = points2D[l]
            if not q.has_point3D():
                continue
            point = rec.points3D[q.point3D_id]
            if any(e.image_id == image.image_id for e in point.track.elements):
                continue
            xy.append(keypoints[k])
            xyz.append(point.xyz)
    return np.array(xy).reshape(-1, 2), np.array(xyz).reshape(-1, 3)


def _inliers(camera: pycolmap.Camera, pose: pycolmap.Rigid3d, xy: np.ndarray, xyz: np.ndarray, px: float) -> int:
    cam = xyz @ pose.rotation.matrix().T + pose.translation
    ok = cam[:, 2] > 0
    projected = camera.img_from_cam(cam, check_cheirality=False)
    return int(((np.linalg.norm(projected - xy, axis=1) < px) & ok).sum())


def repair(
    rec: pycolmap.Reconstruction, database: Path, flagged: dict[str, list[str]], options: RepairOptions
) -> list[dict[str, Any]]:
    """Re-register in place the images of `flagged` (name -> signals) that have at least `min_signals` signals;
    returns one record per candidate (`image`, `signals`, `correspondences`, `inliers_old`, `inliers_new`, `moved`,
    `applied`). The model is left for a bundle adjustment by the caller."""
    candidates = {n: s for n, s in flagged.items() if len(s) >= options.min_signals}
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    records = []
    db = pycolmap.Database.open(database)
    try:
        for name, signals in sorted(candidates.items()):
            image = by_name.get(name)
            if image is None:
                continue
            xy, xyz = _correspondences(rec, db, image, set(flagged))
            record: dict[str, Any] = {"image": name, "signals": signals, "correspondences": len(xy), "applied": False}
            records.append(record)
            if len(xy) < options.min_inliers:
                continue
            result = pycolmap.estimate_and_refine_absolute_pose(xy, xyz, image.camera)
            if result is None:
                continue
            old, new = image.cam_from_world(), result["cam_from_world"]
            record["inliers_old"] = _inliers(image.camera, old, xy, xyz, options.inlier_px)
            record["inliers_new"] = _inliers(image.camera, new, xy, xyz, options.inlier_px)
            record["moved"] = float(np.linalg.norm(new.inverse().translation - old.inverse().translation))
            if record["inliers_new"] >= options.min_inliers and record["inliers_new"] > options.gain * record["inliers_old"]:
                image.frame.set_cam_from_world(image.camera_id, new)
                record["applied"] = True
    finally:
        db.close()
    return records



def remap_on_jumps(
    rec: pycolmap.Reconstruction, continuity: dict[str, Any], mapper_used: str, database: Path, image_root: Path,
    work_dir: Path, images: pd.DataFrame, gyro: dict[str, GyroRotations], options: Any, seed: int,
    use_gpu: bool, gpu_index: int, use_gravity: bool, log_file: Path | None,
) -> tuple[dict[str, Any], tuple | None]:
    """The remap report and, if kept, `(raw, mapper, attempts, rec, continuity, refine_steps, dropped)`."""
    report: dict[str, Any] = {"attempted": False, "kept": False}
    num_jumps = sum(len(v.get("jumps", [])) for v in continuity.get("sequences", {}).values())
    if not (options.mapper.retry_on_jumps and num_jumps and mapper_used == "global"):
        return report, None
    # a path in pieces is typical of the global mapper on stretches with little translation: map incrementally
    report.update(attempted=True, jumps_before=num_jumps)
    with colmap_log(log_file):
        raw2, used2, attempts2 = map_with_fallback(
            database, image_root, work_dir, len(images), replace(options.mapper, backend="incremental", fallback=False),
            seed, use_gpu, gpu_index, use_gravity,
        )
        candidate = refine(raw2, database, image_root, options.refine, use_gpu, gpu_index) if raw2 else None
    if candidate is None:
        return report, None
    rec2, steps2, dropped2 = candidate
    after = analyze_continuity(rec2, images, gyro, options.continuity)
    jumps_after = sum(len(v.get("jumps", [])) for v in after["sequences"].values())
    keep = jumps_after < num_jumps and rec2.num_reg_images() >= 0.95 * rec.num_reg_images()
    report.update(
        jumps_after=jumps_after, flagged_before=continuity["num_flagged"], flagged_after=after["num_flagged"],
        registered_before=rec.num_reg_images(), registered_after=rec2.num_reg_images(), kept=keep,
    )
    log.info(
        "path in pieces (%d jumps): incremental mapping gives %d jumps, %d -> %d flagged, registered %d -> %d, %s",
        num_jumps, jumps_after, continuity["num_flagged"], after["num_flagged"], rec.num_reg_images(),
        rec2.num_reg_images(), "kept" if keep else "discarded",
    )
    return report, (raw2, used2, attempts2, rec2, after, steps2, dropped2) if keep else None


def repair_model(
    rec: pycolmap.Reconstruction, continuity: dict[str, Any], database: Path, images: pd.DataFrame,
    gyro: dict[str, GyroRotations], options: Any, use_gpu: bool, gpu_index: int,
) -> tuple[dict[str, Any], pycolmap.Reconstruction, dict[str, Any]]:
    """The repair report and the kept model with its continuity."""
    report: dict[str, Any] = {"enabled": options.repair.enabled, "kept": False, "records": []}
    if not (options.repair.enabled and continuity.get("num_flagged_strong")):
        return report, rec, continuity
    candidate = pycolmap.Reconstruction(rec)
    report["records"] = repair(candidate, database, continuity["flagged"], options.repair)
    applied = [r["image"] for r in report["records"] if r["applied"]]
    if not applied:
        return report, rec, continuity
    polish(candidate, options.refine, use_gpu, gpu_index)
    after = analyze_continuity(candidate, images, gyro, options.continuity)
    before_err, after_err = rec.compute_mean_reprojection_error(), candidate.compute_mean_reprojection_error()
    better = after["num_flagged"] < continuity["num_flagged"] and after["num_flagged_strong"] <= continuity["num_flagged_strong"]
    keep = better and after_err <= 1.1 * before_err and candidate.num_reg_images() >= rec.num_reg_images()
    report.update(
        kept=keep, flagged_before=continuity["num_flagged"], flagged_after=after["num_flagged"],
        reproj_before=before_err, reproj_after=after_err,
    )
    log.info(
        "repaired %d image(s): flagged %d -> %d, reprojection %.3f -> %.3f px, %s", len(applied),
        continuity["num_flagged"], after["num_flagged"], before_err, after_err, "kept" if keep else "discarded",
    )
    return (report, candidate, after) if keep else (report, rec, continuity)
