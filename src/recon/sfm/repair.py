"""Re-register cameras that sit out of place in a mapped model: the pose of a flagged camera is estimated again from
2D-3D correspondences with its registered neighbours (their verified matches and triangulated points), and kept only
when it explains clearly more of those correspondences than the old pose."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


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
        mine, theirs = (matches[:, 0], matches[:, 1]) if image.image_id < other_id else (matches[:, 1], matches[:, 0])
        points2D = other.points2D
        for k, l in zip(mine, theirs):
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
