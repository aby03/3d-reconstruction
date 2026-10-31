"""IMU priors in the COLMAP database: gravity pose priors, and a gyro check that drops pairs with a wrong rotation."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Rotation

from recon.keyframes.priors import cam_rotation_from_device

PAIR_FILTER_COLUMNS = ["image_i", "image_j", "dt_s", "check", "err_deg", "num_inliers", "removed"]
PAIR_CHECKS = ("gyro", "tilt", "none")


@dataclass(slots=True)
class PairFilterOptions:
    enabled: bool = True
    max_rot_err_deg: float = 10.0  # gyro check: image vs gyro relative rotation
    max_dt_s: float = 5.0  # pairs further apart in time (or across sequences) get the tilt check
    tilt_err_deg: float = 5.0  # tilt check: the image rotation must carry one gravity prior onto the other
    max_removed_frac: float = 0.3  # removing more than this means the IMU is suspect: remove nothing


@dataclass(slots=True)
class PriorOptions:
    use_gravity_prior: bool = True
    gravity_min_conf: float = 0.3
    imu_pair_filter: PairFilterOptions = field(default_factory=PairFilterOptions)


@dataclass(slots=True)
class GyroRotations:
    """Chained gyro rotations of one sequence: `R_cum[idx]` (xyzw, NaN where invalid) maps frame-idx device
    vectors to frame 0; `R_cam_imu` maps device to camera vectors."""

    R_cum: np.ndarray
    R_cam_imu: np.ndarray

    def cam_rotation(self, a: int, b: int) -> np.ndarray | None:
        """cam_b_from_cam_a, None when either frame has no gyro rotation."""
        qa, qb = self.R_cum[a], self.R_cum[b]
        if np.isnan(qa).any() or np.isnan(qb).any():
            return None
        R_Da_Db = (Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)).as_matrix()
        return cam_rotation_from_device(R_Da_Db, self.R_cam_imu)


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def write_gravity_priors(database: Path, gravity: pd.DataFrame, min_conf: float) -> int:
    """Gravity pose priors (unit down vector in the camera frame) for the images of `gravity`
    (`image_name, gx, gy, gz, conf`) with `conf >= min_conf`; returns how many were written."""
    rows = gravity.dropna(subset=["gx", "gy", "gz", "conf"])
    rows = rows[rows["conf"] >= min_conf].set_index("image_name")
    db = pycolmap.Database.open(database)
    try:
        written = 0
        for image in db.read_all_images():
            if image.name not in rows.index:
                continue
            g = rows.loc[image.name, ["gx", "gy", "gz"]].to_numpy(dtype=float)
            sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=image.camera_id)
            prior = pycolmap.PosePrior(
                gravity=g / np.linalg.norm(g), corr_data_id=pycolmap.data_t(sensor_id=sensor, id=image.image_id)
            )
            db.write_pose_prior(prior)
            written += 1
    finally:
        db.close()
    return written


def pair_rotation_errors(
    database: Path,
    images: pd.DataFrame,
    gyro: dict[str, GyroRotations],
    gravity: dict[str, np.ndarray],
    options: PairFilterOptions,
) -> pd.DataFrame:
    """How far each verified pair's relative rotation is from the IMU.

    `images` has `image_name, sequence_id, idx, t_ns` per keyframe; `gyro` the rotations of the sequences that have
    them; `gravity` the down vector in the camera frame per image name. Pairs of one sequence up to `max_dt_s` apart
    are checked against the gyro, other pairs by how well the rotation carries one gravity prior onto the other
    (`tilt`); pairs with neither, or without a relative pose, are `none`. Returns `PAIR_FILTER_COLUMNS` without `removed`.
    """
    info = images.set_index("image_name")
    db = pycolmap.Database.open(database)
    try:
        names = {image.image_id: image.name for image in db.read_all_images()}
        pair_ids, geometries = db.read_two_view_geometries()
    finally:
        db.close()
    rows = []
    for pair_id, tvg in zip(pair_ids, geometries):
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        name1, name2 = names[id1], names[id2]
        a, b = info.loc[name1], info.loc[name2]
        same_sequence = a["sequence_id"] == b["sequence_id"]
        dt_s = abs(int(b["t_ns"]) - int(a["t_ns"])) * 1e-9 if same_sequence else np.nan
        check, err = "none", np.nan
        rows.append([name1, name2, dt_s, check, err, len(tvg.inlier_matches)])
        if tvg.cam2_from_cam1 is None:  # no relative pose estimated (e.g. a degenerate configuration)
            continue
        R_img = tvg.cam2_from_cam1.rotation.matrix()
        R_gyro = None
        if same_sequence and dt_s <= options.max_dt_s and a["sequence_id"] in gyro:
            R_gyro = gyro[a["sequence_id"]].cam_rotation(int(a["idx"]), int(b["idx"]))
        if R_gyro is not None:
            check, err = "gyro", _angle_deg(R_img @ R_gyro.T)
        elif name1 in gravity and name2 in gravity:
            check, err = "tilt", _vector_angle_deg(R_img @ gravity[name1], gravity[name2])
        rows[-1][3:5] = check, err
    return pd.DataFrame(rows, columns=PAIR_FILTER_COLUMNS[:-1])


def filter_pairs(
    database: Path, errors: pd.DataFrame, options: PairFilterOptions
) -> tuple[pd.DataFrame, float | None]:
    """Delete the pairs whose rotation disagrees with the IMU from `database`; returns `errors` with `removed`, and
    the fraction of disagreeing pairs when that is above `max_removed_frac` (the filter then removes nothing)."""
    errors = errors.copy()
    limit = errors["check"].map({"gyro": options.max_rot_err_deg, "tilt": options.tilt_err_deg})
    disagree = (errors["err_deg"] > limit).fillna(False).astype(bool)
    disabled = options.enabled and len(errors) > 0 and disagree.mean() > options.max_removed_frac
    errors["removed"] = disagree & options.enabled & (not disabled)
    if errors["removed"].any():
        db = pycolmap.Database.open(database)
        try:
            ids = {image.name: image.image_id for image in db.read_all_images()}
            for name1, name2 in errors.loc[errors["removed"], ["image_i", "image_j"]].itertuples(index=False):
                db.delete_two_view_geometry(ids[name1], ids[name2])
                db.delete_inlier_matches(ids[name1], ids[name2])
        finally:
            db.close()
    return errors, float(disagree.mean()) if disabled else None
