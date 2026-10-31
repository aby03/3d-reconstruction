"""Per-keyframe IMU priors: gravity in the camera frame and gyro rotations between keyframes."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from recon.core.geometry import R_to_quat
from recon.core.types import SensorStream
from recon.sensors.gyro import integrate_rotation
from recon.sensors.sync import GRAV_CAM_COLUMNS

GRAVITY_COLUMNS = ["idx", "gx", "gy", "gz", "conf"]
ROTATION_COLUMNS = ["idx_i", "idx_j", "qx", "qy", "qz", "qw", "dt_s"]


def gravity_priors(frame_sensor_data: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    """Unit down vector in the camera frame and its confidence, one row per selected frame."""
    rows = frame_sensor_data.set_index("idx").loc[list(indices)]
    df = pd.DataFrame(rows[GRAV_CAM_COLUMNS].to_numpy(), columns=["gx", "gy", "gz"])
    df.insert(0, "idx", list(indices))
    df["conf"] = rows["grav_conf"].to_numpy()
    return df


def cam_rotation_from_device(R_Di_Dj: np.ndarray, R_cam_imu: np.ndarray) -> np.ndarray:
    """cam_j_from_cam_i from the device rotation of j relative to i (x_Di = R_Di_Dj @ x_Dj)."""
    return R_cam_imu @ R_Di_Dj.T @ R_cam_imu.T


def relative_rotation_priors(
    gyro: SensorStream, t_ns: np.ndarray, indices: Sequence[int], R_cam_imu: np.ndarray, window: int
) -> pd.DataFrame:
    """cam_j_from_cam_i for every keyframe pair up to `window` apart in keyframe order.

    `t_ns` holds the sensor-clock time of every frame. Pairs not covered by the gyro stream are left out.
    """
    rows = []
    for a, i in enumerate(indices):
        for j in indices[a + 1 : a + 1 + window]:
            t_i, t_j = int(t_ns[i]), int(t_ns[j])
            if not (gyro.t_ns[0] <= t_i and t_j <= gyro.t_ns[-1]):
                continue
            R = cam_rotation_from_device(integrate_rotation(gyro, t_i, t_j), R_cam_imu)
            rows.append([i, j, *R_to_quat(R), (t_j - t_i) * 1e-9])
    df = pd.DataFrame(rows, columns=ROTATION_COLUMNS)
    return df.astype({"idx_i": np.int64, "idx_j": np.int64})
