"""Camera–IMU calibration checks from camera rotations (e.g. SfM) between frame pairs and the gyro stream.

Conventions: `R_cam` are cam_b_from_cam_a rotations (`x_Cb = R x_Ca`), `R_dev` the device rotations of the same pairs
as R_Da_Db (`x_Da = R x_Db`, what `integrate_rotation` returns). With `R_cam_imu` mapping device to camera vectors,
cam_b_from_cam_a = R_cam_imu · R_Da_Dbᵀ · R_cam_imuᵀ, so rotvec(R_cam) = R_cam_imu · rotvec(R_Da_Dbᵀ).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from recon.core.types import SensorStream

CLIP_DEG = 5.0  # time offset cost: residuals are clipped here so a few bad pairs don't dominate
SHARPNESS_MS = 10.0  # sharpness: cost this far from the best offset, over the best cost


@dataclass(slots=True)
class GyroTrack:
    """Cumulative device rotation at the gyro sample times: `R[k]` maps device vectors at `t_ns[k]` to the device
    frame at `t_ns[0]` (midpoint rule), interpolated by slerp between samples."""

    t_ns: np.ndarray  # (N,) int64
    R: Rotation  # (N,)

    @classmethod
    def from_stream(cls, gyro: SensorStream) -> "GyroTrack":
        dt = np.diff(gyro.t_ns) * 1e-9
        steps = Rotation.from_rotvec(0.5 * (gyro.values[:-1] + gyro.values[1:]) * dt[:, None])
        quats = np.empty((len(gyro.t_ns), 4))
        R = Rotation.identity()
        quats[0] = R.as_quat()
        for k, step in enumerate(steps, start=1):
            R = R * step
            quats[k] = R.as_quat()
        return cls(np.asarray(gyro.t_ns, dtype=np.int64), Rotation.from_quat(quats))

    @property
    def range_ns(self) -> tuple[int, int]:
        return int(self.t_ns[0]), int(self.t_ns[-1])

    def _at(self, t_ns: np.ndarray) -> Rotation:
        t0 = self.t_ns[0]
        return Slerp((self.t_ns - t0) * 1e-9, self.R)((np.asarray(t_ns, dtype=np.int64) - t0) * 1e-9)

    def rotation(self, a_ns: np.ndarray, b_ns: np.ndarray) -> np.ndarray:
        """R_Da_Db (..., 3, 3) between the times `a_ns` and `b_ns` (arrays of equal shape, inside the track)."""
        return (self._at(a_ns).inv() * self._at(b_ns)).as_matrix()


@dataclass(slots=True)
class HandEyeResult:
    R_cam_imu: np.ndarray | None  # (3, 3); None with fewer than 3 usable pairs
    num_pairs: int  # pairs rotating more than min_rotation_deg
    axis_singular_values: list[float]  # of the unit gyro rotation axes, / sqrt(num_pairs)
    well_conditioned: bool  # smallest singular value >= 0.1: the axes span 3D


@dataclass(slots=True)
class OffsetResult:
    offset_ns: int  # gyro at t_frame + offset matches the camera best
    cost: np.ndarray  # (M,) mean clipped squared residual (deg²) per offset
    offsets_ns: np.ndarray  # (M,)
    cost_at_zero: float
    cost_best: float
    sharpness: float  # mean cost SHARPNESS_MS either side of the best, over the best cost
    at_edge: bool  # the best offset is at the end of the search range
    corr_peak_ns: int  # cross-check: offset with the best correlation of camera and gyro angular speeds


def _cam_from_dev(R_dev: np.ndarray, R_cam_imu: np.ndarray) -> np.ndarray:
    """cam_b_from_cam_a predicted from R_Da_Db."""
    return R_cam_imu @ np.swapaxes(R_dev, -1, -2) @ R_cam_imu.T


def rotation_residuals_deg(R_cam: np.ndarray, R_dev: np.ndarray, R_cam_imu: np.ndarray) -> np.ndarray:
    """Angle (degrees) between each camera rotation and the one the gyro predicts with `R_cam_imu`."""
    error = R_cam @ np.swapaxes(_cam_from_dev(R_dev, R_cam_imu), -1, -2)
    return np.degrees(Rotation.from_matrix(error.reshape(-1, 3, 3)).magnitude()).reshape(error.shape[:-2])


def hand_eye_rotation(R_cam: np.ndarray, R_dev: np.ndarray, min_rotation_deg: float) -> HandEyeResult:
    """R_cam_imu from pairs (N, 3, 3) of camera and device rotations: rotvec(R_cam) = X · rotvec(R_Da_Dbᵀ), solved in
    closed form (Kabsch) over the pairs whose camera and gyro rotations both exceed `min_rotation_deg`."""
    a = Rotation.from_matrix(R_cam).as_rotvec()
    b = Rotation.from_matrix(np.swapaxes(R_dev, -1, -2)).as_rotvec()
    keep = np.minimum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)) > np.radians(min_rotation_deg)
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return HandEyeResult(None, len(a), [], False)
    axes = b / np.linalg.norm(b, axis=1, keepdims=True)
    singular = np.linalg.svd(axes, compute_uv=False) / np.sqrt(len(a))
    R, _ = Rotation.align_vectors(a, b)
    return HandEyeResult(R.as_matrix(), len(a), singular.tolist(), bool(singular[-1] >= 0.1))


def time_offset(
    R_cam: np.ndarray,
    t_a_ns: np.ndarray,
    t_b_ns: np.ndarray,
    track: GyroTrack,
    R_cam_imu: np.ndarray,
    max_offset_ms: float,
    step_ms: float = 1.0,
) -> OffsetResult:
    """The offset δ in ±`max_offset_ms` for which the gyro rotations between `t_a + δ` and `t_b + δ` best match the
    camera rotations `R_cam` (N, 3, 3) of the pairs (cost: mean of the squared residuals clipped at CLIP_DEG). Every
    shifted time must be inside the track."""
    offsets_ns = np.round(np.arange(-max_offset_ms, max_offset_ms + step_ms / 2, step_ms) * 1e6).astype(np.int64)
    a = t_a_ns[None, :] + offsets_ns[:, None]
    b = t_b_ns[None, :] + offsets_ns[:, None]
    R_dev = track.rotation(a.ravel(), b.ravel()).reshape(len(offsets_ns), len(t_a_ns), 3, 3)
    errors = rotation_residuals_deg(R_cam[None], R_dev, R_cam_imu)
    cost = (np.minimum(errors, CLIP_DEG) ** 2).mean(axis=1)
    best = int(np.argmin(cost))
    zero = int(np.argmin(np.abs(offsets_ns)))
    near = np.abs(np.abs(offsets_ns - offsets_ns[best]) - SHARPNESS_MS * 1e6) < step_ms * 1e6 / 2
    # angular speeds of the camera and of the shifted gyro over each pair
    dt = (t_b_ns - t_a_ns) * 1e-9
    speed_cam = np.linalg.norm(Rotation.from_matrix(R_cam).as_rotvec(), axis=1) / dt
    speed_dev = np.linalg.norm(Rotation.from_matrix(R_dev.reshape(-1, 3, 3)).as_rotvec(), axis=1)
    speed_dev = speed_dev.reshape(len(offsets_ns), -1) / dt
    corr = [np.corrcoef(speed_cam, s)[0, 1] for s in speed_dev]
    return OffsetResult(
        offset_ns=int(offsets_ns[best]),
        cost=cost,
        offsets_ns=offsets_ns,
        cost_at_zero=float(cost[zero]),
        cost_best=float(cost[best]),
        sharpness=float(cost[near].mean() / max(cost[best], 1e-12)) if near.any() else float("nan"),
        at_edge=best in (0, len(offsets_ns) - 1),
        corr_peak_ns=int(offsets_ns[int(np.nanargmax(corr))]),
    )
