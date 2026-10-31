"""Gyro integration between timestamps. All rotations are in the device frame."""

import numpy as np
from scipy.spatial.transform import Rotation

from recon.core.types import SensorStream


def _omega_at(gyro: SensorStream, t_ns: np.ndarray) -> np.ndarray:
    t = (gyro.t_ns - t_ns[0]) * 1e-9
    tq = (t_ns - t_ns[0]) * 1e-9
    return np.stack([np.interp(tq, t, gyro.values[:, d]) for d in range(3)], axis=1)


def _integrate(gyro: SensorStream, t_a_ns: int, t_b_ns: int) -> Rotation:
    if not gyro.t_ns[0] <= t_a_ns <= t_b_ns <= gyro.t_ns[-1]:
        raise ValueError(f"[{t_a_ns}, {t_b_ns}] is not inside the gyro range [{gyro.t_ns[0]}, {gyro.t_ns[-1]}]")
    i, j = np.searchsorted(gyro.t_ns, [t_a_ns, t_b_ns], side="right")
    knots = np.concatenate([[t_a_ns], gyro.t_ns[i:j], [t_b_ns]]).astype(np.int64)
    knots = knots[np.concatenate([[True], np.diff(knots) > 0])]
    if len(knots) < 2:
        return Rotation.identity()
    omega = _omega_at(gyro, knots)
    rotvecs = 0.5 * (omega[:-1] + omega[1:]) * (np.diff(knots) * 1e-9)[:, None]
    R = Rotation.identity()
    for step in Rotation.from_rotvec(rotvecs):
        R = R * step
    return R


def integrate_rotation(gyro: SensorStream, t_a_ns: int, t_b_ns: int) -> np.ndarray:
    """Rotation (3, 3) of the device at t_b relative to t_a: x_a = R @ x_b.

    Midpoint rule over the gyro samples inside [t_a, t_b], with the endpoints interpolated: R <- R Exp(w_mid dt).
    """
    return _integrate(gyro, int(t_a_ns), int(t_b_ns)).as_matrix()


def frame_rotations(gyro: SensorStream, t_frames_ns: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw: rotation of frame i relative to frame i-1, identity at index 0.

    Rows whose interval is not covered by the gyro stream are NaN.
    """
    dR = np.full((len(t_frames_ns), 4), np.nan)
    if len(t_frames_ns):
        dR[0] = [0.0, 0.0, 0.0, 1.0]
    for k in range(1, len(t_frames_ns)):
        t_a, t_b = int(t_frames_ns[k - 1]), int(t_frames_ns[k])
        if gyro.t_ns[0] <= t_a and t_b <= gyro.t_ns[-1]:
            dR[k] = _integrate(gyro, t_a, t_b).as_quat()
    return dR


def chain_rotations(dR_prev: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw: rotation of frame i relative to frame 0 (x_0 = R x_i), chained from `dR_prev`.

    Drifts with the gyro bias. Rows from the first NaN step onward are NaN.
    """
    R_cum = np.full_like(dR_prev, np.nan)
    if not len(dR_prev):
        return R_cum
    R = Rotation.from_quat(dR_prev[0])
    R_cum[0] = R.as_quat()
    for k in range(1, len(dR_prev)):
        if np.isnan(dR_prev[k]).any():
            break
        R = R * Rotation.from_quat(dR_prev[k])
        R_cum[k] = R.as_quat()
    return R_cum
