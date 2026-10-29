"""Gravity direction in the device frame at frame times. All outputs are unit vectors pointing DOWN."""

import numpy as np
from scipy.signal import butter, filtfilt

from recon.core.types import SensorStream

GRAVITY = 9.81  # m/s^2
ACCEL_TOLERANCE = 0.5  # m/s^2: | |a| - g | above this is treated as motion, not gravity
OMEGA_SCALE = 1.0  # rad/s
INIT_WINDOW_NS = 500_000_000


def _sample(t_ns: np.ndarray, values: np.ndarray, t_query_ns: np.ndarray) -> np.ndarray:
    """Per-axis linear interpolation; NaN outside [t_ns[0], t_ns[-1]]."""
    t0 = t_query_ns[0]
    t, tq = (t_ns - t0) * 1e-9, (t_query_ns - t0) * 1e-9
    out = np.stack([np.interp(tq, t, values[:, d]) for d in range(values.shape[1])], axis=1)
    out[(t_query_ns < t_ns[0]) | (t_query_ns > t_ns[-1])] = np.nan
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def gravity_lowpass(
    accel: SensorStream, t_query_ns: np.ndarray, cutoff_hz: float = 1.5, order: int = 2
) -> np.ndarray:
    """Zero-phase Butterworth low-pass of the accelerometer, then down = -a/|a|. Returns (N, 3)."""
    b, a = butter(order, cutoff_hz, fs=accel.rate_hz)
    smooth = filtfilt(b, a, accel.values, axis=0)
    return -_unit(_sample(accel.t_ns, smooth, t_query_ns))


def _accel_weight(accel_norm: float) -> float:
    excess = abs(accel_norm - GRAVITY)
    return 1.0 if excess <= ACCEL_TOLERANCE else (ACCEL_TOLERANCE / excess) ** 2


def _rotate(v: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of one vector."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return v
    k = rotvec / angle
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(k, v) * s + k * (k @ v) * (1 - c)


def _mahony_pass(t_s: np.ndarray, omega: np.ndarray, accel: np.ndarray, kp: float, v0: np.ndarray) -> np.ndarray:
    """Track the down vector v in the device frame over samples in the given order (time may run backward).

    Gyro propagation: dv/dt = -omega x v. Correction: rotate v toward -a/|a| at rate kp, down-weighted
    when |a| is far from g.
    """
    v = np.empty_like(omega)
    v[0] = v0
    for k in range(1, len(t_s)):
        dt = t_s[k] - t_s[k - 1]
        a = accel[k]
        a_norm = np.linalg.norm(a)
        down = -a / a_norm
        prev = v[k - 1]
        rotvec = -0.5 * (omega[k - 1] + omega[k]) * dt + kp * _accel_weight(a_norm) * abs(dt) * np.cross(prev, down)
        nxt = _rotate(prev, rotvec)
        v[k] = nxt / np.linalg.norm(nxt)
    return v


def gravity_complementary(
    accel: SensorStream,
    gyro: SensorStream,
    t_query_ns: np.ndarray,
    kp: float = 1.0,
    bidirectional: bool = True,
) -> np.ndarray:
    """Complementary (Mahony, ki=0) filter at gyro rate, sampled at `t_query_ns`. Returns (N, 3).

    With `bidirectional`, a second pass runs backward in time and the two estimates are averaged
    (normalized sum, the SLERP midpoint of two unit vectors).
    """
    t = gyro.t_ns
    t_s = (t - t[0]) * 1e-9
    accel_at_gyro = _sample(accel.t_ns, accel.values, t)
    ok = np.isfinite(accel_at_gyro).all(axis=1)
    t, t_s, omega, acc = t[ok], t_s[ok], gyro.values[ok], accel_at_gyro[ok]

    def init(window: np.ndarray) -> np.ndarray:
        return -_unit(acc[window].mean(axis=0))

    down = _mahony_pass(t_s, omega, acc, kp, init(t - t[0] <= INIT_WINDOW_NS))
    if bidirectional:
        backward = _mahony_pass(t_s[::-1], omega[::-1], acc[::-1], kp, init(t[-1] - t <= INIT_WINDOW_NS))[::-1]
        down = _unit(down + backward)
    return _unit(_sample(t, down, t_query_ns))


def gravity_confidence(accel_norm: np.ndarray, omega_norm: np.ndarray) -> np.ndarray:
    """In [0, 1]; 1 when |a| is close to g and the device is barely rotating."""
    return np.exp(-(((accel_norm - GRAVITY) / ACCEL_TOLERANCE) ** 2)) * np.exp(-((omega_norm / OMEGA_SCALE) ** 2))
