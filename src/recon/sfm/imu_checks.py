"""Second opinions on a registered model from the raw accelerometer and magnetometer. Neither sensor is integrated over
the whole capture: the accelerometer is integrated inside short windows of consecutive keyframes (each window fits its
own start velocity, bias and model scale, so errors never accumulate), the magnetometer is compared with the camera
orientations through one fitted offset and one constant field vector. Thresholds are robust statistics of the capture."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Rotation, Slerp

from recon.io.colmap import parse_image_name

MAD_TO_STD = 1.4826
GRAVITY = 9.80665  # m/s^2, the accelerometer's unit


@dataclass(slots=True)
class MotionOptions:
    enabled: bool = True
    window: int = 8  # consecutive keyframes per window (at least 6: 7 unknowns against 3 equations per step)
    stride: int = 4  # windows start this many keyframes apart
    max_step_s: float = 1.5  # a window never spans a gap between keyframes longer than this
    min_motion: float = 3.0  # a window is skipped when its path is shorter than this many median steps of the capture
    max_condition: float = 1e5  # ... or when the fit is worse conditioned than this
    k: float = 4.0  # a step residual is an outlier above the window median by this many scaled MADs
    min_step_factor: float = 1.0  # ... and above this many median steps of the window


@dataclass(slots=True)
class HeadingOptions:
    enabled: bool = True
    k: float = 4.0  # a camera is an outlier above the median residual by this many scaled MADs
    floor_frac: float = 0.15  # ... and the residual exceeds this fraction of the field strength


def _scale(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    return med, max(float(np.median(np.abs(x - med))) * MAD_TO_STD, 1e-9 * max(abs(med), 1.0))


def down_in_model(rec: pycolmap.Reconstruction, gravity: dict[str, np.ndarray]) -> np.ndarray | None:
    """Unit down vector of the model frame: the median over the registered images of their gravity prior carried
    into the model frame."""
    vectors = []
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        if image.name in gravity:
            vectors.append(image.cam_from_world().rotation.matrix().T @ gravity[image.name])
    if len(vectors) < 3:
        return None
    mean = np.median(np.array(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else None


def _slerp(rec_images: dict[str, Any], names: list[str], t: np.ndarray, R_cam_imu: np.ndarray) -> Slerp:
    rots = [Rotation.from_matrix(rec_images[n].cam_from_world().rotation.matrix().T @ R_cam_imu) for n in names]
    return Slerp(t, Rotation.concatenate(rots))


def _window_fit(
    P: np.ndarray, t: np.ndarray, time: np.ndarray, accel_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Fit `P[i+1] - P[i] = x dt_i + s C_i - y T_i` (start velocity `x`, bias `y`, scale `s`, all in model units) to
    one window; returns the per-step residual vectors, the solution (x, y, s), the condition number and the scale."""
    grid = np.arange(t[0], t[-1], 0.005)
    a = np.stack([np.interp(grid, time, accel_w[:, c]) for c in range(3)], axis=1)
    dg = np.diff(grid)[:, None]
    vel = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (a[1:] + a[:-1]) * dg, axis=0)])
    pos = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dg, axis=0)])
    at = lambda q: np.stack([np.interp(q, grid, pos[:, c]) for c in range(3)], axis=1)  # noqa: E731
    C = np.diff(at(t), axis=0)
    dt = np.diff(t)
    T = 0.5 * ((t[1:] - t[0]) ** 2 - (t[:-1] - t[0]) ** 2)
    n = len(dt)
    A = np.zeros((3 * n, 7))
    b = np.diff(P, axis=0).reshape(-1)
    for c in range(3):
        A[c::3, c] = dt
        A[c::3, 3 + c] = -T
        A[c::3, 6] = C[:, c]
    cond = float(np.linalg.cond(A / np.maximum(np.linalg.norm(A, axis=0), 1e-12)))
    if not np.isfinite(cond):
        return None
    w = np.ones(n)
    for _ in range(4):  # reweighted: a displaced keyframe must not bend the fit
        W = np.repeat(w, 3)
        sol, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        r = (b - A @ sol).reshape(n, 3)
        size = np.linalg.norm(r, axis=1)
        med, mad = _scale(size)
        w = 1.0 / np.maximum(1.0, size / max(med + 3 * mad, 1e-12))
    return r, sol, cond, float(sol[6])


def analyze_motion(
    rec: pycolmap.Reconstruction,
    images: pd.DataFrame,
    accel: dict[str, pd.DataFrame],
    R_cam_imu: dict[str, np.ndarray],
    down: np.ndarray | None,
    options: MotionOptions | None = None,
) -> dict[str, Any]:
    """Windowed accelerometer integration against the registered path. `accel` maps a sequence id to its
    `t_ns, x, y, z` samples (device frame), `R_cam_imu` to the device-to-camera rotation, `down` is the model frame's
    down direction. Per window: the fitted scale (model units per metre), the condition and the step residuals;
    `flagged` maps an image name to ["imu_motion"] when both steps next to it are residual outliers in at least half
    of the windows that cover them; `scale` holds the median and spread of the scales over the windows."""
    options = options or MotionOptions()
    report: dict[str, Any] = {"enabled": True, "windows": 0, "skipped": 0, "flagged": {}, "scale": {}}
    if down is None or not accel:
        report["note"] = "no gravity direction or no accelerometer data"
        return report
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    info = images[images["image_name"].isin(by_name)].copy()
    info["group"] = info["image_name"].map(lambda n: parse_image_name(n)[1])
    scales: list[float] = []
    flags_votes: dict[str, list[int]] = {}
    for sequence_id, seq in info.groupby("sequence_id"):
        if sequence_id not in accel or sequence_id not in R_cam_imu:
            continue
        main = seq["group"].value_counts().index[0]
        path = seq[seq["group"] == main].sort_values("t_ns")
        if len(path) < options.window:
            continue
        names = path["image_name"].tolist()
        t = path["t_ns"].to_numpy(dtype=float) * 1e-9
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        names, t = [n for n, k in zip(names, keep) if k], t[keep]
        P = np.array([by_name[n].projection_center() for n in names])
        step_med = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        table = accel[sequence_id]
        time = table["t_ns"].to_numpy(dtype=float) * 1e-9
        slerp = _slerp(by_name, names, t, R_cam_imu[sequence_id])
        inside = (time >= t[0]) & (time <= t[-1])
        time = time[inside]
        f_dev = table[["x", "y", "z"]].to_numpy(dtype=float)[inside]
        rotations = slerp(time)
        f_w = rotations.apply(f_dev)
        a_w = f_w + GRAVITY * down  # specific force plus gravity is the acceleration
        starts = range(0, len(names) - options.window + 1, options.stride)
        step_votes: dict[int, list[int]] = {}
        for s0 in starts:
            sel = slice(s0, s0 + options.window)
            tw, Pw = t[sel], P[sel]
            if np.diff(tw).max() > options.max_step_s:
                report["skipped"] += 1
                continue
            if np.linalg.norm(np.diff(Pw, axis=0), axis=1).sum() < options.min_motion * step_med:
                report["skipped"] += 1
                continue
            lo, hi = np.searchsorted(time, [tw[0] - 1e-9, tw[-1] + 1e-9])
            if hi - lo < 4 * options.window:
                report["skipped"] += 1
                continue
            fit = _window_fit(Pw, tw, time[lo:hi], a_w[lo:hi])
            if fit is None or fit[2] > options.max_condition or fit[3] <= 0:
                report["skipped"] += 1
                continue
            r, _, _, s = fit
            report["windows"] += 1
            scales.append(s)
            size = np.linalg.norm(r, axis=1)
            med, mad = _scale(size)
            window_step = float(np.median(np.linalg.norm(np.diff(Pw, axis=0), axis=1)))
            out = (size > med + options.k * mad) & (size > options.min_step_factor * window_step)
            for j, bad in enumerate(out):
                step_votes.setdefault(s0 + j, []).append(int(bad))
        for i in range(1, len(names) - 1):
            left, right = step_votes.get(i - 1, []), step_votes.get(i, [])
            if left and right and np.mean(left) >= 0.5 and np.mean(right) >= 0.5:
                flags_votes[names[i]] = ["imu_motion"]
    report["flagged"] = flags_votes
    if scales:
        med, mad = _scale(np.log(np.array(scales)))
        report["scale"] = {"median": float(np.exp(med)), "log_spread": mad, "windows": len(scales)}
    return report


def analyze_heading(
    rec: pycolmap.Reconstruction,
    images: pd.DataFrame,
    magnetic: dict[str, pd.DataFrame],
    R_cam_imu: dict[str, np.ndarray],
    options: HeadingOptions | None = None,
) -> dict[str, Any]:
    """Magnetometer against the camera orientations. In the model frame the magnetic field is a constant vector
    `m`, and the device reading carries a constant offset `o` (hard iron): `R_i (m_i - o) = m` for every keyframe `i`
    (`R_i` the device-to-model rotation), linear in `(m, o)`. A keyframe whose residual stands out in the capture's
    own distribution has a rotation the magnetometer does not support (or a local field disturbance): report only."""
    options = options or HeadingOptions()
    report: dict[str, Any] = {"enabled": True, "checked": 0, "outliers": [], "field": None}
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    rows = []
    for sequence_id, seq in images[images["image_name"].isin(by_name)].groupby("sequence_id"):
        if sequence_id not in magnetic or sequence_id not in R_cam_imu:
            continue
        table = magnetic[sequence_id]
        time = table["t_ns"].to_numpy(dtype=float)
        xyz = table[["x", "y", "z"]].to_numpy(dtype=float)
        for name, t_ns in zip(seq["image_name"], seq["t_ns"]):
            if not time[0] <= t_ns <= time[-1]:
                continue
            m = np.array([np.interp(t_ns, time, xyz[:, c]) for c in range(3)])
            R = by_name[name].cam_from_world().rotation.matrix().T @ R_cam_imu[sequence_id]
            rows.append((name, R, m))
    if len(rows) < 12:
        report["note"] = "too few keyframes with magnetometer samples"
        return report
    names = [r[0] for r in rows]
    R = np.array([r[1] for r in rows])
    m = np.array([r[2] for r in rows])
    rm = np.einsum("nij,nj->ni", R, m)
    A = np.concatenate([np.tile(np.eye(3), (len(rows), 1)), np.concatenate(list(R), axis=0)], axis=1)
    b = rm.reshape(-1)
    w = np.ones(len(rows))
    for _ in range(5):
        W = np.repeat(w, 3)
        sol, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        res = np.linalg.norm((b - A @ sol).reshape(-1, 3), axis=1)
        med, mad = _scale(res)
        w = 1.0 / np.maximum(1.0, res / max(med + 3 * mad, 1e-12))
    field = float(np.linalg.norm(sol[:3]))
    med, mad = _scale(res)
    bad = (res > med + options.k * mad) & (res > options.floor_frac * field)
    report.update(
        checked=len(rows), field=field, median_residual=med,
        outliers=[{"image": n, "residual": float(r)} for n, r, flag in zip(names, res, bad) if flag],
    )
    return report
