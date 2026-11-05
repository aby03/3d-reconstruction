"""Pose continuity of a registered model against capture time: cameras that were taken close in time must be close
in position and rotation. Cameras of one capture are grouped by camera folder; the largest group of a sequence
(the video) gives the path, the others (e.g. stills) are checked against the path camera nearest in time. Every
threshold is a robust statistic of the capture itself (median and scaled MAD), never an absolute number."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Rotation

from recon.io.colmap import parse_image_name
from recon.sfm.priors import GyroRotations

MAD_TO_STD = 1.4826


@dataclass(slots=True)
class ContinuityOptions:
    enabled: bool = True
    k: float = 4.0  # a value is an outlier above the median by this many scaled MADs ...
    min_step_factor: float = 2.0  # ... and a step or offset also needs to exceed this many median steps
    max_still_dt_s: float = 1.0  # a still is compared with the path camera nearest in time when it is this close
    rotation_floor_deg: float = 3.0  # a gyro disagreement also needs to exceed this
    gyro_max_dt_s: float = 1.0  # rotation steps are checked against the gyro up to this time gap


def _scale(x: np.ndarray) -> tuple[float, float]:
    """Median and scaled MAD (never zero)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * MAD_TO_STD
    return med, max(mad, 1e-9 * max(abs(med), 1.0))


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def analyze(
    rec: pycolmap.Reconstruction,
    images: pd.DataFrame,
    gyro: dict[str, GyroRotations] | None = None,
    options: ContinuityOptions | None = None,
) -> dict[str, Any]:
    """Continuity of the registered images of `rec`. `images` has `image_name, sequence_id, idx, t_ns` per keyframe
    (and still); `gyro` the integrated gyro rotations per sequence. Per sequence: the largest camera group is the
    path; its steps (distance over time gap) give `jumps` (a step far above the median speed) and `isolated`
    cameras (far from the line between their neighbours); other groups give `still_offsets`; `rotation` compares
    time-adjacent rotations with the gyro. `flagged` maps an image name to the signals that flag it."""
    options = options or ContinuityOptions()
    gyro = gyro or {}
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    info = images[images["image_name"].isin(by_name)].copy()
    info["group"] = info["image_name"].map(lambda n: parse_image_name(n)[1])
    flagged: dict[str, list[str]] = {}
    report: dict[str, Any] = {"enabled": True, "sequences": {}, "flagged": flagged}

    def flag(name: str, signal: str) -> None:
        flagged.setdefault(name, [])
        if signal not in flagged[name]:
            flagged[name].append(signal)

    for sequence_id, seq in info.groupby("sequence_id"):
        counts = seq["group"].value_counts()
        main = str(counts.index[0])
        path = seq[seq["group"] == main].sort_values("t_ns")
        out: dict[str, Any] = {"path_camera": main, "path_images": len(path)}
        report["sequences"][sequence_id] = out
        if len(path) < 4:
            continue
        names = path["image_name"].tolist()
        t = path["t_ns"].to_numpy(dtype=float) * 1e-9
        P = np.array([by_name[n].projection_center() for n in names])
        d, dt = np.linalg.norm(np.diff(P, axis=0), axis=1), np.diff(t)
        step_med = float(np.median(d))
        ok = dt > 1e-6
        log_speed = np.log(np.maximum(d[ok], 1e-12) / dt[ok])
        med_ls, mad_ls = _scale(log_speed)
        z = np.full(len(d), np.nan)
        z[ok] = (np.log(np.maximum(d[ok], 1e-12) / dt[ok]) - med_ls) / mad_ls
        jump = (z > options.k) & (d > options.min_step_factor * step_med)
        out["median_step"], out["median_speed"] = step_med, float(np.exp(med_ls))
        out["jumps"] = [
            {"from": names[i], "to": names[i + 1], "dist": float(d[i]), "dt_s": float(dt[i]), "z": float(z[i])}
            for i in np.flatnonzero(jump)
        ]
        # a camera off the line between its neighbours (an isolated outlier makes two long steps)
        resid = np.zeros(len(P))
        w = (t[1:-1] - t[:-2]) / np.maximum(t[2:] - t[:-2], 1e-9)
        resid[1:-1] = np.linalg.norm(P[1:-1] - (P[:-2] + w[:, None] * (P[2:] - P[:-2])), axis=1)
        med_r, mad_r = _scale(resid[1:-1])
        bad = (resid > med_r + options.k * mad_r) & (resid > options.min_step_factor * step_med)
        out["isolated"] = [
            {"image": names[i], "offset": float(resid[i])} for i in np.flatnonzero(bad)
        ]
        for i in np.flatnonzero(jump):  # the end of a sequence has no neighbour on one side
            if i == 0:
                flag(names[0], "jump")
            if i == len(d) - 1:
                flag(names[-1], "jump")
        for entry in out["isolated"]:
            flag(entry["image"], "isolated")
        for i in np.flatnonzero(jump):
            if i > 0 and jump[i - 1]:
                flag(names[i], "jump")

        # cameras of the other groups against the path camera nearest in time
        others = seq[seq["group"] != main]
        offsets = []
        if len(others):
            tt = others["t_ns"].to_numpy(dtype=float) * 1e-9
            nearest = np.abs(t[None, :] - tt[:, None]).argmin(axis=1)
            for name, j, tn in zip(others["image_name"], nearest, tt):
                if abs(t[j] - tn) <= options.max_still_dt_s:
                    offsets.append((name, float(np.linalg.norm(by_name[name].projection_center() - P[j])), float(abs(t[j] - tn))))
        out["still_offsets"] = {"checked": len(offsets), "outliers": []}
        if len(offsets) >= 4:
            dist = np.array([o[1] for o in offsets])
            med_o, mad_o = _scale(dist)
            out["still_offsets"].update(median=med_o)
            for name, dd, dtn in offsets:
                if dd > med_o + options.k * mad_o and dd > options.min_step_factor * step_med:
                    out["still_offsets"]["outliers"].append({"image": name, "dist": dd, "dt_s": dtn})
                    flag(name, "still_offset")

        # rotation steps against the gyro
        g = gyro.get(sequence_id)
        out["rotation"] = {"checked": 0, "outliers": []}
        if g is not None:
            idx = path["idx"].to_numpy(dtype=int)
            R = [by_name[n].cam_from_world().rotation.matrix() for n in names]
            errs, pairs = [], []
            for i in range(len(names) - 1):
                if dt[i] > options.gyro_max_dt_s:
                    continue
                R_gyro = g.cam_rotation(int(idx[i]), int(idx[i + 1]))
                if R_gyro is None:
                    continue
                errs.append(_angle_deg((R[i + 1] @ R[i].T) @ R_gyro.T))
                pairs.append(i)
            out["rotation"]["checked"] = len(errs)
            if len(errs) >= 8:
                e = np.array(errs)
                med_e, mad_e = _scale(e)
                out["rotation"].update(median_err_deg=med_e, p90_err_deg=float(np.percentile(e, 90)))
                for k, i in enumerate(pairs):
                    if e[k] > med_e + options.k * mad_e and e[k] > options.rotation_floor_deg:
                        out["rotation"]["outliers"].append(
                            {"from": names[i], "to": names[i + 1], "err_deg": float(e[k]), "dt_s": float(dt[i])}
                        )
                        flag(names[i], "rotation")
                        flag(names[i + 1], "rotation")
    report["num_flagged"] = len(flagged)
    report["num_flagged_strong"] = sum(len(s) >= 2 for s in flagged.values())
    return report


def summary(report: dict[str, Any]) -> str:
    """One line per sequence for the log."""
    lines = []
    for sequence_id, s in report["sequences"].items():
        lines.append(
            f"{sequence_id}: path {s['path_camera']} ({s['path_images']} images), {len(s.get('jumps', []))} jump(s),"
            f" {len(s.get('isolated', []))} isolated, {len(s.get('still_offsets', {}).get('outliers', []))} still offset(s),"
            f" {len(s.get('rotation', {}).get('outliers', []))} rotation outlier(s)"
        )
    return "; ".join(lines)
