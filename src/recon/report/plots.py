"""Diagnostic plots, rendered off-screen with matplotlib's Agg canvas. Times are seconds from the first frame."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from matplotlib.figure import Figure

from recon.core.types import SensorStream

AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    return path


def plot_sensor_timeline(
    sensors: Mapping[str, SensorStream], t0_ns: int, frame_span_s: tuple[float, float], path: Path
) -> Path:
    """Magnitude of each sensor over time, with the span covered by video frames shaded."""
    fig = Figure(figsize=(10, 2.2 * max(len(sensors), 1)))
    axes = fig.subplots(max(len(sensors), 1), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (name, stream) in zip(axes, sensors.items()):
        ax.plot((stream.t_ns - t0_ns) * 1e-9, np.linalg.norm(stream.values, axis=1), lw=0.5)
        ax.axvspan(*frame_span_s, color="tab:orange", alpha=0.15, label="frames")
        ax.set_ylabel(f"|{name}| [{stream.units}]")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    return _save(fig, path)


def plot_sample_rates(sensors: Mapping[str, SensorStream], path: Path) -> Path:
    """Histogram of the instantaneous sample rate of each sensor."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    for name, stream in sensors.items():
        dt_s = np.diff(stream.t_ns) * 1e-9
        rate = 1.0 / dt_s[dt_s > 0]
        ax.hist(rate, bins=200, histtype="step", label=f"{name} (median {np.median(rate):.1f} Hz)")
    ax.set_yscale("log")
    ax.set_xlabel("sample rate [Hz]")
    ax.set_ylabel("samples")
    ax.legend()
    return _save(fig, path)


def plot_vector_components(
    t_s: np.ndarray, vectors: np.ndarray, labels: Sequence[str], path: Path, ylabel: str = ""
) -> Path:
    """One line per column of `vectors` (N, 3) over time."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    for d, (label, color) in enumerate(zip(labels, AXIS_COLORS)):
        ax.plot(t_s, vectors[:, d], lw=0.8, color=color, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_series_with_events(
    t_s: np.ndarray, values: np.ndarray, event_t_s: np.ndarray, path: Path, ylabel: str = "", event_label: str = ""
) -> Path:
    """A time series with vertical markers at event times."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(t_s, values, lw=0.8)
    for k, t in enumerate(event_t_s):
        ax.axvline(t, color="tab:red", lw=0.8, ls="--", label=event_label if k == 0 else None)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    if len(event_t_s) and event_label:
        ax.legend(loc="upper right")
    return _save(fig, path)


def plot_scores_timeline(
    t_s: np.ndarray,
    series: Mapping[str, np.ndarray],
    keyframe_t_s: np.ndarray,
    path: Path,
    thresholds: Mapping[str, float] | None = None,
) -> Path:
    """One panel per score over time, keyframes marked as dots on each line, optional threshold lines."""
    thresholds = thresholds or {}
    fig = Figure(figsize=(12, 2.2 * len(series)))
    axes = fig.subplots(len(series), 1, sharex=True, squeeze=False)[:, 0]
    keyframe_t_s = np.asarray(keyframe_t_s)
    for ax, (name, values) in zip(axes, series.items()):
        ax.plot(t_s, values, lw=0.6)
        on_key = np.searchsorted(t_s, keyframe_t_s).clip(0, len(t_s) - 1)
        ax.plot(keyframe_t_s, np.asarray(values)[on_key], "o", ms=3, color="tab:red", label="keyframes")
        if name in thresholds:
            ax.axhline(thresholds[name], color="tab:gray", lw=0.8, ls="--", label="threshold")
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    return _save(fig, path)


def contact_sheet(
    images: Sequence[tuple[str, Path]], path: Path, max_images: int = 48, columns: int = 8, thumb_width: int = 200
) -> Path:
    """A JPEG grid of up to `max_images` labelled thumbnails, evenly sampled from `images`."""
    if len(images) > max_images:
        images = [images[k] for k in np.linspace(0, len(images) - 1, max_images).round().astype(int)]
    thumbs = []
    for label, image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            raise OSError(f"cannot read {image_path}")
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (thumb_width, round(h * thumb_width / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    th = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in thumbs]
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % columns)
    rows = [np.hstack(thumbs[r : r + columns]) for r in range(0, len(thumbs), columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"failed to write {path}")
    return path


def plot_histogram(
    values: np.ndarray, path: Path, title: str = "", xlabel: str = "", bins: int | Sequence = 50
) -> Path:
    """Histogram of `values` with its median marked."""
    values = np.asarray(values)
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(values, bins=bins)
    if len(values):
        ax.axvline(np.median(values), color="tab:red", lw=0.8, ls="--", label=f"median {np.median(values):.3g}")
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    return _save(fig, path)


def plot_trajectory_topdown(
    centres: np.ndarray,
    points: np.ndarray,
    path: Path,
    max_points: int = 50_000,
    seed: int = 0,
    axes: Literal["pca", "xy"] = "pca",
) -> Path:
    """Camera centres (in order, coloured by index) over the points, seen from above: projected onto the two main axes
    of the centres (`pca`, any model frame) or onto the x/y plane (`xy`, a Z-up world frame)."""
    centres, points = np.asarray(centres, dtype=float), np.asarray(points, dtype=float).reshape(-1, 3)
    if axes == "xy":
        mean, basis = np.zeros(3), np.eye(3)
    else:
        mean = centres.mean(axis=0)
        _, _, basis = np.linalg.svd(centres - mean, full_matrices=False)

    def to_2d(xyz: np.ndarray) -> np.ndarray:
        return (xyz - mean) @ basis[:2].T

    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    fig = Figure(figsize=(8, 8))
    ax = fig.subplots()
    if len(points):
        p2 = to_2d(points)
        lo, hi = np.percentile(p2, [1, 99], axis=0)
        p2 = p2[((p2 >= lo) & (p2 <= hi)).all(axis=1)]
        ax.scatter(p2[:, 0], p2[:, 1], s=0.2, color="tab:gray", alpha=0.4, label="points")
    c2 = to_2d(centres)
    ax.plot(c2[:, 0], c2[:, 1], lw=0.6, color="black")
    sc = ax.scatter(c2[:, 0], c2[:, 1], s=8, c=np.arange(len(c2)), cmap="viridis", label="cameras", zorder=3)
    ax.plot(*c2[0], "o", ms=8, mfc="none", color="tab:green", label="first")
    fig.colorbar(sc, ax=ax, label="keyframe")
    ax.set_aspect("equal")
    ax.set_xlabel("x" if axes == "xy" else "main axis 1")
    ax.set_ylabel("y" if axes == "xy" else "main axis 2")
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_height_histogram(
    z: np.ndarray, floor_z: float | None, camera_z: np.ndarray, path: Path, bins: int = 200
) -> Path:
    """Histogram of point heights (1st–99th percentile), the camera heights as a rug and the floor height."""
    z, camera_z = np.asarray(z, dtype=float), np.asarray(camera_z, dtype=float)
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    if len(z):
        lo, hi = np.percentile(np.concatenate([z, camera_z]), [1, 99])
        ax.hist(z[(z >= lo) & (z <= hi)], bins=bins, color="tab:gray", label="points")
    ax.plot(camera_z, np.zeros_like(camera_z), "|", ms=12, color="tab:blue", label="cameras")
    if floor_z is not None:
        ax.axvline(floor_z, color="tab:red", lw=1.0, ls="--", label=f"floor z = {floor_z:.3g}")
    ax.set_title("point heights")
    ax.set_xlabel("z (up)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_yaw_histogram(histogram: np.ndarray, yaw_deg: float | None, applied: bool, path: Path) -> Path:
    """Wall-normal directions mod 90° (1° bins) with the chosen yaw marked."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.bar(np.arange(len(histogram)) + 0.5, histogram, width=1.0, color="tab:gray")
    if yaw_deg is not None:
        label = f"yaw {yaw_deg:.1f}° ({'applied' if applied else 'not applied'})"
        ax.axvline(yaw_deg % 90.0, color="tab:red" if applied else "tab:orange", lw=1.0, ls="--", label=label)
        ax.legend(loc="upper right")
    ax.set_title("wall normal directions (mod 90°)")
    ax.set_xlabel("atan2(n_y, n_x) mod 90 (degrees)")
    ax.set_ylabel("count")
    ax.set_xlim(0, 90)
    return _save(fig, path)


def plot_timings(durations: Mapping[str, float], path: Path) -> Path:
    """Horizontal bars of the stage durations (seconds), in pipeline order from the top."""
    labels, values = list(durations), list(durations.values())
    fig = Figure(figsize=(7, 0.4 * len(labels) + 1.2))
    ax = fig.subplots()
    ax.barh(labels[::-1], values[::-1], color="tab:blue")
    for y, value in enumerate(values[::-1]):
        ax.text(value, y, f" {value:.1f} s", va="center", fontsize=8)
    ax.set_xlabel("duration (s)")
    ax.set_title(f"stage durations (total {sum(values):.0f} s)")
    ax.set_xlim(0, max(values, default=1.0) * 1.2)
    return _save(fig, path)
