"""Diagnostic plots, rendered off-screen with matplotlib's Agg canvas. Times are seconds from the first frame."""

from collections.abc import Mapping, Sequence
from pathlib import Path

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
            ax.axhline(thresholds[name], color="tab:gray", lw=0.8, ls="--", label="reject")
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
