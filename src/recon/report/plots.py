"""Diagnostic plots, rendered off-screen with matplotlib's Agg canvas. Times are seconds from the first frame."""

from collections.abc import Mapping, Sequence
from pathlib import Path

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
