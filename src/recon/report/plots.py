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


def _waypoint_labels(path_xy: np.ndarray, picks: np.ndarray, extent: float) -> np.ndarray:
    """Label positions for the path points `picks` (indices into `path_xy`): each label sits beside the path on the
    side with more free room, then labels that are too close to each other are pushed apart."""
    gap = 0.035 * extent
    positions = []
    for i in picks:
        lo, hi = max(i - 3, 0), min(i + 3, len(path_xy) - 1)
        tangent = path_xy[hi] - path_xy[lo]
        norm = np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) / norm if norm > 0 else np.array([0.0, 1.0])
        others = np.delete(path_xy, np.arange(max(i - 15, 0), min(i + 16, len(path_xy))), axis=0)
        room = [np.min(np.linalg.norm(others - (path_xy[i] + side * normal * gap), axis=1)) if len(others) else 1.0 for side in (1, -1)]
        positions.append(path_xy[i] + (1 if room[0] >= room[1] else -1) * normal * gap)
    positions = np.array(positions)
    for _ in range(200):  # push labels apart until none is closer than 1.6 gaps
        moved = False
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                d = positions[b] - positions[a]
                dist = np.linalg.norm(d)
                if dist < 1.6 * gap:
                    push = (d / dist if dist > 1e-9 else np.array([1.0, 0.0])) * (1.6 * gap - dist) / 2
                    positions[a] -= push
                    positions[b] += push
                    moved = True
        if not moved:
            break
    return positions


def plot_path_topdown(
    centres: np.ndarray,
    forward: np.ndarray,
    groups: Sequence[str],
    points: np.ndarray,
    path: Path,
    max_points: int = 40_000,
    max_arrows: int = 60,
    seed: int = 0,
    times: Sequence[float] | None = None,
    waypoints: int = 12,
) -> Path:
    """The walk seen from above: sparse points in grey, the camera path of the largest group (given in time order)
    coloured from first to last with arrowheads along the walking direction, numbered waypoints at equal time steps
    (the table on the right gives the elapsed seconds of each number; `times` are the camera times in seconds, the
    camera order when None), thin red arrows for the viewing direction (every n-th camera, at most `max_arrows`) and the
    other groups (e.g. stills) as small markers. Labels are kept apart from each other."""
    centres, forward = np.asarray(centres, dtype=float), np.asarray(forward, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    if len(points):
        lo, hi = np.percentile(points, [1, 99], axis=0)
        points = points[((points >= lo) & (points <= hi)).all(axis=1)]
    groups = np.asarray(groups)
    labels = sorted(set(groups.tolist()))
    main = max(labels, key=groups.tolist().count)
    idx = np.flatnonzero(groups == main)
    seconds = np.asarray(times, dtype=float)[idx] if times is not None else np.arange(len(idx), dtype=float)
    seconds = seconds - seconds[0]
    path_xy = centres[idx, :2]
    everything = np.concatenate([centres[:, :2], points[:, :2]]) if len(points) else centres[:, :2]
    extent = float(np.max(everything.max(axis=0) - everything.min(axis=0))) or 1.0
    fig = Figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.03)
    ax, table = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=0.3, color="0.8", alpha=0.5, rasterized=True)
    pick = idx[:: max(1, len(idx) // max_arrows)]
    ax.quiver(
        centres[pick, 0], centres[pick, 1], forward[pick, 0], forward[pick, 1], color="tab:red", alpha=0.45,
        angles="xy", scale_units="xy", scale=1 / (0.035 * extent), width=0.0015, zorder=2,
    )
    ax.plot(path_xy[:, 0], path_xy[:, 1], lw=0.6, color="0.35", zorder=3)
    sc = ax.scatter(path_xy[:, 0], path_xy[:, 1], c=seconds, cmap="viridis", s=8, zorder=4)
    for group in labels:
        if group != main:
            other = np.flatnonzero(groups == group)
            ax.scatter(centres[other, 0], centres[other, 1], s=9, marker="s", facecolors="none", edgecolors="tab:orange", linewidths=0.6, alpha=0.8, label=f"{group} (stills, {len(other)})", zorder=3)
    count = min(waypoints, len(idx))
    targets = np.linspace(0, seconds[-1], count)
    picks = np.unique([int(np.abs(seconds - t).argmin()) for t in targets])
    for k in range(len(picks) - 1):  # an arrowhead between consecutive waypoints, on the path
        mid = (picks[k] + picks[k + 1]) // 2
        a, b = path_xy[max(mid - 1, 0)], path_xy[min(mid + 1, len(path_xy) - 1)]
        if np.linalg.norm(b - a) > 0:
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="black", lw=0, mutation_scale=14), zorder=5)
    spots = _waypoint_labels(path_xy, picks, extent)
    rows = []
    for number, (i, spot) in enumerate(zip(picks, spots), start=1):
        ax.plot([path_xy[i, 0], spot[0]], [path_xy[i, 1], spot[1]], lw=0.7, color="black", zorder=5)
        ax.scatter(*path_xy[i], s=14, color="black", zorder=6)
        ax.text(spot[0], spot[1], str(number), ha="center", va="center", fontsize=9, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.9))
        rows.append((number, seconds[i], "start" if number == 1 else "end" if number == len(picks) else ""))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"camera path from above ({main}, {len(idx)} cameras)")
    handles, names = ax.get_legend_handles_labels()
    table.axis("off")
    table.text(0, 1.0, "waypoint   time", fontsize=10, fontweight="bold", va="top", transform=table.transAxes)
    for r, (number, sec, note) in enumerate(rows):
        table.text(0, 0.96 - r * 0.045, f"{number:>2}", fontsize=10, va="top", transform=table.transAxes,
                   bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.8))
        table.text(0.2, 0.96 - r * 0.045, f"{sec:5.1f} s  {note}", fontsize=10, va="top", transform=table.transAxes)
    table.text(0, 0.96 - len(rows) * 0.045 - 0.04, "colour: time, first to last\nblack arrows: walking direction\nred arrows: viewing direction", fontsize=8.5, va="top", color="0.25", transform=table.transAxes)
    if handles:
        table.legend(handles, names, loc="upper left", bbox_to_anchor=(0, 0.96 - len(rows) * 0.045 - 0.13), fontsize=8.5, frameon=False)
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01, label="elapsed time (s)")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.07)
    fig.savefig(path, dpi=130)
    return path


def plot_cameras_3d(
    centres: np.ndarray,
    forward: np.ndarray,
    groups: Sequence[str],
    points: np.ndarray,
    path: Path,
    max_points: int = 40_000,
    max_arrows: int = 80,
    seed: int = 0,
) -> Path:
    """The model in 3D, Z up: sparse points coloured by height, the camera path (the largest group, in order) and an arrow per camera
    along its viewing direction (every n-th camera, at most `max_arrows`), cameras coloured by `groups` (e.g. the
    camera model: video frames or stills). Left an oblique view, right the view from above."""
    centres, forward = np.asarray(centres, dtype=float), np.asarray(forward, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    if len(points):
        lo, hi = np.percentile(points, [1, 99], axis=0)
        points = points[((points >= lo) & (points <= hi)).all(axis=1)]
    everything = np.concatenate([centres, points]) if len(points) else centres
    low, high = everything.min(axis=0), everything.max(axis=0)
    extent = float(np.max(high - low)) or 1.0
    step = max(1, len(centres) // max_arrows)
    arrow = 0.06 * extent
    labels = sorted(set(groups))
    colors = {g: f"C{i}" for i, g in enumerate(labels)}
    fig = Figure(figsize=(14, 6.5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    if len(points):
        ax3.scatter(*points.T, c=points[:, 2], cmap="Greys", s=0.3, alpha=0.5)
        ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="Greys", s=0.3, alpha=0.4)
    path_group = max(labels, key=list(groups).count)  # the path joins the largest group (the video), not the stills
    track = centres[np.asarray(groups) == path_group]
    ax3.plot(*track.T, lw=0.6, color="black")
    ax2.plot(track[:, 0], track[:, 1], lw=0.6, color="black")
    for group in labels:
        idx = np.flatnonzero(np.asarray(groups) == group)
        pick = idx[::step]
        ax3.scatter(*centres[idx].T, s=6, color=colors[group], label=f"{group} ({len(idx)})")
        ax2.scatter(centres[idx, 0], centres[idx, 1], s=6, color=colors[group], label=f"{group} ({len(idx)})")
        ax3.quiver(*centres[pick].T, *(forward[pick] * arrow).T, color=colors[group], linewidth=0.8, arrow_length_ratio=0.3)
        ax2.quiver(
            centres[pick, 0], centres[pick, 1], forward[pick, 0], forward[pick, 1], color=colors[group],
            angles="xy", scale_units="xy", scale=1 / arrow, width=0.002,
        )
    ax3.scatter(*centres[0], s=60, facecolors="none", edgecolors="tab:green", label="first")
    ax2.plot(*centres[0, :2], "o", ms=9, mfc="none", color="tab:green")
    ax3.set_box_aspect(np.maximum(high - low, 1e-9))
    ax3.view_init(elev=28, azim=-60)
    for ax, title in ((ax3, "oblique view (arrows: viewing direction)"), (ax2, "from above")):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax3.set_zlabel("z (up)")
    ax2.set_aspect("equal")
    ax3.legend(loc="upper left")
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


def plot_mono_fit(median_residuals: np.ndarray, inlier_ratios: np.ndarray, path: Path) -> Path:
    """Per-view median relative residual and inlier ratio of the mono depth fit to the SfM points."""
    fig = Figure(figsize=(10, 3.5))
    ax1, ax2 = fig.subplots(1, 2)
    for ax, values, label in ((ax1, median_residuals, "median relative residual"), (ax2, inlier_ratios, "inlier ratio")):
        ax.hist(values, bins=20)
        if len(values):
            ax.axvline(np.median(values), color="tab:red", lw=0.8, ls="--", label=f"median {np.median(values):.3g}")
            ax.legend(loc="upper right")
        ax.set_xlabel(label)
        ax.set_ylabel("views")
    return _save(fig, path)


def plot_mono_example(image_bgr: np.ndarray, depth: np.ndarray, confidence: np.ndarray, path: Path) -> Path:
    """One keyframe next to its fitted mono depth and the confidence mask."""
    fig = Figure(figsize=(12, 5))
    axes = fig.subplots(1, 3)
    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    valid = depth > 0
    axes[1].imshow(np.where(valid, depth, np.nan), cmap="turbo", vmax=np.percentile(depth[valid], 98) if valid.any() else None)
    axes[2].imshow(confidence, cmap="gray", vmin=0, vmax=1)
    for ax, title in zip(axes, ("keyframe", "fitted depth", "confidence")):
        ax.set_title(title)
        ax.axis("off")
    return _save(fig, path)
