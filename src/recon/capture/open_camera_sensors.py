"""Open Camera Sensors captures: one video plus per-frame timestamps and raw IMU CSVs.

Layout:
    <root>/VID_<stamp>.mp4
    <root>/<stamp>/VID_<stamp>_imu_timestamps.csv   one int64 ns timestamp per video frame
    <root>/<stamp>/VID_<stamp>{accel,gyro,magnetic}.csv   headerless x,y,z,t_ns
    <root>/<stamp>/<t_ns>.jpg                        sparse reference frames
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recon.core.types import SensorKind, SensorStream

log = logging.getLogger(__name__)

VIDEO_RE = re.compile(r"VID_(?P<stamp>\d{8}_\d{6})\.mp4")
REFERENCE_RE = re.compile(r"(?P<t_ns>\d+)\.jpg")

DEVICE_FRAME = "android_device"
SENSOR_UNITS = {SensorKind.ACCEL: "m/s^2", SensorKind.GYRO: "rad/s", SensorKind.MAG: "uT"}


@dataclass(slots=True)
class CaptureFiles:
    root: Path
    stamp: str
    video: Path
    frame_timestamps: Path
    sensors: dict[SensorKind, Path] = field(default_factory=dict)
    reference_images: dict[int, Path] = field(default_factory=dict)  # t_ns -> jpg
    warnings: list[str] = field(default_factory=list)

    @property
    def stamp_dir(self) -> Path:
        return self.root / self.stamp


def _videos_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("VID_*.mp4") if VIDEO_RE.fullmatch(p.name))


def _resolve_root(path: Path) -> tuple[Path, Path]:
    """Return (capture root, video) for a path that is either the root or its stamp dir."""
    for root in (path, path.parent):
        videos = _videos_in(root)
        if len(videos) > 1:
            raise ValueError(f"expected one VID_<stamp>.mp4 in {root}, found {[v.name for v in videos]}")
        if videos and (root == path or VIDEO_RE.fullmatch(videos[0].name)["stamp"] == path.name):
            return root, videos[0]
    raise FileNotFoundError(f"no Open Camera Sensors capture at {path}")


def discover_files(path: Path | str) -> CaptureFiles:
    """Find the files of a capture, given its root dir or its stamp dir."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")
    root, video = _resolve_root(path)
    stamp = VIDEO_RE.fullmatch(video.name)["stamp"]
    stamp_dir = root / stamp

    frame_timestamps = stamp_dir / f"VID_{stamp}_imu_timestamps.csv"
    if not frame_timestamps.is_file():
        raise FileNotFoundError(f"missing frame timestamps: {frame_timestamps}")

    files = CaptureFiles(root=root, stamp=stamp, video=video, frame_timestamps=frame_timestamps)
    for kind in SensorKind:
        sensor_path = stamp_dir / f"VID_{stamp}{kind}.csv"
        if sensor_path.is_file():
            files.sensors[kind] = sensor_path
        else:
            files.warnings.append(f"missing {kind} file: {sensor_path.name}")
            log.warning(files.warnings[-1])

    for jpg in sorted(stamp_dir.glob("*.jpg")):
        if m := REFERENCE_RE.fullmatch(jpg.name):
            files.reference_images[int(m["t_ns"])] = jpg
    return files


def read_frame_timestamps(path: Path | str) -> np.ndarray:
    """One int64 ns timestamp per video frame, in decode order."""
    t_ns = pd.read_csv(path, header=None, dtype=np.int64).to_numpy()
    if t_ns.ndim != 2 or t_ns.shape[1] != 1:
        raise ValueError(f"{path}: expected one timestamp per line")
    return t_ns[:, 0]


def read_sensor_csv(path: Path | str, kind: SensorKind) -> SensorStream:
    """Parse a headerless `x,y,z,t_ns` CSV. Timestamps are parsed straight to int64, never via float."""
    path = Path(path)
    df = pd.read_csv(path, header=None, dtype=str)
    if df.shape[1] != 4:
        raise ValueError(f"{path}: expected 4 columns (x,y,z,t_ns), got {df.shape[1]}")
    if df.isna().any().any():
        raise ValueError(f"{path}: missing fields")

    values = df.iloc[:, :3].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite sensor values")
    t_ns = df.iloc[:, 3].astype(np.int64).to_numpy()
    return SensorStream(
        kind=kind, t_ns=t_ns, values=values, units=SENSOR_UNITS[kind], frame=DEVICE_FRAME, source_file=path
    )
