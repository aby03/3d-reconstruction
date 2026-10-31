"""Open Camera Sensors captures: one video plus per-frame timestamps and raw IMU CSVs.

Layout:
    <root>/VID_<stamp>.mp4
    <root>/<stamp>/VID_<stamp>_imu_timestamps.csv   one int64 ns timestamp per video frame
    <root>/<stamp>/VID_<stamp>{accel,gyro,magnetic}.csv   headerless x,y,z,t_ns
    <root>/<stamp>/<t_ns>.jpg                        sparse reference frames
"""

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recon.capture.base import BaseCaptureLoader
from recon.capture.reference import check_reference_images
from recon.capture.timing import check_frame_timing
from recon.core.frames import (
    ANDROID_DEVICE,
    R_cam_imu_upright,
    device_orientation_from_display,
    device_orientation_from_gravity,
)
from recon.core.types import CameraSensorExtrinsics, CaptureSequence, Frame, SensorKind, SensorStream, VideoStream
from recon.io.video import VideoReader, probe_video
from recon.sensors.streams import clean_stream, stream_stats

log = logging.getLogger(__name__)

VIDEO_RE = re.compile(r"VID_(?P<stamp>\d{8}_\d{6})\.mp4")
REFERENCE_RE = re.compile(r"(?P<t_ns>\d+)\.jpg")

SENSOR_UNITS = {SensorKind.ACCEL: "m/s^2", SensorKind.GYRO: "rad/s", SensorKind.MAG: "uT"}
SOURCE_TYPE = "open_camera_sensors"
MIN_REFERENCE_PSNR_DB = 25.0


@dataclass(slots=True)
class LoaderOptions:
    allow_mismatch: bool = False  # truncate video frames and timestamps to the shorter instead of failing
    verify_reference_jpegs: bool = True  # compare decoded frames with the saved JPEGs on load
    max_reference_checks: int | None = None  # None checks every reference JPEG


@dataclass(slots=True)
class OpenCameraFiles:
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

    @property
    def sequence_id(self) -> str:
        return f"{self.root.name}_{self.stamp}"


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


def discover_files(path: Path | str) -> OpenCameraFiles:
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

    files = OpenCameraFiles(root=root, stamp=stamp, video=video, frame_timestamps=frame_timestamps)
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
        kind=kind, t_ns=t_ns, values=values, units=SENSOR_UNITS[kind], frame=ANDROID_DEVICE, source_file=path
    )


class OpenCameraSensorsLoader(BaseCaptureLoader):
    name = SOURCE_TYPE

    def __init__(self, root: Path | str, options: LoaderOptions | None = None):
        self.root = Path(root)
        self.options = options or LoaderOptions()
        self.sequence: CaptureSequence | None = None

    @classmethod
    def can_load(cls, root: Path | str) -> bool:
        try:
            discover_files(root)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def load(self) -> CaptureSequence:
        """Sensors, video properties and per-frame timestamps, validated; no pixels are decoded."""
        files = discover_files(self.root)
        checks: dict = {"warnings": list(files.warnings), "sensors": {}}

        sensors = {}
        for kind, path in files.sensors.items():
            stream, counts = clean_stream(read_sensor_csv(path, kind))
            sensors[kind] = stream
            checks["sensors"][str(kind)] = counts | stream_stats(stream)

        frame_t_ns = read_frame_timestamps(files.frame_timestamps)
        probe = probe_video(files.video)
        timing = check_frame_timing(probe.pts_s, frame_t_ns, self.options.allow_mismatch)
        checks["frame_timing"] = asdict(timing) | {"ok": timing.ok}
        if not timing.ok:
            checks["warnings"].append(f"frame timing check failed: {checks['frame_timing']}")
            log.warning(checks["warnings"][-1])
        n = timing.num_frames
        frame_t_ns = frame_t_ns[:n]

        rotation_deg = (-probe.display_rotation) % 360
        w, h = probe.stored_size
        camera = VideoStream(
            camera_id="cam0",
            path=files.video,
            codec=probe.codec,
            stored_size=(w, h),
            upright_size=(h, w) if rotation_deg in (90, 270) else (w, h),
            rotation_deg=rotation_deg,
            num_frames=n,
            nominal_fps=probe.nominal_fps,
            pts_s=probe.pts_s[:n],
            frame_t_ns=frame_t_ns,
        )

        frame_index = {int(t): i for i, t in enumerate(frame_t_ns)}
        reference_images = {}
        for t_ns, jpg in files.reference_images.items():
            if t_ns in frame_index:
                reference_images[frame_index[t_ns]] = jpg
            else:
                checks["warnings"].append(f"reference image {jpg.name} matches no frame timestamp")
                log.warning(checks["warnings"][-1])

        orientation = self._device_orientation(rotation_deg, sensors.get(SensorKind.ACCEL), checks)
        extrinsics = CameraSensorExtrinsics(
            camera_id=camera.camera_id,
            R_cam_imu=R_cam_imu_upright(orientation),
            t_cam_imu=np.zeros(3),
            time_offset_ns=0,
            source="nominal",
        )

        self.sequence = CaptureSequence(
            sequence_id=files.sequence_id,
            source_type=SOURCE_TYPE,
            root=files.root,
            cameras={camera.camera_id: camera},
            sensors=sensors,
            reference_images=reference_images,
            checks=checks,
            extrinsics={camera.camera_id: extrinsics},
        )
        if self.options.verify_reference_jpegs and reference_images:
            self._verify_reference_images()
        return self.sequence

    @staticmethod
    def _device_orientation(rotation_deg: int, accel: SensorStream | None, checks: dict) -> int:
        """Orientation from the display rotation, cross-checked against the mean accelerometer direction."""
        from_display = device_orientation_from_display(rotation_deg)
        from_gravity = None
        if accel is not None and len(accel.t_ns):
            from_gravity = device_orientation_from_gravity(-accel.values.mean(axis=0))
        checks["device_orientation"] = {"deg": from_display, "from_display": from_display, "from_gravity": from_gravity}
        if from_gravity is not None and from_gravity != from_display:
            checks["warnings"].append(
                f"device orientation from gravity ({from_gravity}) differs from the display rotation ({from_display})"
            )
            log.warning(checks["warnings"][-1])
        return from_display

    def _verify_reference_images(self) -> None:
        checks = self.sequence.checks
        result = check_reference_images(self, self.sequence, max_checks=self.options.max_reference_checks)
        checks["reference_images"] = result
        if result["rotation_mismatch"]:
            checks["rotation_mismatch"] = True
            checks["warnings"].append(
                f"reference JPEGs fit rotation {result['best_rotation']} best, "
                f"not the chosen {self.sequence.cameras['cam0'].rotation_deg}: {result['rotation_psnr_db']}"
            )
            log.warning(checks["warnings"][-1])
        if result["min_psnr_db"] < MIN_REFERENCE_PSNR_DB:
            checks["warnings"].append(f"reference JPEG min PSNR {result['min_psnr_db']:.1f} dB < {MIN_REFERENCE_PSNR_DB}")
            log.warning(checks["warnings"][-1])

    def iter_frames(
        self, camera_id: str = "cam0", indices: Iterable[int] | None = None, scale: float = 1.0
    ) -> Iterator[Frame]:
        if self.sequence is None:
            self.load()
        camera = self.sequence.cameras[camera_id]
        n = camera.num_frames
        if indices is None:
            indices = range(n)
        else:
            indices = sorted(set(indices))
            if indices and (indices[0] < 0 or indices[-1] >= n):
                raise IndexError(f"frame indices must be in [0, {n}), got {indices[0]}..{indices[-1]}")
        reader = VideoReader(camera.path, camera.rotation_deg)
        for idx, image in reader.iter_frames(indices, scale):
            yield Frame(camera_id=camera_id, idx=idx, t_ns=int(camera.frame_t_ns[idx]), image=image)
