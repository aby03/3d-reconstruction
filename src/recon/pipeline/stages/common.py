"""Helpers shared by stage adapters for reading upstream artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from recon.capture.base import BaseCaptureLoader
from recon.capture.open_camera_sensors import SENSOR_UNITS, SOURCE_TYPE, LoaderOptions, OpenCameraSensorsLoader
from recon.core.frames import ANDROID_DEVICE
from recon.core.types import SensorKind, SensorStream
from recon.io.tables import read_csv, read_json
from recon.pipeline.run_dir import read_manifest, stage_dir
from recon.pipeline.stages import s01_ingest, s02_keyframes
from recon.sfm.priors import GyroRotations


def open_loader(
    sequence_dir: Path, decode_backend: Literal["nvdec", "pyav"] = "pyav", gpu_index: int = 0
) -> BaseCaptureLoader:
    """The loader of the capture that `01_ingest/sequence.json` came from, for decoding its frames again."""
    info = read_json(stage_dir(sequence_dir, s01_ingest.DIR_NAME) / s01_ingest.ARTIFACTS["sequence"])
    if info["source_type"] != SOURCE_TYPE:
        raise ValueError(f"unsupported source type {info['source_type']!r}")
    options = LoaderOptions(verify_reference_jpegs=False, decode_backend=decode_backend)
    return OpenCameraSensorsLoader(info["root"], options, gpu_index)


def read_sensor_stream(sequence_dir: Path, kind: SensorKind) -> SensorStream:
    """A normalized sensor stream from `01_ingest/sensors/<kind>.csv`."""
    path = stage_dir(sequence_dir, s01_ingest.DIR_NAME) / s01_ingest.ARTIFACTS[f"sensors/{kind}"]
    df = read_csv(path, s01_ingest.SENSOR_COLUMNS)
    return SensorStream(
        kind=kind,
        t_ns=df["t_ns"].to_numpy(),
        values=df[["x", "y", "z"]].to_numpy(),
        units=SENSOR_UNITS[kind],
        frame=ANDROID_DEVICE,
        source_file=path,
    )


@dataclass(slots=True)
class KeyframeInputs:
    """What the scene stages read from every sequence: keyframes, their gravity priors and the gyro rotations."""

    images: pd.DataFrame  # image_name, sequence_id, idx, t_ns; sorted by name
    image_dirs: dict[str, Path]  # sequence_id -> keyframe image dir
    gravity: pd.DataFrame  # gravity_cam.csv rows of every sequence
    gyro: dict[str, GyroRotations]  # sequences that have gyro rotations

    @property
    def names(self) -> list[str]:
        return self.images["image_name"].tolist()


def read_keyframe_inputs(sequence_dirs: dict[str, Path]) -> KeyframeInputs:
    """Keyframes, gravity priors and gyro rotations of every sequence (sequence_id -> sequence dir)."""
    images, gravity, image_dirs, gyro = [], [], {}, {}
    for sequence_id, sequence_dir in sequence_dirs.items():
        keyframes = stage_dir(sequence_dir, s02_keyframes.DIR_NAME)
        table = read_csv(keyframes / s02_keyframes.ARTIFACTS["keyframes"], s02_keyframes.KEYFRAMES_COLUMNS)
        images.append(table[["image_name", "idx", "t_ns"]].assign(sequence_id=sequence_id))
        image_dirs[sequence_id] = keyframes / s02_keyframes.ARTIFACTS["images"]
        gravity.append(read_csv(keyframes / s02_keyframes.ARTIFACTS["gravity_priors"], s02_keyframes.GRAVITY_COLUMNS))
        ingest = stage_dir(sequence_dir, s01_ingest.DIR_NAME)
        if "frame_rotations" in read_manifest(ingest).artifacts:
            R_cum = np.load(ingest / s01_ingest.ARTIFACTS["frame_rotations"])["R_cum"]
            info = read_json(ingest / s01_ingest.ARTIFACTS["sequence"])
            R_cam_imu = np.asarray(info["extrinsics"][s02_keyframes.CAMERA_ID]["R_cam_imu"])
            gyro[sequence_id] = GyroRotations(R_cum, R_cam_imu)
    table = pd.concat(images, ignore_index=True).sort_values("image_name", ignore_index=True)
    return KeyframeInputs(table, image_dirs, pd.concat(gravity, ignore_index=True), gyro)


def gravity_vectors(gravity: pd.DataFrame, min_conf: float) -> dict[str, np.ndarray]:
    """Image name -> down vector in the camera frame, for the finite `gravity_cam.csv` rows with `conf >= min_conf`."""
    rows = gravity.dropna(subset=["gx", "gy", "gz", "conf"])
    rows = rows[rows["conf"] >= min_conf]
    vectors = rows[["gx", "gy", "gz"]].to_numpy(dtype=float)
    return dict(zip(rows["image_name"], vectors))
