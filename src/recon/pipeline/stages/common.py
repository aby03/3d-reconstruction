"""Helpers shared by stage adapters for reading upstream artifacts."""

from pathlib import Path
from typing import Literal

from recon.capture.base import BaseCaptureLoader
from recon.capture.open_camera_sensors import SENSOR_UNITS, SOURCE_TYPE, LoaderOptions, OpenCameraSensorsLoader
from recon.core.frames import ANDROID_DEVICE
from recon.core.types import SensorKind, SensorStream
from recon.io.tables import read_csv, read_json
from recon.pipeline.run_dir import stage_dir
from recon.pipeline.stages import s01_ingest


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
