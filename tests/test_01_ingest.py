import numpy as np
import pytest

from recon.capture.open_camera_sensors import OpenCameraSensorsLoader, discover_files, read_sensor_csv
from recon.core.types import SensorKind
from recon.sensors.streams import clean_stream, stream_stats

EXPECTED_RATE_HZ = {SensorKind.ACCEL: 457.0, SensorKind.GYRO: 457.0, SensorKind.MAG: 100.0}


def test_discovery_from_root_and_stamp_dir(tiny_capture):
    files = discover_files(tiny_capture)
    assert discover_files(files.stamp_dir) == files
    assert files.video.is_file() and files.frame_timestamps.is_file()
    assert set(files.sensors) == set(SensorKind)
    assert len(files.reference_images) >= 3
    assert files.warnings == []


@pytest.mark.parametrize("kind", list(SensorKind))
def test_sensor_streams(tiny_capture, kind):
    path = discover_files(tiny_capture).sensors[kind]
    stream, counts = clean_stream(read_sensor_csv(path, kind))

    assert stream.t_ns.dtype == np.int64
    assert stream.values.shape == (len(stream.t_ns), 3)
    assert np.all(np.diff(stream.t_ns) > 0)
    assert counts == {"out_of_order": 0, "duplicates": 0}
    # int64 end to end: the last timestamp equals the raw text exactly
    assert stream.t_ns[-1] == int(path.read_text().split()[-1].rsplit(",", 1)[-1])
    assert stream_stats(stream)["rate_hz"] == pytest.approx(EXPECTED_RATE_HZ[kind], rel=0.05)


@pytest.fixture(scope="module")
def sequence(tiny_capture):
    return OpenCameraSensorsLoader(tiny_capture).load()


def test_frame_timing(sequence):
    timing = sequence.checks["frame_timing"]
    assert timing["num_pts"] == timing["num_timestamps"] == 300
    assert timing["drops_pts"] == timing["drops_t"] == [1]
    assert timing["delta_corr"] > 0.95
    assert timing["residual_max_ms"] < 1.0
    assert timing["ok"]


def test_sequence(sequence):
    assert sequence.sequence_id == "room2_tiny_20261021_211400"
    assert set(sequence.sensors) == set(SensorKind)
    assert sequence.checks["warnings"] == []
    assert len(sequence.reference_images) == 5

    cam = sequence.cameras["cam0"]
    assert cam.num_frames == len(cam.pts_s) == len(cam.frame_t_ns) == 300
    assert np.all(np.diff(cam.pts_s) > 0)
    assert cam.stored_size == (1920, 1080)
    assert cam.rotation_deg == 90  # display matrix says -90
    assert cam.upright_size == (1080, 1920)
