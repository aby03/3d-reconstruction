import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recon.capture.open_camera_sensors import (
    LoaderOptions,
    OpenCameraSensorsLoader,
    discover_files,
    read_sensor_csv,
)
from recon.core.frames import DEVICE_ORIENTATIONS, R_cam_imu_upright, device_orientation_from_gravity
from recon.core.types import SensorKind
from recon.sensors.gyro import integrate_rotation
from recon.sensors.streams import clean_stream, stream_stats
from recon.sensors.sync import COLUMNS, SyncOptions, synchronize

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


def test_iter_frames_upright(tiny_capture, sequence):
    loader = OpenCameraSensorsLoader(tiny_capture, LoaderOptions(verify_reference_jpegs=False))
    t_ns = sequence.cameras["cam0"].frame_t_ns
    frames = list(loader.iter_frames(indices=[299, 0, 150, 1]))
    assert [f.idx for f in frames] == [0, 1, 150, 299]
    for f in frames:
        assert f.camera_id == "cam0"
        assert f.image.shape == (1920, 1080, 3) and f.image.dtype == np.uint8
        assert f.t_ns == t_ns[f.idx]

    (half,) = loader.iter_frames(indices=[0], scale=0.5)
    assert half.image.shape == (960, 540, 3)


def test_extract_frames(tiny_capture, tmp_path):
    loader = OpenCameraSensorsLoader(tiny_capture, LoaderOptions(verify_reference_jpegs=False))
    paths = loader.extract_frames([150], tmp_path)
    assert paths == {150: tmp_path / "cam0" / "000150.jpg"}
    assert paths[150].stat().st_size > 0


def test_reference_images(sequence):
    ref = sequence.checks["reference_images"]
    assert ref["checked"] == 5
    assert ref["min_psnr_db"] > 25
    assert ref["best_rotation"] == sequence.cameras["cam0"].rotation_deg == 90
    assert not ref["rotation_mismatch"]
    assert "rotation_mismatch" not in sequence.checks


def test_synchronize(sequence):
    sync = synchronize(sequence, SyncOptions())
    df = sync.frames
    assert list(df.columns) == COLUMNS
    assert len(df) == 300 and np.array_equal(df["idx"], np.arange(300))
    assert np.array_equal(df["t_ns"], sequence.cameras["cam0"].frame_t_ns)
    assert df["valid"].all()
    assert df["accel_dt_ms"].max() < 3 and df["gyro_dt_ms"].max() < 3
    assert df[["mag_x", "mag_y", "mag_z"]].notna().all().all()
    assert sync.report["frames_valid_ratio"] == 1.0


def test_nominal_R_cam_imu():
    # back camera, portrait: image x/y/z = device x/-y/-z
    assert np.array_equal(R_cam_imu_upright(0), np.diag([1.0, -1.0, -1.0]))
    for o in DEVICE_ORIENTATIONS:
        assert device_orientation_from_gravity(R_cam_imu_upright(o).T @ [0.0, 1.0, 0.0]) == o


def test_gravity_and_gyro_rotations(sequence):
    assert sequence.checks["device_orientation"] == {"deg": 0, "from_display": 0, "from_gravity": 0}
    ext = sequence.extrinsics["cam0"]
    assert ext.source == "nominal" and np.array_equal(ext.R_cam_imu, R_cam_imu_upright(0))

    sync = synchronize(sequence, SyncOptions())
    df = sync.frames
    grav_cam = df[["grav_cam_x", "grav_cam_y", "grav_cam_z"]].to_numpy()
    assert np.allclose(np.linalg.norm(grav_cam, axis=1), 1.0)
    assert grav_cam.mean(axis=0)[1] > 0.95
    assert df["grav_conf"].median() > 0.5

    t = sequence.cameras["cam0"].frame_t_ns
    dR = sync.dR_prev
    assert dR.shape == (300, 4) and np.allclose(dR[0], [0, 0, 0, 1])
    chained = Rotation.identity()
    for q in dR[1:]:
        chained = chained * Rotation.from_quat(q)
    whole = Rotation.from_matrix(integrate_rotation(sequence.sensors[SensorKind.GYRO], t[0], t[-1]))
    assert (chained.inv() * whole).magnitude() < 1e-6
