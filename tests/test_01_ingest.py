import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recon.capture.open_camera_sensors import LoaderOptions, OpenCameraSensorsLoader, discover_files
from recon.core.frames import ANDROID_DEVICE, DEVICE_ORIENTATIONS, R_cam_imu_upright, device_orientation_from_gravity
from recon.core.types import SensorKind, SensorStream
from recon.io.tables import read_csv, read_json
from recon.pipeline.cli import main
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s01_ingest import ARTIFACTS, FRAMES_COLUMNS, SENSOR_COLUMNS
from recon.sensors.gyro import integrate_rotation
from recon.sensors.sync import COLUMNS, GRAV_CAM_COLUMNS

EXPECTED_RATE_HZ = {SensorKind.ACCEL: 457.0, SensorKind.GYRO: 457.0, SensorKind.MAG: 100.0}


def test_discovery_from_root_and_stamp_dir(tiny_capture):
    files = discover_files(tiny_capture)
    assert discover_files(files.stamp_dir) == files
    assert files.sequence_id == "room2_tiny_20261021_211400"
    assert files.video.is_file() and files.frame_timestamps.is_file()
    assert set(files.sensors) == set(SensorKind)
    assert len(files.reference_images) >= 3
    assert files.warnings == []


@pytest.fixture(scope="module")
def out(ingest):
    return ingest[1]


@pytest.fixture(scope="module")
def report(out):
    return read_json(out / ARTIFACTS["sync_report"])


@pytest.fixture(scope="module")
def sequence_json(out):
    return read_json(out / ARTIFACTS["sequence"])


def test_manifest_and_artifacts(ingest):
    manifest, out = ingest
    assert manifest.status == "success"
    assert manifest.warnings == []
    assert manifest.artifacts == ARTIFACTS
    for rel in manifest.artifacts.values():
        assert (out / rel).is_file() and (out / rel).stat().st_size > 0
    assert read_manifest(out) == manifest


@pytest.mark.parametrize("kind", list(SensorKind))
def test_sensor_streams(tiny_capture, out, report, kind):
    df = read_csv(out / ARTIFACTS[f"sensors/{kind}"], SENSOR_COLUMNS)
    assert list(df.columns) == SENSOR_COLUMNS
    assert df["t_ns"].dtype == np.int64
    assert np.all(np.diff(df["t_ns"]) > 0)
    # int64 end to end: the last timestamp equals the raw text exactly
    raw = discover_files(tiny_capture).sensors[kind]
    assert df["t_ns"].iloc[-1] == int(raw.read_text().split()[-1].rsplit(",", 1)[-1])

    stats = report["sensors"][kind]
    assert stats["n"] == len(df)
    assert stats["dup_removed"] == stats["out_of_order"] == 0
    assert stats["rate_hz"] == pytest.approx(EXPECTED_RATE_HZ[kind], rel=0.05)


def test_frame_timing(out, report):
    assert report["num_frames"] == 300
    assert report["dropped_frames"] == [1]
    assert report["pts_ts_corr"] > 0.95
    assert report["pts_ts_resid_max_ms"] < 1.0
    assert report["frame_timing_ok"]

    frames = read_csv(out / ARTIFACTS["frames"], FRAMES_COLUMNS)
    assert len(frames) == 300 and np.array_equal(frames["idx"], np.arange(300))
    assert np.flatnonzero(frames["dropped_before"]).tolist() == [1]
    assert np.all(np.diff(frames["pts_s"]) > 0)
    assert (frames["ref_jpeg"].fillna("") != "").sum() == 5


def test_sequence(sequence_json):
    assert sequence_json["sequence_id"] == "room2_tiny_20261021_211400"
    assert set(sequence_json["sensors"]) == {str(k) for k in SensorKind}

    cam = sequence_json["cameras"]["cam0"]
    assert cam["num_frames"] == 300
    assert cam["stored_size"] == [1920, 1080]
    assert cam["rotation_deg"] == 90  # display matrix says -90
    assert cam["upright_size"] == [1080, 1920]


def test_iter_frames_upright(tiny_capture, out):
    loader = OpenCameraSensorsLoader(tiny_capture, LoaderOptions(verify_reference_jpegs=False))
    t_ns = read_csv(out / ARTIFACTS["frames"], ["t_ns"])["t_ns"].to_numpy()
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


def test_reference_images(report, sequence_json):
    ref = report["ref_jpeg_psnr_db"]
    assert ref["checked"] == 5
    assert ref["min"] > 25
    assert ref["best_rotation"] == sequence_json["cameras"]["cam0"]["rotation_deg"] == 90
    assert not ref["rotation_mismatch"]


def test_frame_sensor_data(out, report):
    df = read_csv(out / ARTIFACTS["frame_sensor_data"], COLUMNS)
    assert list(df.columns) == COLUMNS
    assert len(df) == 300 and np.array_equal(df["idx"], np.arange(300))
    frames = read_csv(out / ARTIFACTS["frames"], ["t_ns"])
    assert np.array_equal(df["t_ns"], frames["t_ns"])
    assert df["valid"].all()
    assert df["accel_dt_ms"].max() < 3 and df["gyro_dt_ms"].max() < 3
    assert df[["mag_x", "mag_y", "mag_z"]].notna().all().all()
    assert report["frames_valid_ratio"] == 1.0


def test_nominal_R_cam_imu():
    # back camera, portrait: image x/y/z = device x/-y/-z
    assert np.array_equal(R_cam_imu_upright(0), np.diag([1.0, -1.0, -1.0]))
    for o in DEVICE_ORIENTATIONS:
        assert device_orientation_from_gravity(R_cam_imu_upright(o).T @ [0.0, 1.0, 0.0]) == o


def test_gravity_and_gyro_rotations(out, sequence_json):
    assert sequence_json["device_orientation_deg"] == 0
    ext = sequence_json["extrinsics"]["cam0"]
    assert ext["source"] == "nominal" and np.array_equal(ext["R_cam_imu"], R_cam_imu_upright(0))

    df = read_csv(out / ARTIFACTS["frame_sensor_data"], COLUMNS)
    grav_cam = df[GRAV_CAM_COLUMNS].to_numpy()
    assert np.allclose(np.linalg.norm(grav_cam, axis=1), 1.0)
    assert grav_cam.mean(axis=0)[1] > 0.95
    assert df["grav_conf"].median() > 0.5

    rotations = np.load(out / ARTIFACTS["frame_rotations"])
    dR, R_cum, t = rotations["dR_prev"], rotations["R_cum"], rotations["t_ns"]
    assert dR.shape == R_cum.shape == (300, 4)
    assert np.allclose(dR[0], [0, 0, 0, 1]) and np.allclose(R_cum[0], [0, 0, 0, 1])
    assert np.array_equal(t, df["t_ns"])

    chained = Rotation.identity()
    for q in dR[1:]:
        chained = chained * Rotation.from_quat(q)
    assert (chained.inv() * Rotation.from_quat(R_cum[-1])).magnitude() < 1e-9

    gyro_df = read_csv(out / ARTIFACTS["sensors/gyro"], SENSOR_COLUMNS)
    gyro = SensorStream(
        kind=SensorKind.GYRO,
        t_ns=gyro_df["t_ns"].to_numpy(),
        values=gyro_df[["x", "y", "z"]].to_numpy(),
        units="rad/s",
        frame=ANDROID_DEVICE,
    )
    whole = Rotation.from_matrix(integrate_rotation(gyro, t[0], t[-1]))
    assert (Rotation.from_quat(R_cum[-1]).inv() * whole).magnitude() < 1e-6


def test_cli_keeps_finished_ingest(tiny_capture, output_root, ingest):
    manifest, out = ingest
    assert main(["ingest", "--input", str(tiny_capture), "--output-root", str(output_root)]) == 0
    assert read_manifest(out) == manifest
    assert (out.parent / "logs" / "pipeline.log").is_file()
