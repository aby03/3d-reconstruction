import numpy as np
import pytest
from conftest import assert_stage_ok

from recon.capture.open_camera_sensors import discover_files
from recon.core.frames import DEVICE_ORIENTATIONS, R_cam_imu_upright, device_orientation_from_gravity
from recon.core.types import SensorKind


def test_discovery_from_root_and_stamp_dir(smoke_capture):
    files = discover_files(smoke_capture)
    assert discover_files(files.stamp_dir) == files
    assert files.video.is_file() and files.frame_timestamps.is_file()
    assert set(files.sensors) == set(SensorKind)
    assert files.reference_images
    assert files.warnings == []


def test_smoke(smoke_run):
    for seq in smoke_run.inputs:
        assert_stage_ok(smoke_run.stage_dir("ingest", seq))


def test_nominal_R_cam_imu():
    # back camera, portrait: image x/y/z = device x/-y/-z
    assert np.array_equal(R_cam_imu_upright(0), np.diag([1.0, -1.0, -1.0]))
    for o in DEVICE_ORIENTATIONS:
        assert device_orientation_from_gravity(R_cam_imu_upright(o).T @ [0.0, 1.0, 0.0]) == o


def test_frame_timing_matches_extra_timestamps_by_video_time():
    from recon.capture.timing import check_frame_timing

    jitter = np.random.default_rng(0).normal(0, 2e-4, 103)  # a real clock is not perfectly even
    pts = (np.arange(100) / 60.0 + jitter[:100])
    stamps = ((np.arange(103) / 60.0 + jitter) * 1e9).astype(np.int64) + 5_000_000_000
    check = check_frame_timing(pts, stamps)  # extra timestamps at the end
    assert check.ok and check.t_offset == 0 and check.num_frames == 100
    check = check_frame_timing(pts, stamps[1:101])  # one extra at the start of a shifted list
    assert check.num_frames == 100
    with pytest.raises(ValueError):
        check_frame_timing(pts, stamps[:90])
    assert not check_frame_timing(pts, stamps[:90], allow_mismatch=True).ok
