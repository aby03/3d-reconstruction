import re

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recon.io.tables import read_csv, read_json
from recon.keyframes.priors import cam_rotation_from_device
from recon.keyframes.select import KeyframeOptions
from recon.pipeline.cli import main
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s02_keyframes import (
    ARTIFACTS,
    DIR_NAME,
    GRAVITY_COLUMNS,
    KEYFRAMES_COLUMNS,
    ROTATION_COLUMNS,
    run_keyframes,
)

IMAGE_NAME_RE = re.compile(r"room2_tiny_20261021_211400/cam0/\d{6}\.jpg")


@pytest.fixture(scope="module")
def keyframes(ingest):
    seq_dir = ingest[1].parent
    manifest = run_keyframes(seq_dir, KeyframeOptions(uniform_fps=3.0))
    return manifest, seq_dir / DIR_NAME


@pytest.fixture(scope="module")
def out(keyframes):
    return keyframes[1]


@pytest.fixture(scope="module")
def table(out):
    return read_csv(out / ARTIFACTS["keyframes"], KEYFRAMES_COLUMNS)


def test_manifest_and_report(keyframes):
    manifest, out = keyframes
    assert manifest.status == "success"
    assert manifest.warnings == []
    assert manifest.artifacts == ARTIFACTS
    for rel in manifest.artifacts.values():
        assert (out / rel).exists()
    assert read_manifest(out) == manifest

    report = read_json(out / ARTIFACTS["selection_report"])
    assert report["num_frames"] == 300 and report["num_keyframes"] == 15
    assert report["method"] == "uniform" and report["imu_used"]
    assert report["max_gap_s"] <= 0.4


def test_uniform_selection(table):
    assert len(table) == 15
    assert np.all(np.diff(table["idx"]) > 0)
    assert np.all(np.diff(table["t_ns"]) > 0)
    assert np.isnan(table["gap_s"].iloc[0]) and (table["gap_s"].iloc[1:] <= 0.4).all()
    assert (table["trigger"] == "uniform").all()
    assert table["quality"].isna().all() and table["sharp_rel"].isna().all()


def test_images(out, table):
    for name, idx in zip(table["image_name"], table["idx"]):
        assert IMAGE_NAME_RE.fullmatch(name) and name.endswith(f"/{idx:06d}.jpg")
        image = cv2.imread(str(out / "images" / name.split("/", 1)[1]))
        assert image.shape == (1920, 1080, 3)


def test_gravity_priors(out, table):
    df = read_csv(out / ARTIFACTS["gravity_priors"], GRAVITY_COLUMNS)
    assert df["image_name"].tolist() == table["image_name"].tolist()
    g = df[["gx", "gy", "gz"]].to_numpy()
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)
    assert g.mean(axis=0)[1] > 0.9


def test_relative_rotation_priors(out, table):
    df = read_csv(out / ARTIFACTS["relative_rotations"], ROTATION_COLUMNS)
    pos = {name: k for k, name in enumerate(table["image_name"])}
    steps = df["image_j"].map(pos) - df["image_i"].map(pos)
    assert set(steps) == {1, 2, 3}
    assert len(df) == 14 + 13 + 12
    assert (df["dt_s"] > 0).all()

    R = {(i, j): Rotation.from_quat(q) for i, j, *q in df[["image_i", "image_j", "qx", "qy", "qz", "qw"]].to_numpy()}
    names = table["image_name"].tolist()
    for i, j, k in zip(names, names[1:], names[2:]):
        assert np.degrees((R[i, k].inv() * R[j, k] * R[i, j]).magnitude()) < 0.5


def test_cam_rotation_convention():
    # the device turns by theta about its y axis; seen from the camera (y = -device y) that is a turn about -y,
    # and the stored cam_j_from_cam_i is its inverse
    theta = 0.3
    R_Di_Dj = Rotation.from_rotvec([0.0, theta, 0.0]).as_matrix()
    R_Cj_Ci = cam_rotation_from_device(R_Di_Dj, np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(R_Cj_Ci.T, Rotation.from_rotvec([0.0, -theta, 0.0]).as_matrix())


def test_cli(tiny_capture, output_root, keyframes, tmp_path):
    manifest, out = keyframes
    assert main(["keyframes", "--input", str(tiny_capture), "--output-root", str(output_root)]) == 0
    assert read_manifest(out) == manifest
    assert main(["keyframes", "--input", str(tiny_capture), "--output-root", str(tmp_path)]) == 1
