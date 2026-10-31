import dataclasses
import re
import shutil

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import yaml

from recon.io.tables import read_csv, read_json
from recon.keyframes.metrics import METRIC_COLUMNS, SCORE_COLUMNS, sharpness, to_gray
from recon.keyframes.priors import cam_rotation_from_device
from recon.pipeline.cli import main
from recon.pipeline.config import DEFAULT_CONFIG, SECTIONS, ConfigError, PipelineConfig, load_config
from recon.pipeline.run_dir import manifest_hash, read_manifest
from recon.pipeline.stages.common import open_loader
from recon.pipeline.stages.s02_keyframes import (
    ARTIFACTS,
    GRAVITY_COLUMNS,
    KEYFRAMES_COLUMNS,
    ROTATION_COLUMNS,
)

IMAGE_NAME_RE = re.compile(r"room2_tiny_20261021_211400/cam0/\d{6}\.jpg")


@pytest.fixture(scope="module")
def keyframes(pipeline_run):
    out = pipeline_run.stage_dir("keyframes", pipeline_run.sequence_ids[0])
    return read_manifest(out), out


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
    assert manifest.upstream == {"ingest": manifest_hash(out.parent / "01_ingest")}

    report = read_json(out / ARTIFACTS["selection_report"])
    assert report["num_frames"] == 300 and report["num_keyframes"] == 15
    assert report["method"] == "uniform" and report["imu_used"]
    assert report["max_gap_s"] <= 0.7
    assert sum(report["num_rejected_by_reason"].values()) == manifest.metrics["num_rejected"]
    assert report["mean_sharp_rel_selected"] >= report["mean_sharp_rel_all"]


def test_uniform_selection(table):
    assert len(table) == 15
    assert np.all(np.diff(table["idx"]) > 0)
    assert np.all(np.diff(table["t_ns"]) > 0)
    # one keyframe within half a tick of each tick, so neighbours are less than two ticks apart
    assert np.isnan(table["gap_s"].iloc[0]) and (table["gap_s"].iloc[1:] < 2 / 3).all()
    assert set(table["trigger"]) <= {"uniform", "forced"}
    assert table["quality"].notna().all() and table["sharp_rel"].notna().all()


@pytest.fixture(scope="module")
def scores(out):
    return read_csv(out / ARTIFACTS["frame_scores"], SCORE_COLUMNS)


def test_frame_scores(scores):
    assert len(scores) == 300 and scores["idx"].tolist() == list(range(300))
    assert not scores[[*METRIC_COLUMNS, "sharp_rel", "omega_norm", "q"]].isna().any().any()
    assert scores["rejected_reason"].dropna().isin(["sharp_rel", "clipped", "feat_coverage", "omega_norm"]).all()


def test_quality_aware_selection(scores, table):
    selected = scores.set_index("idx").loc[table["idx"]]
    assert selected["sharp_rel"].median() >= scores["sharp_rel"].median()
    assert np.allclose(selected["q"], table["quality"]) and np.allclose(selected["sharp_rel"], table["sharp_rel"])
    rejected = selected["rejected_reason"].notna().to_numpy()
    assert (table["trigger"].to_numpy()[rejected] == "forced").all()


def test_blur_lowers_sharpness(pipeline_run):
    loader = open_loader(pipeline_run.sequence_dir(pipeline_run.sequence_ids[0]))
    for frame in loader.iter_frames(indices=[0, 150, 299], scale=0.5):
        gray = to_gray(frame.image)
        assert sharpness(cv2.GaussianBlur(gray, (0, 0), 2.0)) < sharpness(gray)


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


def _run(capture, root, *args):
    return main(["run", "--input", str(capture), "--output-root", str(root), "--to", "keyframes", *args])


def _last_run(root):
    run_dir = sorted((root / "room2_tiny" / "runs").iterdir())[-1]
    return {s["stage"]: s["status"] for s in read_json(run_dir / "run_manifest.json")["stages"]}


def _mtimes(seq_dir):
    return {p: p.stat().st_mtime_ns for p in seq_dir.rglob("*")}


def test_rerun_skips_up_to_date_stages(tiny_capture, output_root, keyframes):
    seq_dir = keyframes[1].parent
    before = _mtimes(seq_dir)
    assert _run(tiny_capture, output_root) == 0
    assert _last_run(output_root) == {"ingest": "skipped", "keyframes": "skipped"}
    assert _mtimes(seq_dir) == before


def test_changed_key_reruns_only_keyframes(tiny_capture, output_root, keyframes, tmp_path):
    # on a copy of the shared run, so the other tests keep their 3 fps keyframes
    shutil.copytree(output_root / "room2_tiny" / "sequences", tmp_path / "room2_tiny" / "sequences")
    seq_dir = tmp_path / "room2_tiny" / "sequences" / keyframes[1].parent.name
    ingest_before = read_manifest(seq_dir / "01_ingest")

    assert _run(tiny_capture, tmp_path, "--set", "keyframes.uniform_fps=2") == 0
    assert _last_run(tmp_path) == {"ingest": "skipped", "keyframes": "success"}
    assert read_manifest(seq_dir / "01_ingest") == ingest_before
    manifest = read_manifest(seq_dir / "02_keyframes")
    assert manifest.metrics["num_keyframes"] == 10
    assert manifest.config_hash != keyframes[0].config_hash
    assert len(list((seq_dir / "02_keyframes" / "images" / "cam0").iterdir())) == 10


def test_config_errors(tiny_capture, output_root):
    with pytest.raises(ConfigError, match=r"keyframes\.no_such_key"):
        load_config(overrides=["keyframes.no_such_key=1"])
    with pytest.raises(ConfigError, match=r"keyframes\.reject\.no_such_key"):
        load_config(overrides=["keyframes.reject.no_such_key=1"])
    with pytest.raises(ConfigError, match=r"keyframes\.uniform_fps"):
        load_config(overrides=["keyframes.uniform_fps=fast"])
    assert _run(tiny_capture, output_root, "--set", "keyframes.no_such_key=1") == 2


def _keys(cls, data=None, prefix=""):
    """Dotted names of the option fields of `cls`, or of the keys of `data` laid over it; nested options recurse."""
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    keys = set()
    for name in types if data is None else data:
        nested = dataclasses.is_dataclass(types.get(name))
        sub = None if data is None else data[name]
        keys |= _keys(types[name], sub, f"{prefix}{name}.") if nested else {prefix + name}
    return keys


def test_default_config_lists_every_key():
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert {name: _keys(cls, data[name]) for name, cls in SECTIONS.items()} == {
        name: _keys(cls) for name, cls in SECTIONS.items()
    }
    assert load_config() == PipelineConfig()
