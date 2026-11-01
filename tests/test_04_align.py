import shutil

import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from recon.core.geometry import Sim3, angle_deg
from recon.io.tables import read_csv, read_json
from recon.pipeline.cli import main
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s02_keyframes import ARTIFACTS as KEYFRAME_ARTIFACTS
from recon.pipeline.stages.s02_keyframes import GRAVITY_COLUMNS, KEYFRAMES_COLUMNS
from recon.pipeline.stages.s03_sfm import ARTIFACTS as SFM_ARTIFACTS
from recon.pipeline.stages.s04_align import ARTIFACTS
from recon.sfm.quality import camera_centres

REPORT_KEYS = {
    "input_model",
    "num_registered",
    "mean_reproj_px",
    "sfm",
    "gravity",
    "floor",
    "W_from_sfm",
    "extent",
    "units",
    "timings",
    "warnings",
}
GRAVITY_KEYS = {
    "method",
    "num_priors",
    "num_used",
    "iterations",
    "down_vs_model_y_deg",
    "residual_median_deg",
    "residual_p90_deg",
    "residual_max_deg",
    "num_over_huber",
}
FLOOR_KEYS = {
    "detected",
    "tilt_to_gravity_deg",
    "tilt_applied",
    "inlier_frac",
    "band_frac",
    "rms",
    "plane",
    "cameras_above_frac",
    "camera_height_median",
}


@pytest.fixture(scope="module")
def report(align_run):
    return read_json(align_run / ARTIFACTS["report"])


@pytest.fixture(scope="module")
def transform(align_run):
    return read_json(align_run / ARTIFACTS["transform"])


@pytest.fixture(scope="module")
def sid(pipeline_run):
    return pipeline_run.sequence_ids[0]


@pytest.fixture(scope="module")
def models(sfm_run, align_run):
    """The refined SfM model and the aligned one."""
    rec = pycolmap.Reconstruction(sfm_run / SFM_ARTIFACTS["model"])
    return rec, pycolmap.Reconstruction(align_run / ARTIFACTS["model"])


def _extent(points):
    lo, hi = np.percentile(points, [2, 98], axis=0)
    return (hi - lo).max()


def _up_in_sfm(transform):
    """World up (+Z) expressed in the SfM frame."""
    return Sim3.from_dict(transform["W_from_sfm"]).R.T @ [0.0, 0.0, 1.0]


def test_manifest(align_run, sid):
    manifest = read_manifest(align_run)
    assert manifest.status == "success"
    assert manifest.artifacts == ARTIFACTS
    for rel in manifest.artifacts.values():
        assert (align_run / rel).exists()
    assert sorted(p.name for p in (align_run / ARTIFACTS["trajectory"]).iterdir()) == [f"{sid}.txt"]
    assert list(manifest.upstream) == ["sfm"]
    assert manifest.backends == {}
    assert manifest.warnings == []
    assert manifest.metrics["gravity_method"] == "imu" and manifest.metrics["floor_detected"]


def test_report(report, sfm_run):
    assert set(report) == REPORT_KEYS
    assert set(report["gravity"]) == GRAVITY_KEYS and set(report["floor"]) == FLOOR_KEYS
    assert report["input_model"] == "03_sfm/sparse/refined" and report["units"] == "arbitrary"
    assert set(report["timings"]) == {"load", "gravity", "floor", "write"}
    gravity, floor = report["gravity"], report["floor"]
    assert gravity["method"] == "imu" and gravity["num_used"] >= 3
    assert gravity["residual_median_deg"] < 1.5 and gravity["residual_p90_deg"] < 3
    assert floor["detected"] and floor["tilt_to_gravity_deg"] < 2 and floor["tilt_applied"]
    assert floor["cameras_above_frac"] >= 0.9
    sfm_report = read_json(sfm_run / SFM_ARTIFACTS["report"])
    assert np.isclose(report["mean_reproj_px"], sfm_report["mean_reproj_px"])
    assert report["num_registered"] == sfm_report["num_registered"]
    assert report["sfm"]["mapper_used"] == sfm_report["mapper_used"]


def test_model(models):
    rec, aligned = models
    assert {i.name for i in aligned.images.values()} == {i.name for i in rec.images.values()}
    assert aligned.num_reg_images() == rec.num_reg_images()
    assert aligned.num_points3D() == rec.num_points3D()


def test_transform(models, transform):
    rec, aligned = models
    assert transform["units"] == "arbitrary" and transform["up_axis"] == "+Z"
    assert transform["gravity"] == [0.0, 0.0, -1.0]
    assert "enabled" not in transform and [s["name"] for s in transform["steps"]] == ["gravity", "floor"]
    W = Sim3.from_dict(transform["W_from_sfm"])
    assert W.scale == 1.0
    assert np.allclose(W.R @ W.R.T, np.eye(3)) and np.isclose(np.linalg.det(W.R), 1.0)
    # the steps, applied in list order, compose to W_from_sfm
    composed = Sim3()
    for step in transform["steps"]:
        composed = Sim3.from_dict(step) @ composed
    assert np.allclose(composed.matrix(), W.matrix())
    before, after = camera_centres(rec), camera_centres(aligned)
    assert list(before) == list(after)
    assert np.allclose(W.apply(np.array(list(before.values()))), np.array(list(after.values())), atol=1e-6)


def test_world_frame(pipeline_run, sid, models, report):
    _, aligned = models
    points = np.array([p.xyz for p in aligned.points3D.values()])
    extent = _extent(points)
    floor = points[np.abs(points[:, 2]) < 0.03 * extent]
    assert len(floor) > 0.05 * len(points)
    assert abs(np.median(floor[:, 2])) < 0.01 * extent
    centres = np.array(list(camera_centres(aligned).values()))
    assert (centres[:, 2] > 0).mean() >= 0.9
    # the gravity priors, rotated into W, point down (up to the applied floor tilt)
    gravity = read_csv(pipeline_run.stage_dir("keyframes", sid) / KEYFRAME_ARTIFACTS["gravity_priors"], GRAVITY_COLUMNS)
    gravity = gravity[gravity["conf"] >= 0.3].set_index("image_name")
    down = [
        image.cam_from_world().rotation.matrix().T @ gravity.loc[image.name, ["gx", "gy", "gz"]].to_numpy(float)
        for image in (aligned.images[i] for i in aligned.reg_image_ids())
        if image.name in gravity.index
    ]
    assert angle_deg(np.mean(down, axis=0), [0.0, 0.0, -1.0]) < 2


def test_tum(pipeline_run, sid, align_run, models):
    _, aligned = models
    keyframes = read_csv(pipeline_run.stage_dir("keyframes", sid) / KEYFRAME_ARTIFACTS["keyframes"], KEYFRAMES_COLUMNS)
    t_s = dict(zip(keyframes["image_name"], keyframes["t_ns"].astype(int)))
    rows = np.loadtxt(align_run / ARTIFACTS["trajectory"] / f"{sid}.txt", ndmin=2)
    centres = camera_centres(aligned)
    names = sorted(centres, key=t_s.get)
    assert rows.shape == (aligned.num_reg_images(), 8) == (len(names), 8)
    assert (np.diff(rows[:, 0]) > 0).all()
    assert np.allclose(rows[:, 0], [t_s[name] * 1e-9 for name in names], rtol=0, atol=1e-6)
    assert np.allclose(np.linalg.norm(rows[:, 4:], axis=1), 1.0, atol=1e-6)
    assert np.allclose(rows[:, 1:4], [centres[name] for name in names], atol=1e-6)
    # camera-to-world rotations
    images = {image.name: image for image in aligned.images.values()}
    R_WC = [images[name].cam_from_world().rotation.matrix().T for name in names]
    assert np.allclose(Rotation.from_quat(rows[:, 4:]).as_matrix(), R_WC, atol=1e-6)


def _align(capture, root, *args):
    return main(["align", "--input", str(capture), "--output-root", str(root), *args])


def test_cli_skips(tiny_capture, output_root, align_run):
    mtime = (align_run / "stage_manifest.json").stat().st_mtime_ns
    assert _align(tiny_capture, output_root, "--resume", align_run.parent.name) == 0
    assert (align_run / "stage_manifest.json").stat().st_mtime_ns == mtime


def test_cli_needs_sfm(tiny_capture, output_root, pipeline_run, tmp_path, capsys):
    shutil.copytree(output_root / "room2_tiny" / "sequences", tmp_path / "room2_tiny" / "sequences")
    assert _align(tiny_capture, tmp_path) == 1
    assert "run `recon sfm` first" in capsys.readouterr().out


@pytest.fixture(scope="module")
def camera_up(tiny_capture, output_root, align_run, tmp_path_factory):
    """align with `gravity.method=camera_up` on a copy of the shared sequences and run; returns its stage dir."""
    root = tmp_path_factory.mktemp("camera_up")
    run_dir = align_run.parent
    shutil.copytree(output_root / "room2_tiny" / "sequences", root / "room2_tiny" / "sequences")
    shutil.copytree(run_dir, root / "room2_tiny" / "runs" / run_dir.name, symlinks=True)
    args = ["--resume", run_dir.name, "--force", "--set", "align.gravity.method=camera_up"]
    assert _align(tiny_capture, root, *args) == 0
    stages = read_json(root / "room2_tiny" / "runs" / run_dir.name / "run_manifest.json")["stages"]
    assert [(s["stage"], s["status"]) for s in stages] == [("align", "success")]
    return root / "room2_tiny" / "runs" / run_dir.name / align_run.name


def test_camera_up_fallback(camera_up, transform):
    report = read_json(camera_up / ARTIFACTS["report"])
    assert report["gravity"]["method"] == "camera_up"
    assert any("up from the image-down direction" in w for w in read_manifest(camera_up).warnings)
    assert report["floor"]["detected"]
    fallback = read_json(camera_up / ARTIFACTS["transform"])
    assert angle_deg(_up_in_sfm(fallback), _up_in_sfm(transform)) < 5
