import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from recon.io.tables import read_csv, read_json
from recon.pipeline.cli import main
from recon.pipeline.device import probe_device
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s02_keyframes import ARTIFACTS as KEYFRAME_ARTIFACTS
from recon.pipeline.stages.s02_keyframes import GRAVITY_COLUMNS, KEYFRAMES_COLUMNS
from recon.pipeline.stages.s03_sfm import ARTIFACTS
from recon.sfm.priors import PAIR_FILTER_COLUMNS

REPORT_KEYS = {
    "num_images_input",
    "num_registered",
    "registered_ratio",
    "num_models",
    "num_points3D",
    "num_observations",
    "mean_reproj_px",
    "median_track_len",
    "mean_track_len",
    "num_verified_pairs",
    "matching_strategy",
    "loop_detection",
    "mapper_used",
    "mapper_attempts",
    "num_gravity_priors",
    "gravity_prior_used",
    "gravity_spread_deg",
    "pair_filter_removed",
    "pair_filter_checked",
    "pair_filter_disabled",
    "focal_px",
    "distortion",
    "raw",
    "refine",
    "unregistered_images",
    "per_image",
    "backend",
    "timings",
}
INTRINSICS_KEYS = {"model", "width", "height", "params", "fx", "fy", "cx", "cy", "dist"}


@pytest.fixture(scope="module")
def report(sfm_run):
    return read_json(sfm_run / ARTIFACTS["report"])


@pytest.fixture(scope="module")
def sid(pipeline_run):
    return pipeline_run.sequence_ids[0]


@pytest.fixture(scope="module")
def keyframe_names(pipeline_run, sid):
    out = pipeline_run.stage_dir("keyframes", sid)
    return read_csv(out / KEYFRAME_ARTIFACTS["keyframes"], KEYFRAMES_COLUMNS)["image_name"].tolist()


def test_manifest(sfm_run, sid):
    manifest = read_manifest(sfm_run)
    assert manifest.status == "success"
    assert manifest.artifacts == ARTIFACTS
    for rel in manifest.artifacts.values():
        assert (sfm_run / rel).exists()
    assert list(manifest.upstream) == [f"keyframes/{sid}"]
    backend = "cuda" if probe_device().has("sfm") else "cpu"
    assert manifest.backends == {"features": backend, "matching": backend, "mapper": backend}


def test_report(report, keyframe_names):
    assert set(report) == REPORT_KEYS
    assert report["num_images_input"] == len(keyframe_names)
    assert report["registered_ratio"] >= 0.9
    assert report["mean_reproj_px"] < 1.0
    assert report["median_track_len"] >= 3
    assert report["num_points3D"] > 1000
    assert report["num_models"] >= 1 and report["num_verified_pairs"] > 0
    assert report["mapper_used"] == "global" and report["matching_strategy"] == "exhaustive"
    assert set(report["per_image"]) | set(report["unregistered_images"]) == set(keyframe_names)
    assert set(report["timings"]) == {"features", "matching", "pair_filter", "mapping", "refine", "outputs"}


@pytest.mark.parametrize("model, camera_model", [("model_raw", "SIMPLE_RADIAL"), ("model", "OPENCV")])
def test_model(sfm_run, report, keyframe_names, model, camera_model):
    rec = pycolmap.Reconstruction(sfm_run / ARTIFACTS[model])
    stats = report if model == "model" else report["raw"]
    assert rec.num_reg_images() == stats["num_registered"]
    assert np.isclose(rec.compute_mean_reprojection_error(), stats["mean_reproj_px"])
    assert {image.name for image in rec.images.values()} <= set(keyframe_names)
    assert len(rec.cameras) == 1
    camera = next(iter(rec.cameras.values()))
    assert camera.model.name == camera_model
    assert (camera.width, camera.height) == (1080, 1920)
    assert 0.5 * 1920 <= camera.mean_focal_length() <= 3 * 1920
    # every image resolves through the symlinked image root
    assert all((sfm_run / "images" / name).is_file() for name in keyframe_names)


def test_intrinsics(sfm_run, report, sid):
    intrinsics = read_json(sfm_run / ARTIFACTS["intrinsics"])
    key = f"{sid}/cam0"
    assert list(intrinsics) == [key] == list(report["focal_px"])
    cam = intrinsics[key]
    assert set(cam) == INTRINSICS_KEYS and set(cam["dist"]) == {"k1", "k2", "p1", "p2"}
    assert cam["model"] == "OPENCV" and (cam["width"], cam["height"]) == (1080, 1920)
    # the upgrade starts from fx == fy; bundle adjustment refines them separately
    assert cam["fx"] == cam["params"][0] == report["focal_px"][key] and abs(cam["fy"] / cam["fx"] - 1) < 0.01
    assert cam["dist"] == report["distortion"][key]
    assert np.isclose(cam["cx"], 540) and np.isclose(cam["cy"], 960)


def _run(capture, root, *args):
    return main(["run", "--input", str(capture), "--output-root", str(root), *args])


def test_resume_skips_sfm(tiny_capture, output_root, sfm_run):
    mtime = (sfm_run / "stage_manifest.json").stat().st_mtime_ns
    run_id = sfm_run.parent.name
    assert _run(tiny_capture, output_root, "--resume", run_id, "--to", "sfm") == 0
    stages = read_json(sfm_run.parent / "run_manifest.json")["stages"]
    assert [(s["stage"], s["status"]) for s in stages] == [
        ("ingest", "skipped"),
        ("keyframes", "skipped"),
        ("sfm", "skipped"),
    ]
    assert (sfm_run / "stage_manifest.json").stat().st_mtime_ns == mtime
    assert main(["sfm", "--input", str(tiny_capture), "--output-root", str(output_root), "--resume", run_id]) == 0
    assert (sfm_run / "stage_manifest.json").stat().st_mtime_ns == mtime


def test_new_run_without_resume(tiny_capture, output_root, sfm_run):
    runs = output_root / "room2_tiny" / "runs"
    before = set(runs.iterdir())
    assert _run(tiny_capture, output_root, "--only", "keyframes") == 0
    (new,) = set(runs.iterdir()) - before
    assert not (new / "03_sfm").exists()


def test_resume_errors(tiny_capture, output_root, sfm_run, tmp_path):
    assert _run(tiny_capture, output_root, "--resume", "no_such_run") == 2
    assert _run(tiny_capture, tmp_path, "--resume") == 2


def test_cli_needs_keyframes(tiny_capture, tmp_path, capsys):
    assert main(["sfm", "--input", str(tiny_capture), "--output-root", str(tmp_path)]) == 1
    assert "run `recon keyframes` first" in capsys.readouterr().out


def test_gravity_priors(pipeline_run, sid, sfm_run, report):
    gravity = read_csv(pipeline_run.stage_dir("keyframes", sid) / KEYFRAME_ARTIFACTS["gravity_priors"], GRAVITY_COLUMNS)
    assert report["num_gravity_priors"] == (gravity["conf"] >= 0.3).sum() > 0
    assert report["gravity_prior_used"]
    db = pycolmap.Database.open(sfm_run / ARTIFACTS["database"])
    priors = db.read_all_pose_priors()
    db.close()
    assert sum(prior.has_gravity() for prior in priors) == report["num_gravity_priors"]
    # the model and the IMU agree on "down"
    assert report["gravity_spread_deg"]["median"] < 3


def test_pair_filter(sfm_run, report):
    errors = read_csv(sfm_run / ARTIFACTS["pair_filter"], PAIR_FILTER_COLUMNS)
    assert len(errors) == report["num_verified_pairs"]
    assert report["pair_filter_checked"] == {"gyro": len(errors), "tilt": 0, "none": 0}  # all within 5 s
    assert not report["pair_filter_disabled"]
    assert report["pair_filter_removed"] == errors["removed"].sum() <= 0.3 * len(errors)
    removed = errors["removed"]
    assert (errors.loc[removed, "err_deg"] > 10).all() and (errors.loc[~removed, "err_deg"] <= 10).all()
    # image and gyro rotations agree: also checks cam2_from_cam1 against our cam_j_from_cam_i
    assert errors["err_deg"].median() < 3
    db = pycolmap.Database.open(sfm_run / ARTIFACTS["database"])
    num_pairs = len(db.read_two_view_geometries()[0])
    db.close()
    assert num_pairs == len(errors) - report["pair_filter_removed"]


@pytest.fixture(scope="module")
def no_imu(tiny_capture, output_root, sfm_run, tmp_path_factory):
    """sfm without gravity priors and pair filter on a copy of the shared sequences; returns its report."""
    root = tmp_path_factory.mktemp("no_imu")
    shutil.copytree(output_root / "room2_tiny" / "sequences", root / "room2_tiny" / "sequences")
    overrides = ["sfm.priors.use_gravity_prior=false", "sfm.priors.imu_pair_filter.enabled=false"]
    assert _run(tiny_capture, root, "--only", "sfm", *[a for o in overrides for a in ("--set", o)]) == 0
    (out,) = (root / "room2_tiny" / "runs").glob("*/03_sfm")
    return out, read_json(out / ARTIFACTS["report"])


def test_without_imu(no_imu, report):
    out, no_imu_report = no_imu
    assert not no_imu_report["gravity_prior_used"] and no_imu_report["num_gravity_priors"] == 0
    assert no_imu_report["pair_filter_removed"] == 0
    errors = read_csv(out / ARTIFACTS["pair_filter"], PAIR_FILTER_COLUMNS)
    assert len(errors) == no_imu_report["num_verified_pairs"] and not errors["removed"].any()
    assert no_imu_report["registered_ratio"] >= 0.9


def test_refinement(sfm_run, report):
    assert report["raw"]["mean_reproj_px"] > report["mean_reproj_px"]
    assert report["num_registered"] >= report["raw"]["num_registered"] - 1
    steps = [step["name"] for step in report["refine"]["steps"]]
    assert steps == [
        "input",
        "retriangulate",
        "filter_retriangulated",
        "bundle_adjustment",
        "filter",
        "final_bundle_adjustment",
    ]
    assert report["refine"]["dropped_images"] == []
    assert report["refine"]["steps"][-1]["mean_reproj_px"] == report["mean_reproj_px"]
    assert report["refine"]["sanity_warnings"] == [] and read_manifest(sfm_run).warnings == []
    assert (sfm_run / ARTIFACTS["model_txt"] / "cameras.txt").is_file()
    # every pair is matched on the tiny clip, so no loop detection; the global mapper registers everything
    assert report["loop_detection"] == {"enabled": False, "vocab_tree": None}
    assert report["mapper_used"] == "global" and len(report["mapper_attempts"]) == 1
    assert report["mapper_attempts"][0]["error"] is None


@pytest.fixture(scope="module")
def fallback(tiny_capture, output_root, sfm_run, tmp_path_factory):
    """sfm on a copy of the shared sequences with an unreachable registration target (every mapper runs) and
    sequential matching (loop detection runs when the vocab tree is available); returns its dir and report."""
    root = tmp_path_factory.mktemp("fallback")
    shutil.copytree(output_root / "room2_tiny" / "sequences", root / "room2_tiny" / "sequences")
    overrides = ["sfm.mapper.min_registered_ratio=1.01", "sfm.matching.strategy=sequential"]
    assert _run(tiny_capture, root, "--only", "sfm", *[a for o in overrides for a in ("--set", o)]) == 0
    (out,) = (root / "room2_tiny" / "runs").glob("*/03_sfm")
    return out, read_json(out / ARTIFACTS["report"])


def test_mapper_fallback(fallback):
    out, report = fallback
    attempts = report["mapper_attempts"]
    assert [a["mapper"] for a in attempts] == ["global", "global_no_gravity", "incremental"]
    best = max(attempts, key=lambda a: (a["num_registered"], -(a["mean_reproj_px"] or 0)))
    assert report["mapper_used"] == best["mapper"]
    assert report["raw"]["num_registered"] == best["num_registered"]
    assert report["registered_ratio"] >= 0.9
    assert any(w.startswith("mapper fallback:") for w in read_manifest(out).warnings)
    assert not (out / "sparse" / "_mapping").exists()


def test_loop_detection(fallback):
    out, report = fallback
    assert report["matching_strategy"] == "sequential"
    warned = "no vocab tree: loop detection is off" in read_manifest(out).warnings
    assert report["loop_detection"]["enabled"] != warned
    if report["loop_detection"]["enabled"]:
        assert Path(report["loop_detection"]["vocab_tree"]).is_file()
