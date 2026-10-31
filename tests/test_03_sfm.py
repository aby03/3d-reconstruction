import numpy as np
import pycolmap
import pytest

from recon.io.tables import read_csv, read_json
from recon.pipeline.cli import main
from recon.pipeline.device import probe_device
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s02_keyframes import ARTIFACTS as KEYFRAME_ARTIFACTS
from recon.pipeline.stages.s02_keyframes import KEYFRAMES_COLUMNS
from recon.pipeline.stages.s03_sfm import ARTIFACTS

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
    "mapper_used",
    "gravity_prior_used",
    "pair_filter_removed",
    "focal_px",
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
    assert report["mean_reproj_px"] < 1.5
    assert report["median_track_len"] >= 3
    assert report["num_points3D"] > 1000
    assert report["num_models"] >= 1 and report["num_verified_pairs"] > 0
    assert report["mapper_used"] == "global" and report["matching_strategy"] == "exhaustive"
    assert not report["gravity_prior_used"] and report["pair_filter_removed"] == 0
    assert set(report["per_image"]) | set(report["unregistered_images"]) == set(keyframe_names)
    assert set(report["timings"]) == {"features", "matching", "mapping", "outputs"}


def test_model(sfm_run, report, keyframe_names):
    rec = pycolmap.Reconstruction(sfm_run / ARTIFACTS["model"])
    assert rec.num_reg_images() == report["num_registered"]
    assert {image.name for image in rec.images.values()} <= set(keyframe_names)
    assert len(rec.cameras) == 1
    camera = next(iter(rec.cameras.values()))
    assert camera.model.name == "SIMPLE_RADIAL"
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
    assert cam["model"] == "SIMPLE_RADIAL" and (cam["width"], cam["height"]) == (1080, 1920)
    assert cam["fx"] == cam["fy"] == cam["params"][0] == report["focal_px"][key]
    assert np.isclose(cam["cx"], 540) and np.isclose(cam["cy"], 960)


def _run(capture, root, *args):
    return main(["run", "--input", str(capture), "--output-root", str(root), *args])


def test_resume_skips_sfm(tiny_capture, output_root, sfm_run):
    mtime = (sfm_run / "stage_manifest.json").stat().st_mtime_ns
    run_id = sfm_run.parent.name
    assert _run(tiny_capture, output_root, "--resume", run_id) == 0
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
