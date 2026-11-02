import shutil

import numpy as np
import pycolmap
import pytest
import trimesh

from recon.io.tables import read_json
from recon.pipeline.cli import main
from recon.pipeline.device import probe_device
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s04_align import ARTIFACTS as ALIGN_ARTIFACTS
from recon.pipeline.stages.s05_dense import ARTIFACTS
from recon.sfm.quality import camera_centres

REPORT_KEYS = {
    "input_model",
    "num_images",
    "units",
    "undistorted",
    "backend",
    "cuda_device",
    "openmvs_version",
    "densify",
    "num_dense_points",
    "sparse_bbox",
    "diag",
    "dense",
    "mesh_raw",
    "tools",
    "peak_vram_mb",
    "disk_mb",
    "timings",
    "warnings",
}
DENSIFY_KEYS = {
    "resolution_level",
    "effective_level",
    "effective_long_side",
    "number_views",
    "number_views_fuse",
    "iters",
    "geometric_iters",
    "crop_to_roi",
}
TOOLS = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh"]


@pytest.fixture(scope="module")
def report(dense_run):
    return read_json(dense_run / ARTIFACTS["report"])


@pytest.fixture(scope="module")
def aligned(align_run):
    return pycolmap.Reconstruction(align_run / ALIGN_ARTIFACTS["model"])


@pytest.fixture(scope="module")
def sparse_box(aligned):
    """Axis-aligned box of every aligned sparse point and camera centre."""
    points = np.array([p.xyz for p in aligned.points3D.values()])
    everything = np.concatenate([points, list(camera_centres(aligned).values())])
    return everything.min(axis=0), everything.max(axis=0)


def _inside(xyz, box):
    return ((xyz >= box[0]) & (xyz <= box[1])).all(axis=1).mean()


def test_manifest(dense_run):
    manifest = read_manifest(dense_run)
    assert manifest.status == "success"
    assert manifest.artifacts == ARTIFACTS
    for rel in manifest.artifacts.values():
        assert (dense_run / rel).exists()
    assert list(manifest.upstream) == ["align"]
    backend = "cuda" if probe_device().has("dense") else "cpu"
    assert manifest.backends == {"densify": backend, "mesh": backend}
    assert manifest.tools["OpenMVS"] and manifest.tools["pycolmap"] == pycolmap.__version__
    assert manifest.warnings == []
    assert manifest.metrics["backend"] == backend and manifest.metrics["num_dense_points"] > 0


def test_undistortion(dense_run, aligned, report):
    rec = pycolmap.Reconstruction(dense_run / ARTIFACTS["undistorted_model"])
    assert {cam.model.name for cam in rec.cameras.values()} == {"PINHOLE"}
    names = {image.name for image in rec.images.values()}
    assert names == {image.name for image in aligned.images.values()}
    for name in names:
        assert (dense_run / ARTIFACTS["undistorted_images"] / name).is_file()
    # undistortion changes the cameras, never the poses
    before, after = camera_centres(aligned), camera_centres(rec)
    assert list(before) == list(after)
    assert np.abs(np.array(list(before.values())) - np.array(list(after.values()))).max() <= 1e-9 * report["diag"]
    assert all(c["model"] == "PINHOLE" for c in report["undistorted"].values())


def test_paths(dense_run):
    for key in ("scene", "scene_dense"):
        data = (dense_run / ARTIFACTS[key]).read_bytes()
        assert b"../undistorted/images/" in data
        assert str(dense_run.parent).encode() not in data


def test_report(report, aligned):
    assert set(report) == REPORT_KEYS and set(report["densify"]) == DENSIFY_KEYS
    assert report["input_model"] == "04_alignment/sparse_aligned" and report["units"] == "arbitrary"
    assert report["num_images"] == aligned.num_reg_images()
    assert list(report["tools"]) == TOOLS and all(t["returncode"] == 0 for t in report["tools"].values())
    assert report["cuda_device"] == (0 if report["backend"] == "cuda" else -2)
    assert set(report["timings"]) == {"load", "undistort", "import", "densify", "mesh", "outputs"}
    assert report["densify"]["effective_long_side"] == 928  # 1857 px >> 1, the min_resolution clamp keeps level 1
    assert report["num_dense_points"] > 50_000  # measured 186k
    mesh = report["mesh_raw"]
    assert mesh["faces"] > 50_000 and mesh["largest_component_frac"] > 0.9  # measured 298k, 1.0


def test_world_frame(report, sparse_box, aligned, dense_run):
    dense, diag = report["dense"], report["diag"]
    assert dense["inside_sparse_bbox_frac"] >= 0.99 and report["mesh_raw"]["inside_sparse_bbox_frac"] >= 0.99
    assert dense["floor_band_frac"] > 0.05 and abs(dense["floor_band_median_z"]) < 0.005 * diag
    # the walls reach up: OpenMVS's ROI crop would cut them below the sparse points
    points = np.array([p.xyz for p in aligned.points3D.values()])
    assert dense["bbox"]["p99"][2] >= 0.9 * np.percentile(points[:, 2], 98)
    # recomputed from the files, with the same box (all sparse points and camera centres)
    assert np.allclose([report["sparse_bbox"]["min"], report["sparse_bbox"]["max"]], sparse_box)
    mesh = trimesh.load(dense_run / ARTIFACTS["mesh_raw"], process=False, force="mesh")
    assert np.isclose(_inside(np.asarray(mesh.vertices), sparse_box), report["mesh_raw"]["inside_sparse_bbox_frac"])
    assert len(mesh.faces) == report["mesh_raw"]["faces"]


def test_dense_points(dense_run, report, sparse_box):
    cloud = trimesh.load(dense_run / ARTIFACTS["dense_points"])
    assert isinstance(cloud, trimesh.PointCloud)
    assert len(cloud.vertices) == report["num_dense_points"]
    assert cloud.colors.shape == (len(cloud.vertices), 4) and cloud.colors[:, :3].std() > 0
    data = cloud.metadata["_ply_raw"]["vertex"]["data"]
    assert data.dtype.names == ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")
    normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=1)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
    assert np.isclose(_inside(np.asarray(cloud.vertices), sparse_box), report["dense"]["inside_sparse_bbox_frac"])


def test_tool_logs(dense_run, report):
    logs = dense_run.parent / "logs" / "dense"
    for tool in TOOLS:
        assert (dense_run.parent / report["tools"][tool]["log"]) == logs / f"{tool}.log"
        assert (logs / f"{tool}.log").read_text().startswith(f"\n$ cd {dense_run / 'openmvs'} && ")
        assert list(logs.glob(f"{tool}-*.log")), f"OpenMVS's own {tool} log was not moved to {logs}"
    # OpenMVS writes its logs into the working folder; they are moved out, and the depth maps are removed
    assert not list((dense_run / "openmvs").glob("*.log")) and not list((dense_run / "openmvs").glob("*.dmap"))


def _dense(capture, root, *args):
    return main(["dense", "--input", str(capture), "--output-root", str(root), *args])


def test_cli_skips(tiny_capture, output_root, dense_run):
    mtime = (dense_run / "stage_manifest.json").stat().st_mtime_ns
    assert _dense(tiny_capture, output_root, "--resume", dense_run.parent.name) == 0
    assert (dense_run / "stage_manifest.json").stat().st_mtime_ns == mtime


def test_cli_needs_align(tiny_capture, output_root, sfm_run, tmp_path, capsys):
    """A run with sfm but no align: dense refuses to start."""
    run_dir = sfm_run.parent
    shutil.copytree(output_root / "room2_tiny" / "sequences", tmp_path / "room2_tiny" / "sequences")
    copy = tmp_path / "room2_tiny" / "runs" / run_dir.name
    copy.mkdir(parents=True)
    shutil.copytree(sfm_run, copy / sfm_run.name, symlinks=True)
    shutil.copy(run_dir / "run_manifest.json", copy)
    assert _dense(tiny_capture, tmp_path, "--resume", run_dir.name) == 1
    assert "run `recon align` first" in capsys.readouterr().out
