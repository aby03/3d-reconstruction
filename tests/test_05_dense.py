import shutil

import numpy as np
import pycolmap
import pytest
import trimesh

from recon.dense.openmvs import (
    DENSE_PLY,
    SCENE_DENSE,
    OpenMvsOptions,
    densify,
    effective_long_side,
    is_oom,
    oom_retry_options,
)
from recon.io.tables import read_json
from recon.io.tools import ToolError, ToolResult
from recon.pipeline.cli import main
from recon.pipeline.config import PipelineConfig, load_config, select_hardware
from recon.pipeline.device import DeviceInfo, probe_device
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s04_align import ARTIFACTS as ALIGN_ARTIFACTS
from recon.pipeline.stages.s05_dense import ARTIFACTS, FILTERED_PLY, OPTIONAL_ARTIFACTS
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
    "num_dense_points_raw",
    "filter",
    "mesh_crop",
    "sparse_bbox",
    "diag",
    "dense",
    "mesh_raw",
    "refine",
    "preview",
    "oom_retries",
    "hardware",
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
TOOLS = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]
OOM_LINE = "CUDA error: cuMemAlloc(&pData, size) (CUDA_ERROR_OUT_OF_MEMORY (code 2) - out of memory)"


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
    assert manifest.artifacts == {**ARTIFACTS, **OPTIONAL_ARTIFACTS}
    for rel in manifest.artifacts.values():
        assert (dense_run / rel).exists()
    assert list(manifest.upstream) == ["align"]
    backend = "cuda" if probe_device().has("dense") else "cpu"
    assert manifest.backends == {"densify": backend, "mesh": backend, "refine": "cpu", "texture": backend}
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
    assert set(report["timings"]) == {
        "load", "undistort", "import", "densify", "filter", "mesh", "refine", "preview", "outputs"
    }
    assert report["oom_retries"] == [] and all(t["calls"] == 1 for t in report["tools"].values())
    # ~1857 px >> 1 = 928 (the undistorted size varies a little with the SfM intrinsics): level 2 is clamped to 1
    long_side = max(max(c["width"], c["height"]) for c in report["undistorted"].values())
    assert report["densify"]["effective_long_side"] == long_side >> 1 and report["densify"]["effective_level"] == 1
    assert report["num_dense_points"] > 50_000  # measured 186k
    mesh = report["mesh_raw"]
    assert mesh["faces"] > 50_000 and mesh["largest_component_frac"] > 0.9  # measured 298k, 1.0


def test_filter(report, dense_run, sparse_box):
    f = report["filter"]
    assert f["enabled"] and f["input"] == report["num_dense_points_raw"]
    removed = f["outside_box"] + f["statistical"] + f["radius"]
    assert 0 < removed < 0.05 * f["input"]  # measured 3.6 k of 187 k: a few stray points, not the room
    assert f["output"] == f["input"] - removed == report["num_dense_points"]
    lo, hi = np.array(f["box"]["min"]), np.array(f["box"]["max"])
    cloud = np.asarray(trimesh.load(dense_run / ARTIFACTS["dense_points"]).vertices)
    assert (cloud >= lo - 1e-3).all() and (cloud <= hi + 1e-3).all()
    # ReconstructMesh read the filtered cloud, and RefineMesh got the mesh cut to the same box
    assert (dense_run / "openmvs" / FILTERED_PLY).is_file()
    crop = report["mesh_crop"]
    assert crop["faces_after"] == report["mesh_raw"]["faces"] <= crop["faces_before"]
    mesh = trimesh.load(dense_run / ARTIFACTS["mesh_raw"], process=False, force="mesh")
    assert (mesh.vertices >= lo - 1e-6).all() and (mesh.vertices <= hi + 1e-6).all()


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


def test_refine(dense_run, report, sparse_box):
    refine = report["refine"]
    assert refine["enabled"] and refine["ran"] and not refine["timed_out"]
    assert refine["cuda_device"] == -2  # dense.refine.use_cuda: false
    assert 20_000 < refine["faces"] < report["mesh_raw"]["faces"]  # measured 98.7k: RefineMesh decimates first
    assert refine["largest_component_frac"] > 0.9 and refine["inside_sparse_bbox_frac"] >= 0.99  # 0.997, 0.993
    mesh = trimesh.load(dense_run / ARTIFACTS["mesh_refined"], process=False, force="mesh")
    assert len(mesh.faces) == refine["faces"]
    assert np.isclose(_inside(np.asarray(mesh.vertices), sparse_box), refine["inside_sparse_bbox_frac"])
    assert (dense_run / ARTIFACTS["mesh_refined"]).read_bytes() == (
        dense_run / OPTIONAL_ARTIFACTS["mesh_refine_openmvs"]
    ).read_bytes()


def test_preview(dense_run, report):
    preview = dense_run / "preview"
    assert {p.suffix for p in preview.iterdir()} == {".obj", ".mtl", ".jpg"}
    assert report["preview"]["faces"] == report["refine"]["faces"] and report["preview"]["texture_coords"] > 0
    assert report["preview"]["textures"] == 1 and report["preview"]["atlas_size"] == [2048, 2048]
    assert "map_Kd mesh_material_00_map_Kd.jpg" in (preview / "mesh.mtl").read_text()


def test_hardware_presets(dense_run, report):
    cpu = load_config(hardware="cpu")
    assert cpu.device.preference == "cpu" and cpu.dense.openmvs.resolution_level == 2
    assert load_config(hardware="none") == PipelineConfig() == load_config(hardware="cuda_6gb")
    assert select_hardware(DeviceInfo()) == "cpu" and select_hardware(probe_device(), "cpu") == "cpu"
    assert select_hardware(DeviceInfo(cuda_available=True, openmvs_cuda=True, vram_mb=24576)) == "cuda_large"
    expected = select_hardware(probe_device())
    if probe_device().has("dense") and 4096 <= probe_device().vram_mb < 10240:
        assert expected == "cuda_6gb"
    run_dir = dense_run.parent
    preset = f"configs/hardware/{expected}.yaml" if expected else "none"
    assert (run_dir / "config.resolved.yaml").read_text().startswith(f"# hardware preset: {preset}\n")
    assert read_json(run_dir / "run_manifest.json")["hardware"] == expected == report["hardware"]


def test_oom_pattern(dense_run):
    assert is_oom(f"12:00:00 [App     ] {OOM_LINE}\n")
    assert not is_oom((dense_run.parent / "logs" / "dense" / "DensifyPointCloud.log").read_text())
    options = OpenMvsOptions()
    retry = oom_retry_options(options, 1857)
    sizes = [effective_long_side(1857, o.resolution_level, o.min_resolution, o.max_resolution) for o in (options, retry)]
    assert sizes == [(928, 1), (464, 2)] and retry.number_views == options.number_views - 2


class _OomRunner:
    """Stands in for a ToolRunner: the first `ooms` calls fail with the CUDA OOM line, then the outputs appear."""

    def __init__(self, ooms):
        self.ooms, self.calls, self.results = ooms, [], []

    def run(self, tool, args, cwd, timeout_s=None):
        self.calls.append(args)
        log = cwd / f"{tool}.log"
        log.write_text("Densifying point-cloud completed: 7 points\n")
        self.results.append(ToolResult(tool, args, 0 if len(self.calls) > self.ooms else 1, 1.0, None, log))
        if len(self.calls) <= self.ooms:
            raise ToolError(tool, 1, OOM_LINE)
        (cwd / SCENE_DENSE).touch()
        (cwd / DENSE_PLY).touch()
        return self.results[-1]


def _device(args):
    return int(args[args.index("--cuda-device") + 1])


def test_oom_retry(tmp_path):
    runner = _OomRunner(ooms=1)
    result = densify(runner, tmp_path, OpenMvsOptions(), 0, 1857)
    assert [_device(a) for a in runner.calls] == [0, 0]
    assert [(r.from_level, r.to_level, r.cpu_fallback) for r in result.oom_retries] == [(1, 2, False)]
    assert (result.effective_level, result.effective_long_side, result.cuda_device) == (2, 464, 0)
    assert len(result.results) == 2 and result.num_points == 7

    runner = _OomRunner(ooms=2)
    result = densify(runner, tmp_path, OpenMvsOptions(), 0, 1857, cpu_fallback=True)
    assert [_device(a) for a in runner.calls] == [0, 0, -2] and result.cuda_device == -2
    assert [r.cpu_fallback for r in result.oom_retries] == [False, True]
    with pytest.raises(ToolError):  # no CPU fallback asked: the second OOM is raised
        densify(_OomRunner(ooms=2), tmp_path, OpenMvsOptions(), 0, 1857)
    with pytest.raises(ToolError):  # on the CPU there is nothing to retry
        densify(_OomRunner(ooms=1), tmp_path, OpenMvsOptions(), -2, 1857, cpu_fallback=True)


@pytest.fixture(scope="module")
def no_refine(tiny_capture, output_root, dense_run, tmp_path_factory):
    """dense with refine and preview off and no hardware preset, on a copy of the shared run; returns its stage dir."""
    root, run_dir = tmp_path_factory.mktemp("no_refine"), dense_run.parent
    shutil.copytree(output_root / "room2_tiny" / "sequences", root / "room2_tiny" / "sequences")
    shutil.copytree(run_dir, root / "room2_tiny" / "runs" / run_dir.name, symlinks=True)
    overrides = ["dense.refine.enabled=false", "dense.preview.enabled=false"]
    args = ["--resume", run_dir.name, "--force", "--hardware", "none", *[a for o in overrides for a in ("--set", o)]]
    assert _dense(tiny_capture, root, *args) == 0
    return root / "room2_tiny" / "runs" / run_dir.name / dense_run.name


def test_refine_disabled(no_refine):
    manifest = read_manifest(no_refine)
    assert manifest.status == "success" and manifest.artifacts == ARTIFACTS and manifest.warnings == []
    assert set(manifest.backends) == {"densify", "mesh"}
    assert (no_refine / ARTIFACTS["mesh_refined"]).read_bytes() == (no_refine / ARTIFACTS["mesh_raw"]).read_bytes()
    assert not (no_refine / "preview").exists()
    assert not (no_refine / OPTIONAL_ARTIFACTS["mesh_refine_openmvs"]).exists()
    report = read_json(no_refine / ARTIFACTS["report"])
    assert report["refine"]["enabled"] is False and report["refine"]["ran"] is False
    assert report["refine"]["faces"] == report["mesh_raw"]["faces"] and report["preview"] == {"enabled": False}
    assert report["hardware"] is None
    assert (no_refine.parent / "config.resolved.yaml").read_text().startswith("# hardware preset: none\n")
