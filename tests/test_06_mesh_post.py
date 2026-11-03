import importlib.metadata
import shutil

import cv2
import numpy as np
import pytest
import trimesh

from recon.io.tables import read_json
from recon.mesh.decimate import DecimateOptions, decimate
from recon.mesh.texturing import read_mtl_maps, read_obj_uv
from recon.pipeline.cli import main
from recon.pipeline.config import PipelineConfig, load_config
from recon.pipeline.device import probe_device
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s05_dense import ARTIFACTS as DENSE_ARTIFACTS
from recon.pipeline.stages.s06_mesh_post import ARTIFACTS

REPORT_KEYS = {
    "input",
    "input_faces",
    "units",
    "diag",
    "cleanup",
    "decimate",
    "final",
    "texture",
    "oom_retries",
    "tools",
    "timings",
    "warnings",
}


@pytest.fixture(scope="module")
def report(mesh_post_run):
    return read_json(mesh_post_run / ARTIFACTS["report"])


def test_manifest(mesh_post_run):
    manifest = read_manifest(mesh_post_run)
    assert manifest.status == "success" and manifest.artifacts == ARTIFACTS
    for rel in manifest.artifacts.values():
        assert (mesh_post_run / rel).exists()
    assert list(manifest.upstream) == ["dense"]
    backend = "cuda" if probe_device().has("dense") else "cpu"
    assert manifest.backends == {"texture": backend}
    assert manifest.tools["OpenMVS"] and manifest.tools["pymeshlab"] == importlib.metadata.version("pymeshlab")
    assert manifest.warnings == []
    assert manifest.metrics["final_faces"] > 0 and manifest.metrics["atlas_count"] >= 1


def test_report(report, dense_run):
    assert set(report) == REPORT_KEYS and report["warnings"] == [] and report["oom_retries"] == []
    assert report["input"] == "05_dense/mesh_refined.ply" and report["units"] == "arbitrary"
    dense = read_json(dense_run / DENSE_ARTIFACTS["report"])
    assert report["input_faces"] == dense["refine"]["faces"] and report["diag"] == dense["diag"]
    assert set(report["timings"]) == {"load", "cleanup", "decimate", "texture", "stats"}
    assert list(report["tools"]) == ["TextureMesh"] and report["tools"]["TextureMesh"]["returncode"] == 0


def test_cleanup(report, mesh_post_run):
    cleanup = report["cleanup"]
    before, after = cleanup["before"], cleanup["after"]
    assert after["components"] == 1  # measured 3 -> 1
    assert cleanup["components_removed"] == before["components"] - 1
    assert 0 < before["faces"] - after["faces"] < 0.01 * before["faces"]  # 32 faces
    assert after["non_manifold_edges"] == 0 and after["non_manifold_vertices"] == 0
    mesh = trimesh.load(mesh_post_run / ARTIFACTS["mesh_clean"], process=False, force="mesh")
    assert len(mesh.faces) == after["faces"] and len(mesh.vertices) == after["vertices"]


def test_decimate(report, mesh_post_run, tmp_path):
    # tiny (101k faces) is within the 600k budget: the final geometry is the cleaned mesh
    faces = report["cleanup"]["after"]["faces"]
    assert report["decimate"] == {"decimated": False, "faces_in": faces, "faces_out": faces, "budget": 600_000}
    clean, final = mesh_post_run / ARTIFACTS["mesh_clean"], mesh_post_run / ARTIFACTS["mesh_final_geometry"]
    assert final.read_bytes() == clean.read_bytes()
    assert report["final"]["faces"] == report["cleanup"]["after"]["faces"]

    result = decimate(clean, tmp_path / "small.ply", DecimateOptions(texture_face_budget=50_000))
    assert result["decimated"] and abs(result["faces_out"] - 50_000) <= 500
    mesh = trimesh.load(tmp_path / "small.ply", process=False, force="mesh")
    assert len(mesh.faces) == result["faces_out"] and mesh.vertex_normals.shape == mesh.vertices.shape


def test_texture(report, mesh_post_run):
    texture, final = report["texture"], report["final"]
    assert texture["faces"] == final["faces"]  # --close-holes 0: TextureMesh adds no faces
    assert texture["untextured_face_frac"] < 0.02  # measured 0.0088
    assert texture["black_texel_frac"] < 0.1  # 0.04; seam leveling turns ~40% black
    assert texture["atlas_count"] >= 1 and 0 < texture["atlas_empty_frac"] < 1
    assert texture["resolution_level"] == 0
    assert texture["cuda_device"] == (0 if read_manifest(mesh_post_run).backends["texture"] == "cuda" else -2)

    textured = mesh_post_run / ARTIFACTS["textured"]
    mtllib, uv, per_material, faces = read_obj_uv(mesh_post_run / ARTIFACTS["textured_obj"])
    assert mtllib == "mesh.mtl" and faces == final["faces"]
    maps = read_mtl_maps(mesh_post_run / ARTIFACTS["textured_mtl"])
    assert set(per_material) == set(maps) and len(maps) == texture["atlas_count"]
    assert {p.name for p in textured.glob("*.jpg")} == set(maps.values())
    sizes = [list(cv2.imread(str(textured / name)).shape[1::-1]) for name in maps.values()]
    assert texture["atlas_size"] in sizes and max(w * h for w, h in sizes) == np.prod(texture["atlas_size"])
    assert uv.min() >= 0 and uv.max() <= 1
    assert all((idx >= 0).all() for idx in per_material.values())


def test_world_frame(report, dense_run, mesh_post_run):
    final = report["final"]
    assert final["inside_sparse_bbox_frac"] >= 0.99  # measured 0.995
    assert final["components"] == 1 and final["largest_component_frac"] == 1.0
    box = read_json(dense_run / DENSE_ARTIFACTS["report"])["sparse_bbox"]
    mesh = trimesh.load(mesh_post_run / ARTIFACTS["mesh_final_geometry"], process=False, force="mesh")
    inside = ((mesh.vertices >= box["min"]) & (mesh.vertices <= box["max"])).all(axis=1).mean()
    assert np.isclose(inside, final["inside_sparse_bbox_frac"])


def test_tool_logs(mesh_post_run):
    logs = mesh_post_run.parent / "logs" / "mesh_post"
    work_dir = mesh_post_run.parent / "05_dense" / "openmvs"
    assert (logs / "TextureMesh.log").read_text().startswith(f"\n$ cd {work_dir} && ")
    assert list(logs.glob("TextureMesh-*.log"))
    assert not list(work_dir.glob("*.log"))


def test_config():
    assert load_config().mesh_post == PipelineConfig().mesh_post
    assert load_config(hardware="cpu").mesh_post == PipelineConfig().mesh_post


def _mesh_post(capture, root, *args):
    return main(["mesh_post", "--input", str(capture), "--output-root", str(root), *args])


def test_cli_skips(tiny_capture, output_root, mesh_post_run):
    mtime = (mesh_post_run / "stage_manifest.json").stat().st_mtime_ns
    assert _mesh_post(tiny_capture, output_root, "--resume", mesh_post_run.parent.name) == 0
    assert (mesh_post_run / "stage_manifest.json").stat().st_mtime_ns == mtime


def test_cli_needs_dense(tiny_capture, output_root, align_run, tmp_path, capsys):
    """A run with align but no dense: mesh_post refuses to start."""
    run_dir = align_run.parent
    shutil.copytree(output_root / "room2_tiny" / "sequences", tmp_path / "room2_tiny" / "sequences")
    copy = tmp_path / "room2_tiny" / "runs" / run_dir.name
    copy.mkdir(parents=True)
    shutil.copytree(align_run, copy / align_run.name, symlinks=True)
    shutil.copy(run_dir / "run_manifest.json", copy)
    assert _mesh_post(tiny_capture, tmp_path, "--resume", run_dir.name) == 1
    assert "run `recon dense` first" in capsys.readouterr().out
