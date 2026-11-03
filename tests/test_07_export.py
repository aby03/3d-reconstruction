import hashlib
import struct

import numpy as np
import pycolmap
import pytest
import trimesh

from recon.core.frames import R_GW
from recon.export.gltf import read_glb, scene_bounds
from recon.io.tables import read_json
from recon.mesh.texturing import read_mtl_maps
from recon.pipeline.cli import main
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s04_align import ARTIFACTS as ALIGN_ARTIFACTS
from recon.pipeline.stages.s06_mesh_post import ARTIFACTS as MESH_ARTIFACTS
from recon.pipeline.stages.s07_export import ARTIFACTS

METADATA_KEYS = {
    "scene",
    "run_id",
    "sequence_ids",
    "pipeline_version",
    "units",
    "world",
    "files",
    "bbox_world",
    "num_cameras",
    "num_planes",
}


@pytest.fixture(scope="module")
def mesh_report(mesh_post_run):
    return read_json(mesh_post_run / MESH_ARTIFACTS["report"])


@pytest.fixture(scope="module")
def source(mesh_post_run):
    """The textured OBJ the export starts from, in W."""
    return trimesh.load(mesh_post_run / MESH_ARTIFACTS["textured_fixed_obj"], force="scene", process=False)


def test_manifest(export_run):
    manifest = read_manifest(export_run)
    assert manifest.status == "success" and manifest.artifacts == ARTIFACTS and manifest.warnings == []
    for rel in ARTIFACTS.values():
        assert (export_run / rel).exists()
    assert list(manifest.upstream) == ["mesh_post"]
    assert manifest.backends == {} and manifest.tools == {"trimesh": trimesh.__version__}
    assert manifest.metrics["texture_count"] >= 1 and manifest.metrics["glb_mb"] > 0


def test_r_gw():
    assert np.allclose(R_GW @ [1, 2, 3], [1, 3, -2])  # x_G = (x_W, z_W, -y_W)
    assert np.allclose(R_GW @ R_GW.T, np.eye(3)) and np.isclose(np.linalg.det(R_GW), 1.0)


def test_glb(export_run, mesh_report, source, mesh_post_run):
    data = (export_run / ARTIFACTS["glb"]).read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    json_len = struct.unpack_from("<I", data, 12)[0]
    bin_len = struct.unpack_from("<I", data, 20 + json_len)[0]
    assert 12 + 8 + json_len + 8 + bin_len == len(data)

    tree, binary = read_glb(data)
    assert tree["images"] and all("bufferView" in i and "uri" not in i for i in tree["images"])
    assert all("uri" not in b for b in tree["buffers"])
    assert {i["mimeType"] for i in tree["images"]} == {"image/jpeg"}
    for material in tree["materials"]:
        pbr = material["pbrMetallicRoughness"]
        assert material["doubleSided"] is True and material["alphaMode"] == "OPAQUE"
        assert "baseColorTexture" in pbr and pbr["metallicFactor"] == 0 and pbr["roughnessFactor"] == 1
    # the atlases are embedded byte for byte
    maps = read_mtl_maps(mesh_post_run / MESH_ARTIFACTS["textured_fixed_mtl"])
    atlases = {(mesh_post_run / MESH_ARTIFACTS["textured_fixed"] / name).read_bytes() for name in maps.values()}
    views = [tree["bufferViews"][i["bufferView"]] for i in tree["images"]]
    assert {binary[v["byteOffset"] : v["byteOffset"] + v["byteLength"]] for v in views} == atlases

    glb = trimesh.load(export_run / ARTIFACTS["glb"], force="scene")
    assert sum(len(g.faces) for g in glb.geometry.values()) == mesh_report["final"]["faces"]
    bounds_w, bounds_g = scene_bounds(source), scene_bounds(glb)
    tol = 1e-4 * mesh_report["diag"]
    expected = np.stack([R_GW @ bounds_w[0], R_GW @ bounds_w[1]])
    assert np.abs(bounds_g - np.sort(expected, axis=0)).max() < tol
    # Y up: the GLB's height is the W z extent, floor at y = 0
    assert np.allclose(bounds_g[:, 1], bounds_w[:, 2], atol=tol)


def test_obj(export_run, mesh_report, source):
    obj = export_run / ARTIFACTS["obj"]
    assert obj.read_text().startswith("# world W: +Z up, units arbitrary\nmtllib model.mtl\n")
    mtl = (export_run / ARTIFACTS["mtl"]).read_text()
    assert "Tr 0.000000" in mtl and "d 1.0" in mtl and "Tr 1" not in mtl
    maps = read_mtl_maps(export_run / ARTIFACTS["mtl"])
    assert maps and all(name.startswith("textures/model_") for name in maps.values())
    textures = {p.name for p in (export_run / ARTIFACTS["textures"]).iterdir()}
    assert textures == {name.split("/")[1] for name in maps.values()}
    copy = trimesh.load(obj, force="scene", process=False)
    assert sum(len(g.faces) for g in copy.geometry.values()) == mesh_report["final"]["faces"]
    assert np.abs(scene_bounds(copy) - scene_bounds(source)).max() < 1e-4 * mesh_report["diag"]
    assert all(g.visual.material.image is not None for g in copy.geometry.values())


def test_metadata(export_run, mesh_report, align_run):
    metadata = read_json(export_run / ARTIFACTS["metadata"])
    assert set(metadata) == METADATA_KEYS
    assert metadata["scene"] == "room2_tiny" and metadata["run_id"] == export_run.parent.name
    assert metadata["sequence_ids"] == ["room2_tiny_20261021_211400"] and metadata["units"] == "arbitrary"
    assert metadata["world"] == {"up": "+Z", "gravity": [0.0, 0.0, -1.0], "floor_z": 0.0}
    assert metadata["pipeline_version"] is None or len(metadata["pipeline_version"]) >= 7
    aligned = pycolmap.Reconstruction(align_run / ALIGN_ARTIFACTS["model"])
    assert metadata["num_cameras"] == aligned.num_reg_images()
    assert metadata["num_planes"] == mesh_report["planes"]["by_label"]
    assert set(metadata["files"]) == {"model.glb", "obj/model.obj"}
    assert metadata["files"]["model.glb"]["up"] == "+Y" and metadata["files"]["obj/model.obj"]["up"] == "+Z"
    for rel, entry in metadata["files"].items():
        data = (export_run / rel).read_bytes()
        assert entry["bytes"] == len(data) and entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["faces"] == mesh_report["final"]["faces"] and entry["textures"]
    bbox = np.array(metadata["bbox_world"])
    assert bbox.shape == (2, 3) and (bbox[1] > bbox[0]).all() and bbox[0, 2] < 0.01 * mesh_report["diag"]


def test_cli_skips(tiny_capture, output_root, export_run):
    mtime = (export_run / "stage_manifest.json").stat().st_mtime_ns
    args = ["--input", str(tiny_capture), "--output-root", str(output_root), "--resume", export_run.parent.name]
    assert main(["export", *args]) == 0
    assert main(["run", *args]) == 0
    assert (export_run / "stage_manifest.json").stat().st_mtime_ns == mtime
    stages = read_json(export_run.parent / "run_manifest.json")["stages"]
    assert [s["stage"] for s in stages][-3:] == ["dense", "mesh_post", "export"]
    assert {s["status"] for s in stages} == {"skipped"}
