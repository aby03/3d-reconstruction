import numpy as np
import pytest
import trimesh
from conftest import assert_stage_ok, rerun_copy

from recon.dense.openmvs import (
    DENSE_PLY,
    SCENE_DENSE,
    OpenMvsOptions,
    densify,
    effective_long_side,
    is_oom,
    oom_retry_options,
)
from recon.io.tools import ToolError, ToolResult
from recon.pipeline.config import PipelineConfig, load_config, select_hardware
from recon.pipeline.device import DeviceInfo, probe_device

TOOLS = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]
OOM_LINE = "CUDA error: cuMemAlloc(&pData, size) (CUDA_ERROR_OUT_OF_MEMORY (code 2) - out of memory)"


def test_smoke(smoke_run):
    assert_stage_ok(smoke_run.stage_dir("dense"))
    logs = smoke_run.run_dir / "logs" / "dense"
    for tool in TOOLS:
        assert (logs / f"{tool}.log").is_file(), tool


def test_simplerecon_smoke(smoke_run, output_root, tmp_path):
    from recon.dense.simplerecon import SimpleReconError, SimpleReconOptions, check_installed

    try:
        check_installed(SimpleReconOptions())
    except SimpleReconError as e:
        pytest.skip(str(e))
    run_dir = rerun_copy(
        smoke_run,
        output_root,
        tmp_path,
        "--to",
        "dense",
        "--set",
        "keyframes.uniform_fps=4",
        "--set",
        "dense.depth_source=simplerecon",
        "--set",
        "dense.simplerecon.scale=10.0",
    )
    assert_stage_ok(run_dir / "05_dense")


def test_hardware_presets():
    cpu = load_config(hardware="cpu")
    assert cpu.device.preference == "cpu" and cpu.dense.openmvs.resolution_level == 2
    assert cpu.mesh_post == PipelineConfig().mesh_post
    assert load_config(hardware="none") == PipelineConfig() == load_config(hardware="cuda_6gb")
    assert select_hardware(DeviceInfo()) == "cpu" and select_hardware(probe_device(), "cpu") == "cpu"
    assert select_hardware(DeviceInfo(cuda_available=True, openmvs_cuda=True, vram_mb=24576)) == "cuda_large"


def test_oom_pattern():
    assert is_oom(f"12:00:00 [App     ] {OOM_LINE}\n")
    assert not is_oom("Densifying point-cloud completed: 7 points\n")
    options = OpenMvsOptions(resolution_level=1)
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
    return int(args[args.index("--gpu-device") + 1])


def test_oom_retry(tmp_path):
    runner = _OomRunner(ooms=1)
    result = densify(runner, tmp_path, OpenMvsOptions(resolution_level=1), 0, 1857)
    assert [_device(a) for a in runner.calls] == [0, 0]
    assert [(r.from_level, r.to_level, r.cpu_fallback) for r in result.oom_retries] == [(1, 2, False)]
    assert (result.effective_level, result.effective_long_side, result.cuda_device) == (2, 464, 0)
    assert len(result.results) == 2 and result.num_points == 7

    runner = _OomRunner(ooms=2)
    result = densify(runner, tmp_path, OpenMvsOptions(resolution_level=1), 0, 1857, cpu_fallback=True)
    assert [_device(a) for a in runner.calls] == [0, 0, -2] and result.cuda_device == -2
    assert [r.cpu_fallback for r in result.oom_retries] == [False, True]
    with pytest.raises(ToolError):  # no CPU fallback asked: the second OOM is raised
        densify(_OomRunner(ooms=2), tmp_path, OpenMvsOptions(resolution_level=1), 0, 1857)
    with pytest.raises(ToolError):  # on the CPU there is nothing to retry
        densify(_OomRunner(ooms=1), tmp_path, OpenMvsOptions(resolution_level=1), -2, 1857, cpu_fallback=True)


def test_mono_fit_to_mesh():
    """A tilted wall in a ray-cast mesh gives back the scale and shift the disparity was made with."""
    import open3d as o3d

    from recon.dense.mono import MonoOptions, ViewFit, fit_to_mesh, mesh_depth

    K, T = np.array([[100, 0, 70], [0, 100, 39], [0, 0, 1.0]]), np.eye(4)
    wall = trimesh.creation.box(extents=[40, 40, 0.1])
    wall.apply_transform(trimesh.transformations.rotation_matrix(0.6, [0, 1, 0]))
    wall.apply_translation([0, 0, 6])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(wall.vertices), o3d.core.float32), o3d.core.Tensor(np.asarray(wall.faces), o3d.core.uint32)
    )
    z = mesh_depth(scene, K, T, (78, 140))
    assert (z > 0).all()
    disparity = (0.8 / z + 0.2).astype(np.float32)  # 1/z = 1.25 d - 0.25
    fit, correction = fit_to_mesh(disparity, ViewFit(1.0, 0.0, 100, 90, 0.9, 0.01, 0.02, True), z, MonoOptions())
    assert fit.mesh_used and np.isclose(fit.scale, 1.25, atol=1e-3) and np.isclose(fit.shift, -0.25, atol=1e-3)
    assert correction is not None and np.allclose(correction, 1.0, atol=1e-3)


def test_fusion_skips_views_without_depth():
    from recon.dense.fusion import FusionView, fuse

    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    full = FusionView(depth=np.full((48, 64), 2.0, np.float32), K=K, cam_from_world=np.eye(4))
    empty = FusionView(depth=np.zeros((48, 64), np.float32), K=K, cam_from_world=np.eye(4))
    *_, report = fuse([full, empty], diag=10.0, tau=0.05)
    assert report["skipped_views"] == 1
