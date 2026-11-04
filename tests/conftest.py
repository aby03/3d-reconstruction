from pathlib import Path

import pytest

from recon.pipeline.config import load_config, resolve_hardware
from recon.pipeline.context import RunContext
from recon.pipeline.runner import open_run, run_stages, select_stages

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_CAPTURE = REPO_ROOT / "data" / "room2_tiny"
SCENE = "room2_tiny"


@pytest.fixture(scope="session")
def tiny_capture() -> Path:
    """A ~5 s Open Camera Sensors capture. Not committed; create it once with `pixi run make-test-clip`."""
    if not TINY_CAPTURE.is_dir():
        pytest.skip(f"{TINY_CAPTURE} missing, create it with `pixi run make-test-clip`")
    return TINY_CAPTURE


@pytest.fixture(scope="session")
def output_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("outputs")


@pytest.fixture(scope="session")
def pipeline_run(tiny_capture, output_root) -> RunContext:
    """ingest → keyframes on the tiny clip through the runner, once per session; later steps read its stage dirs.
    The config is the CLI's: the defaults with the hardware preset picked for this machine."""
    overrides = [f"run.output_root={output_root}", "keyframes.stills=false"]  # the later steps are checked on one camera
    hardware = resolve_hardware("auto", overrides=overrides)
    ctx = open_run(tiny_capture, load_config(overrides=overrides, hardware=hardware or "none"), SCENE, None, hardware)
    result = run_stages(ctx, select_stages(to="keyframes"))
    assert result.status == "success", result.error
    return ctx


@pytest.fixture(scope="session")
def sfm_run(pipeline_run) -> Path:
    """sfm on the keyframes of `pipeline_run`, once per session; returns its stage dir."""
    from recon.pipeline.stages.s03_sfm import SfmStage

    result = run_stages(pipeline_run, [SfmStage])
    assert result.status == "success", result.error
    return pipeline_run.stage_dir("sfm")


@pytest.fixture(scope="session")
def align_run(pipeline_run, sfm_run) -> Path:
    """align on the model of `sfm_run`, once per session; returns its stage dir."""
    from recon.pipeline.stages.s04_align import AlignStage

    result = run_stages(pipeline_run, [AlignStage])
    assert result.status == "success", result.error
    return pipeline_run.stage_dir("align")


@pytest.fixture(scope="session")
def dense_run(pipeline_run, align_run) -> Path:
    """dense on the model of `align_run` with the default config, once per session; returns its stage dir."""
    from recon.pipeline.stages.s05_dense import DenseStage

    result = run_stages(pipeline_run, [DenseStage])
    assert result.status == "success", result.error
    return pipeline_run.stage_dir("dense")


@pytest.fixture(scope="session")
def mesh_post_run(pipeline_run, dense_run) -> Path:
    """mesh_post on the mesh of `dense_run` with the default config, once per session; returns its stage dir."""
    from recon.pipeline.stages.s06_mesh_post import MeshPostStage

    result = run_stages(pipeline_run, [MeshPostStage])
    assert result.status == "success", result.error
    return pipeline_run.stage_dir("mesh_post")


@pytest.fixture(scope="session")
def report_run(pipeline_run, export_run) -> Path:
    """report of the run of `export_run`, once per session; returns its stage dir."""
    from recon.pipeline.stages.s08_report import ReportStage

    result = run_stages(pipeline_run, [ReportStage])
    assert result.status == "success", result.error
    return pipeline_run.stage_dir("report")


@pytest.fixture(scope="session")
def export_run(pipeline_run, mesh_post_run) -> Path:
    """export of the textured mesh of `mesh_post_run`, once per session; returns its stage dir."""
    from recon.pipeline.stages.s07_export import ExportStage

    result = run_stages(pipeline_run, [ExportStage])
    assert result.status == "success", result.error
    return pipeline_run.stage_dir("export")
