import shutil
from pathlib import Path

import pytest

from recon.pipeline.cli import main
from recon.pipeline.config import load_config, resolve_hardware
from recon.pipeline.context import RunContext
from recon.pipeline.run_dir import StageManifest, read_manifest
from recon.pipeline.runner import open_run, run_stages, select_stages

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CAPTURE = REPO_ROOT / "tests" / "resources" / "hotel1_micro"
SMOKE_CONFIG = Path(__file__).with_name("smoke.yaml")


@pytest.fixture(scope="session")
def smoke_capture() -> Path:
    """A 2.5 s Open Camera Sensors clip, committed under tests/resources (regenerate with `pixi run make-smoke-clip`)."""
    assert SMOKE_CAPTURE.is_dir(), f"{SMOKE_CAPTURE} missing"
    return SMOKE_CAPTURE


@pytest.fixture(scope="session")
def output_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("outputs")


@pytest.fixture(scope="session")
def smoke_run(smoke_capture, output_root) -> RunContext:
    """All 8 stages once, in-process, with tests/smoke.yaml and the hardware preset picked for this machine."""
    files, overrides = [SMOKE_CONFIG], [f"run.output_root={output_root}"]
    hardware = resolve_hardware("auto", files, overrides)
    ctx = open_run(smoke_capture, load_config(files, overrides, hardware=hardware or "none"), None, None, hardware)
    result = run_stages(ctx, select_stages())
    assert result.status == "success", result.error
    return ctx


def assert_stage_ok(stage_dir: Path) -> StageManifest:
    """The stage succeeded and every artifact its manifest lists exists; returns the manifest."""
    manifest = read_manifest(stage_dir)
    assert manifest is not None, f"no manifest in {stage_dir}"
    assert manifest.status == "success", manifest.error
    missing = [rel for rel in manifest.artifacts.values() if not (stage_dir / rel).exists()]
    assert not missing, f"missing artifacts in {stage_dir}: {missing}"
    return manifest


def cli(capture: Path, root: Path, *args: str) -> int:
    """`recon <args>` on the smoke config, e.g. cli(capture, root, "run", "--to", "sfm")."""
    return main([*args, "--input", str(capture), "--output-root", str(root), "--config", str(SMOKE_CONFIG)])


def rerun_copy(ctx: RunContext, output_root: Path, dst: Path, *args: str) -> Path:
    """Copy sequences/ and the run dir (symlinks kept) under dst and resume the run there with `recon run *args`;
    asserts exit 0 and returns the copied run dir."""
    scene = ctx.scene_dir.name
    shutil.copytree(ctx.scene_dir / "sequences", dst / scene / "sequences", symlinks=True)
    run_dir = dst / scene / "runs" / ctx.run_dir.name
    shutil.copytree(ctx.run_dir, run_dir, symlinks=True)
    capture = next(iter(ctx.inputs.values()))
    assert cli(capture, dst, "run", "--resume", run_dir.name, *args) == 0
    return run_dir

