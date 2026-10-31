from pathlib import Path

import pytest

from recon.capture.open_camera_sensors import discover_files
from recon.pipeline.run_dir import sequence_dir
from recon.pipeline.stages.s01_ingest import DIR_NAME, run_ingest

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
def ingest(tiny_capture, output_root):
    """Ingest of the tiny clip, run once per session: (manifest, `01_ingest/` dir)."""
    seq_dir = sequence_dir(output_root, SCENE, discover_files(tiny_capture).sequence_id)
    manifest = run_ingest(tiny_capture, seq_dir)
    return manifest, seq_dir / DIR_NAME
