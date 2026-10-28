from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_CAPTURE = REPO_ROOT / "data" / "room2_tiny"


@pytest.fixture(scope="session")
def tiny_capture() -> Path:
    """A ~5 s Open Camera Sensors capture. Not committed; create it once with `pixi run make-test-clip`."""
    if not TINY_CAPTURE.is_dir():
        pytest.skip(f"{TINY_CAPTURE} missing, create it with `pixi run make-test-clip`")
    return TINY_CAPTURE
