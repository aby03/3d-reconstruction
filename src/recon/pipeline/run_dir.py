"""Output directory naming, stage manifests and logging setup."""

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from recon.io.tables import read_json, write_json

MANIFEST_NAME = "stage_manifest.json"
RUN_MANIFEST_NAME = "run_manifest.json"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def scene_dir(output_root: Path | str, scene: str) -> Path:
    return Path(output_root) / scene


def sequence_dir(output_root: Path | str, scene: str, sequence_id: str) -> Path:
    """`<output_root>/<scene>/sequences/<sequence_id>/`, home of the per-sequence stages."""
    return scene_dir(output_root, scene) / "sequences" / sequence_id


def new_run_dir(scene_path: Path) -> Path:
    """Create `<scene>/runs/<YYYYmmdd_HHMMSS[_NN]>/`, the record of one pipeline invocation."""
    runs = scene_path / "runs"
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(100):
        path = runs / (base if n == 0 else f"{base}_{n:02d}")
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"too many runs started at {base} in {runs}")


def stage_dir(parent: Path, dir_name: str) -> Path:
    return parent / dir_name


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class StageManifest:
    stage: str
    status: str  # "success" | "failed" | "interrupted"
    started: str
    finished: str
    duration_s: float
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path relative to the stage dir
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config_hash: str = ""
    upstream: dict[str, str] = field(default_factory=dict)  # upstream stage -> sha1 of its manifest file
    error: str | None = None  # traceback tail of a failed stage


def write_manifest(manifest: StageManifest, stage_path: Path) -> Path:
    return write_json(asdict(manifest), stage_path / MANIFEST_NAME)


def read_manifest(stage_path: Path) -> StageManifest | None:
    path = stage_path / MANIFEST_NAME
    return StageManifest(**read_json(path)) if path.is_file() else None


def manifest_hash(stage_path: Path) -> str | None:
    """sha1 of the stage's manifest file, so a re-run upstream (new times, new results) changes it."""
    path = stage_path / MANIFEST_NAME
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.is_file() else None


def setup_logging(log_file: Path | None = None) -> None:
    """Console at INFO and, if given, `log_file` at DEBUG for the `recon` loggers. Replaces earlier handlers."""
    logger = logging.getLogger("recon")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
