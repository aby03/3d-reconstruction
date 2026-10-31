"""Output directory naming, stage manifests and logging setup."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from recon.io.tables import read_json, write_json

MANIFEST_NAME = "stage_manifest.json"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def sequence_dir(output_root: Path | str, scene: str, sequence_id: str) -> Path:
    """`<output_root>/<scene>/sequences/<sequence_id>/`, home of the per-sequence stages."""
    return Path(output_root) / scene / "sequences" / sequence_id


def stage_dir(parent: Path, dir_name: str) -> Path:
    return parent / dir_name


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class StageManifest:
    stage: str
    status: str  # "success" | "failed"
    started: str
    finished: str
    duration_s: float
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path relative to the stage dir
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def write_manifest(manifest: StageManifest, stage_path: Path) -> Path:
    return write_json(asdict(manifest), stage_path / MANIFEST_NAME)


def read_manifest(stage_path: Path) -> StageManifest | None:
    path = stage_path / MANIFEST_NAME
    return StageManifest(**read_json(path)) if path.is_file() else None


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
