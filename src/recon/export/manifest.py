"""`metadata.json`: what was exported, in which frame and units, with checksums."""

import hashlib
from pathlib import Path
from typing import Any

from recon.io.tables import write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path, up: str, info: dict[str, Any]) -> dict[str, Any]:
    """The `files` entry of an exported file: up axis, counts from its validation `info`, size and sha256."""
    return {
        "up": up,
        "faces": info["faces"],
        "vertices": info["vertices"],
        "textures": info["textures"],
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_metadata(path: Path, **fields: Any) -> Path:
    return write_json(fields, path)
