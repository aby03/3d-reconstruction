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


def file_entry(path: Path, up: str | None = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `files` entry of an exported file: up axis (files in a frame), counts from a mesh's validation `info`,
    size and sha256."""
    entry: dict[str, Any] = {} if up is None else {"up": up}
    if info is not None:
        entry |= {"faces": info["faces"], "vertices": info["vertices"], "textures": info["textures"]}
    return entry | {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_metadata(path: Path, **fields: Any) -> Path:
    return write_json(fields, path)
