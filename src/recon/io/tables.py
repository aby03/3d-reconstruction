"""CSV, JSON and NPZ files. Every write goes to a temporary file first and is then moved into place."""

import json
import math
import os
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _write_atomic(path: Path, write: Callable[[Path], None]) -> Path:
    """Call `write` on a temporary sibling of `path`, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _missing(columns: Sequence[str], present: Sequence[str]) -> list[str]:
    return [c for c in columns if c not in present]


def write_csv(df: pd.DataFrame, path: Path | str, columns: Sequence[str]) -> Path:
    """Write exactly `columns`, in that order."""
    path = Path(path)
    if missing := _missing(columns, df.columns):
        raise KeyError(f"{path.name}: missing columns {missing}")
    return _write_atomic(path, lambda tmp: df[list(columns)].to_csv(tmp, index=False))


def read_csv(path: Path | str, columns: Sequence[str]) -> pd.DataFrame:
    """Read a CSV written by `write_csv`, checking that `columns` are present; returns them in that order."""
    df = pd.read_csv(path)
    if missing := _missing(columns, df.columns):
        raise KeyError(f"{Path(path).name}: missing columns {missing}")
    return df[list(columns)]


def to_jsonable(obj: Any) -> Any:
    """Plain JSON types: numpy values become Python ones, Paths strings, NaN/inf None, dict keys strings."""
    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, Enum) else k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(obj: Any, path: Path | str) -> Path:
    text = json.dumps(to_jsonable(obj), indent=2, allow_nan=False) + "\n"
    return _write_atomic(Path(path), lambda tmp: tmp.write_text(text))


def write_bytes(data: bytes, path: Path | str) -> Path:
    return _write_atomic(Path(path), lambda tmp: tmp.write_bytes(data))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write_npz(path: Path | str, compress: bool = False, **arrays: np.ndarray) -> Path:
    def write(tmp: Path) -> None:
        with open(tmp, "wb") as f:  # a file object stops np.savez from appending ".npz"
            (np.savez_compressed if compress else np.savez)(f, **arrays)

    return _write_atomic(Path(path), write)
