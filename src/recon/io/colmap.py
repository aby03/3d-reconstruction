"""COLMAP helpers on top of pycolmap: symlinked image roots, image names, camera intrinsics, model files, Sim3,
COLMAP's log output."""

import contextlib
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pycolmap

from recon.core.geometry import Sim3

# (fx, fy, cx, cy, k1, k2, p1, p2) positions in `Camera.params`, None where the model lacks the parameter
_PARAM_INDEX = {
    "SIMPLE_PINHOLE": (0, 0, 1, 2, None, None, None, None),
    "PINHOLE": (0, 1, 2, 3, None, None, None, None),
    "SIMPLE_RADIAL": (0, 0, 1, 2, 3, None, None, None),
    "RADIAL": (0, 0, 1, 2, 3, 4, None, None),
    "OPENCV": (0, 1, 2, 3, 4, 5, 6, 7),
}


def link_image_root(root: Path, sequences: dict[str, Path]) -> Path:
    """`root/<sequence_id>` symlinks to each sequence's image dir; stale links are replaced."""
    root.mkdir(parents=True, exist_ok=True)
    for sequence_id, images in sequences.items():
        link = root / sequence_id
        if link.is_symlink():
            if link.resolve() == images.resolve():
                continue
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        os.symlink(images.resolve(), link, target_is_directory=True)
    return root


def parse_image_name(name: str) -> tuple[str, str, int]:
    """`<sequence_id>/<camera_id>/<idx:06d>.jpg` -> (sequence_id, camera_id, idx)."""
    sequence_id, camera_id, file = name.rsplit("/", 2)
    return sequence_id, camera_id, int(Path(file).stem)


def camera_keys(rec: pycolmap.Reconstruction) -> dict[int, str]:
    """camera_id -> `<sequence_id>/<camera_id>`, from the image folder each camera was made for."""
    keys = {}
    for image in rec.images.values():
        sequence_id, camera_id, _ = parse_image_name(image.name)
        keys.setdefault(image.camera_id, f"{sequence_id}/{camera_id}")
    return keys


def camera_intrinsics(camera: pycolmap.Camera) -> dict[str, Any]:
    """Model, size, raw params and the named pinhole + distortion values of a camera."""
    model = camera.model.name
    if model not in _PARAM_INDEX:
        raise ValueError(f"unsupported camera model {model}")
    params = [float(v) for v in camera.params]
    fx, fy, cx, cy, k1, k2, p1, p2 = (0.0 if i is None else params[i] for i in _PARAM_INDEX[model])
    return {
        "model": model,
        "width": camera.width,
        "height": camera.height,
        "params": params,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "dist": {"k1": k1, "k2": k2, "p1": p1, "p2": p2},
    }


def write_model(rec: pycolmap.Reconstruction, path: Path, text: bool = False) -> Path:
    """Write `rec` into the directory `path` (binary, or text with `text=True`)."""
    path.mkdir(parents=True, exist_ok=True)
    rec.write_text(path) if text else rec.write_binary(path)
    return path


def to_sim3d(sim3: Sim3) -> pycolmap.Sim3d:
    """`sim3` as a pycolmap `Sim3d`, e.g. for `Reconstruction.transform`."""
    return pycolmap.Sim3d(sim3.scale, pycolmap.Rotation3d(sim3.R), sim3.t)


@contextlib.contextmanager
def colmap_log(log_file: Path | None) -> Iterator[None]:
    """Send COLMAP's glog output to `log_file` instead of the console (errors still reach stderr)."""
    if log_file is None:
        yield
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    glog = pycolmap.logging
    saved = glog.logtostderr, glog.alsologtostderr, glog.stderrthreshold
    glog.logtostderr, glog.alsologtostderr, glog.stderrthreshold = False, False, 2
    glog.set_log_destination(glog.Level.INFO, f"{log_file}.")  # glog appends a timestamp
    for level in (glog.Level.WARNING, glog.Level.ERROR, glog.Level.FATAL):
        glog.set_log_destination(level, "")
    try:
        yield
    finally:
        glog.set_log_destination(glog.Level.INFO, "")
        glog.logtostderr, glog.alsologtostderr, glog.stderrthreshold = saved
