"""Check decoded frames against the full-size JPEGs a capture app saved alongside the video."""

from pathlib import Path

import cv2
import numpy as np

from recon.capture.base import BaseCaptureLoader
from recon.core.types import CaptureSequence
from recon.io.video import rotate_cw

ROTATIONS = (0, 90, 180, 270)
NUM_ROTATION_CHECKS = 3


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """PSNR in dB of `image` against `reference`, resizing `image` to the reference size if needed."""
    h, w = reference.shape[:2]
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def check_reference_images(
    loader: BaseCaptureLoader, sequence: CaptureSequence, camera_id: str = "cam0", max_checks: int | None = None
) -> dict:
    """PSNR of each decoded upright frame against its reference JPEG.

    On the first few references every rotation is scored too, to confirm the chosen one is the best.
    """
    chosen = sequence.cameras[camera_id].rotation_deg
    refs = dict(sorted(sequence.reference_images.items())[:max_checks])

    psnr_db = {}
    rotation_scores: dict[int, list[float]] = {r: [] for r in ROTATIONS}
    for n, frame in enumerate(loader.iter_frames(camera_id, refs)):
        reference = _read_rgb(refs[frame.idx])
        psnr_db[frame.idx] = _psnr(frame.image, reference)
        if n < NUM_ROTATION_CHECKS:
            for r in ROTATIONS:
                rotation_scores[r].append(_psnr(rotate_cw(frame.image, (r - chosen) % 360), reference))

    rotation_psnr_db = {r: float(np.mean(s)) for r, s in rotation_scores.items() if s}
    best_rotation = max(rotation_psnr_db, key=rotation_psnr_db.get) if rotation_psnr_db else None
    return {
        "checked": len(psnr_db),
        "psnr_db": psnr_db,
        "min_psnr_db": min(psnr_db.values()) if psnr_db else None,
        "rotation_psnr_db": rotation_psnr_db,
        "best_rotation": best_rotation,
        "rotation_mismatch": best_rotation is not None and best_rotation != chosen,
    }
