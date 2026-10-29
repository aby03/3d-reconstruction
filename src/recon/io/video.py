from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass(slots=True)
class VideoProbe:
    codec: str
    stored_size: tuple[int, int]  # (W, H) as encoded
    nominal_fps: float
    display_rotation: int  # degrees counter-clockwise, as in the display matrix
    pts_s: np.ndarray  # (N,) float64, one per decoded frame, in output order

    @property
    def num_frames(self) -> int:
        return len(self.pts_s)


def probe_video(path: Path | str) -> VideoProbe:
    """Read stream properties and every frame's PTS by demuxing only; just the first frame is decoded."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Packets arrive in decode order; with B-frames that differs from output order, so sort by PTS.
        pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
        container.seek(0)
        # PyAV exposes the display matrix only on decoded frames.
        display_rotation = int(next(container.decode(stream)).rotation)
        return VideoProbe(
            codec=stream.codec_context.name,
            stored_size=(stream.codec_context.width, stream.codec_context.height),
            nominal_fps=float(stream.average_rate or stream.guessed_rate or 0),
            display_rotation=display_rotation,
            pts_s=np.asarray(pts, dtype=np.float64) * float(stream.time_base),
        )


def rotate_cw(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Rotate an (H, W, ...) image clockwise by a multiple of 90 degrees."""
    if rotation_deg % 90:
        raise ValueError(f"rotation must be a multiple of 90, got {rotation_deg}")
    return np.ascontiguousarray(np.rot90(image, k=-(rotation_deg // 90)))


def resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


class VideoReader:
    """Sequential CPU decode with PyAV. Rotation is applied here in numpy, never by the decoder."""

    def __init__(self, path: Path | str, rotation_deg: int = 0):
        self.path = Path(path)
        self.rotation_deg = rotation_deg

    def iter_frames(
        self, indices: Iterable[int] | None = None, scale: float = 1.0
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (idx, upright uint8 RGB) in index order. With `indices`, decode through and keep only those."""
        wanted = None if indices is None else set(indices)
        last = None if wanted is None else max(wanted, default=-1)
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for idx, frame in enumerate(container.decode(stream)):
                if last is not None and idx > last:
                    break
                if wanted is None or idx in wanted:
                    image = rotate_cw(frame.to_ndarray(format="rgb24"), self.rotation_deg)
                    yield idx, resize(image, scale)
