import subprocess
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


_CV_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_cw(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Rotate an (H, W, ...) image clockwise by a multiple of 90 degrees."""
    if rotation_deg % 90:
        raise ValueError(f"rotation must be a multiple of 90, got {rotation_deg}")
    code = _CV_ROTATIONS[rotation_deg % 360]
    return np.ascontiguousarray(image) if code is None else cv2.rotate(image, code)


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
                    # resizing first is cheaper and gives the same image: area resampling commutes with 90° turns
                    image = resize(frame.to_ndarray(format="rgb24"), scale)
                    yield idx, rotate_cw(image, self.rotation_deg)


class NvdecVideoReader:
    """Sequential GPU decode: ffmpeg with NVDEC (`<codec>_cuvid`, scaled with `scale_cuda`) piped as raw RGB.

    Same frames, order and indexing as VideoReader; rotation is applied in numpy, never by the decoder.
    """

    def __init__(
        self, path: Path | str, rotation_deg: int, stored_size: tuple[int, int], codec: str = "h264", gpu_index: int = 0
    ):
        self.path = Path(path)
        self.rotation_deg = rotation_deg
        self.stored_size = stored_size
        self.codec = codec
        self.gpu_index = gpu_index

    def _command(self, size: tuple[int, int]) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-hwaccel", "cuda"]
        cmd += ["-hwaccel_device", str(self.gpu_index)]
        scaled = size != self.stored_size
        if scaled:
            cmd += ["-hwaccel_output_format", "cuda"]
        cmd += ["-c:v", f"{self.codec}_cuvid", "-noautorotate", "-i", str(self.path), "-map", "0:v:0"]
        # NVDEC outputs nv12; going through yuv420p makes the RGB conversion bit-identical to PyAV's.
        # Bilinear at a 2x reduction averages 2x2 blocks, close to the CPU path's area resampling.
        vf = "format=yuv420p"
        if scaled:
            vf = f"scale_cuda={size[0]}:{size[1]}:interp_algo=bilinear,hwdownload,format=nv12,{vf}"
        cmd += ["-vf", vf]
        return cmd + ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    def iter_frames(
        self, indices: Iterable[int] | None = None, scale: float = 1.0
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (idx, upright uint8 RGB) in index order. With `indices`, decode through and keep only those."""
        wanted = None if indices is None else set(indices)
        last = None if wanted is None else max(wanted, default=-1)
        w, h = self.stored_size
        size = (round(w * scale), round(h * scale)) if scale != 1.0 else (w, h)
        frame_bytes = size[0] * size[1] * 3
        proc = subprocess.Popen(self._command(size), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        idx = 0
        try:
            while last is None or idx <= last:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                if wanted is None or idx in wanted:
                    image = np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3)
                    yield idx, rotate_cw(image, self.rotation_deg)
                idx += 1
        finally:
            if proc.poll() is None:
                proc.kill()
            _, err = proc.communicate()
        if proc.returncode and (last is None or idx <= last):
            raise RuntimeError(f"ffmpeg NVDEC decode of {self.path} failed: {err.decode(errors='replace').strip()}")
