from dataclasses import dataclass
from pathlib import Path

import av
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
