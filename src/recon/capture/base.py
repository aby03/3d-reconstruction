from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

import cv2

from recon.core.types import CaptureSequence, Frame


class BaseCaptureLoader(ABC):
    """Turns one recording in a device-specific format into a CaptureSequence plus lazy frame access."""

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def can_load(cls, root: Path | str) -> bool:
        """Cheap check that `root` holds a capture this loader understands."""

    @abstractmethod
    def load(self) -> CaptureSequence:
        """Metadata and sensors only; no pixels are decoded."""

    @abstractmethod
    def iter_frames(
        self, camera_id: str = "cam0", indices: Iterable[int] | None = None, scale: float = 1.0
    ) -> Iterator[Frame]:
        """Upright uint8 RGB frames in index order, decoded lazily."""

    def extract_frames(
        self, indices: Iterable[int], out_dir: Path | str, camera_id: str = "cam0", quality: int = 95
    ) -> dict[int, Path]:
        """Write frames as `<out_dir>/<camera_id>/<idx:06d>.jpg`."""
        cam_dir = Path(out_dir) / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for frame in self.iter_frames(camera_id, indices):
            path = cam_dir / f"{frame.idx:06d}.jpg"
            bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise OSError(f"failed to write {path}")
            paths[frame.idx] = path
        return paths
