"""Cut a short clip out of an Open Camera Sensors capture, keeping the capture layout.

Only the selected frames are re-encoded, so frame i of the clip is exactly the i-th row of
its timestamp file. Sensor CSVs keep their original lines over the frame span +-1 s.
The clip is checked before it is moved into place: frame count, frame-interval pattern,
display rotation, and pixel agreement of the first and last frame with the source.
--long-side N shrinks video and reference JPEGs (JPEG quality 85, video CRF 28). Container and JPEG
metadata (EXIF, GPS, creation time, device tags) is never copied.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from recon.capture.open_camera_sensors import discover_files, read_frame_timestamps

SENSOR_MARGIN_NS = 1_000_000_000
MIN_REFERENCE_IMAGES = 3
DROP_FACTOR = 1.5  # frame interval above this multiple of the median counts as a dropped frame
MAX_PTS_ERROR_S = 1e-3
MIN_FRAME_PSNR_DB = 35.0
MIN_RESIZED_PSNR_DB = 28.0
FULL_CRF, RESIZED_CRF = 16, 28
JPEG_QUALITY = 85


def _run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def _rotation(video: Path) -> int:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video)])
    side_data = json.loads(out)["streams"][0].get("side_data_list", [])
    return next((int(d["rotation"]) for d in side_data if "rotation" in d), 0)


def _packet_pts(video: Path) -> np.ndarray:
    """Presentation times of all video packets, sorted (= output frame order)."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                "-of", "csv=p=0", str(video)])
    return np.sort(np.array(out.split(), dtype=np.float64))


def _stored_size(video: Path) -> tuple[int, int]:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "json", str(video)])
    stream = json.loads(out)["streams"][0]
    return stream["width"], stream["height"]


def _scaled_size(size: tuple[int, int], long_side: int) -> tuple[int, int]:
    """Size with the long side `long_side`, aspect kept, both dimensions even."""
    k = long_side / max(size)
    return tuple(2 * round(d * k / 2) for d in size)


def _gray_frame(video: Path, n: int, size: tuple[int, int] | None = None) -> np.ndarray:
    """Decode frame n (stored orientation) as 8-bit gray, optionally scaled to size (w, h)."""
    scale = f",scale={size[0]}:{size[1]}:flags=area" if size else ""
    out = _run(["ffmpeg", "-v", "error", "-noautorotate", "-i", str(video), "-map", "0:v:0",
                "-vf", f"select=eq(n\\,{n}){scale}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    return np.frombuffer(out, dtype=np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _choose_window(t_ns: np.ndarray, refs: list[int], duration_s: float) -> float:
    """Earliest start (0.5 s steps) whose window holds a dropped frame and enough reference JPEGs."""
    rel_s = (t_ns - t_ns[0]) * 1e-9
    dt = np.diff(t_ns)
    drops_s = rel_s[1:][dt > DROP_FACTOR * np.median(dt)]
    refs_s = (np.array(refs, dtype=np.int64) - t_ns[0]) * 1e-9
    for start in np.arange(0.0, rel_s[-1] - duration_s, 0.5):
        end = start + duration_s
        n_refs = np.count_nonzero((refs_s >= start) & (refs_s < end))
        if n_refs >= MIN_REFERENCE_IMAGES and np.any((drops_s > start) & (drops_s < end)):
            return float(start)
    raise SystemExit(f"no {duration_s} s window with a dropped frame and {MIN_REFERENCE_IMAGES} reference JPEGs")


def _encode(src: Path, dst: Path, i0: int, i1: int, rotation: int, size: tuple[int, int] | None) -> None:
    raw = dst.with_suffix(".raw.mp4")
    scale = f",scale={size[0]}:{size[1]}:flags=lanczos" if size else ""
    _run(["ffmpeg", "-v", "error", "-y", "-noautorotate", "-i", str(src), "-map", "0:v:0", "-an",
          "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
          "-vf", f"select=between(n\\,{i0}\\,{i1}),setpts=PTS-STARTPTS{scale}", "-fps_mode", "passthrough",
          "-c:v", "libx264", "-crf", str(RESIZED_CRF if size else FULL_CRF), "-preset", "medium",
          "-x264-params", "no-info=1", "-pix_fmt", "yuv420p", str(raw)])
    # Encoding without autorotation drops the display matrix; restore it with a stream copy.
    # ffmpeg's -display_rotation is counter-clockwise, ffprobe's rotation value is the same angle.
    _run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(raw),
          "-map_metadata", "-1", "-fflags", "+bitexact", "-c", "copy", str(dst)])
    raw.unlink()


def _write_lines_in_range(src: Path, dst: Path, t_lo: int, t_hi: int) -> int:
    """Copy CSV lines whose last field (t_ns) lies in [t_lo, t_hi], unchanged."""
    lines = [line for line in src.read_text().splitlines() if t_lo <= int(line.rsplit(",", 1)[-1]) <= t_hi]
    dst.write_text("\n".join(lines) + "\n")
    return len(lines)


def _write_jpeg(src: Path, dst: Path, size: tuple[int, int] | None) -> None:
    """Re-save a reference JPEG without EXIF/GPS/ICC; `size` is the upright (w, h) or None."""
    with Image.open(src) as im:
        if size:
            im = im.resize(size, Image.LANCZOS)
        im.convert("RGB").save(dst, quality=JPEG_QUALITY)


def _jpeg_size(size: tuple[int, int] | None, rotation: int) -> tuple[int, int] | None:
    return None if size is None else (size[1], size[0]) if rotation % 180 else size


def _verify(src: Path, clip: Path, i0: int, i1: int, rotation: int, resized: bool) -> None:
    n = i1 - i0 + 1
    src_pts, clip_pts = _packet_pts(src)[i0 : i1 + 1], _packet_pts(clip)
    if len(clip_pts) != n:
        raise SystemExit(f"clip has {len(clip_pts)} frames, expected {n}")
    pts_error = np.max(np.abs(np.diff(clip_pts) - np.diff(src_pts)))
    if pts_error > MAX_PTS_ERROR_S:
        raise SystemExit(f"frame intervals differ from the source by up to {pts_error * 1e3:.2f} ms")
    if _rotation(clip) != rotation:
        raise SystemExit(f"clip rotation {_rotation(clip)} != source rotation {rotation}")
    size = _stored_size(clip) if resized else None
    min_psnr = MIN_RESIZED_PSNR_DB if resized else MIN_FRAME_PSNR_DB
    for k in (0, n - 1):
        psnr = _psnr(_gray_frame(src, i0 + k, size), _gray_frame(clip, k))
        if psnr < min_psnr:
            raise SystemExit(f"clip frame {k} does not match source frame {i0 + k} (PSNR {psnr:.1f} dB)")
        print(f"  frame {k} vs source {i0 + k}: PSNR {psnr:.1f} dB")
    print(f"  frame intervals match within {pts_error * 1e3:.3f} ms, rotation {rotation}")


def make_clip(src_root: Path, out: Path, start_s: float | None, duration_s: float,
              long_side: int | None = None) -> None:
    files = discover_files(src_root)
    t_ns = read_frame_timestamps(files.frame_timestamps)
    if start_s is None:
        start_s = _choose_window(t_ns, list(files.reference_images), duration_s)

    rel_s = (t_ns - t_ns[0]) * 1e-9
    idx = np.flatnonzero((rel_s >= start_s) & (rel_s < start_s + duration_s))
    if len(idx) == 0:
        raise SystemExit(f"no frames in [{start_s}, {start_s + duration_s}) s")
    i0, i1 = int(idx[0]), int(idx[-1])
    t_lo, t_hi = int(t_ns[i0]), int(t_ns[i1])
    refs = {t: p for t, p in files.reference_images.items() if t_lo <= t <= t_hi}
    if len(refs) < MIN_REFERENCE_IMAGES:
        raise SystemExit(f"window has {len(refs)} reference JPEGs, need {MIN_REFERENCE_IMAGES}")
    print(f"{files.root} -> {out}: {start_s:.1f}-{start_s + duration_s:.1f} s, frames {i0}..{i1}")

    rotation = _rotation(files.video)
    size = _scaled_size(_stored_size(files.video), long_side) if long_side else None
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}.") as tmp:
        stage = Path(tmp) / out.name
        stamp_dir = stage / files.stamp
        stamp_dir.mkdir(parents=True)

        clip_video = stage / files.video.name
        _encode(files.video, clip_video, i0, i1, rotation, size)
        (stamp_dir / files.frame_timestamps.name).write_text("\n".join(map(str, t_ns[i0 : i1 + 1])) + "\n")
        for kind, path in files.sensors.items():
            n = _write_lines_in_range(path, stamp_dir / path.name, t_lo - SENSOR_MARGIN_NS, t_hi + SENSOR_MARGIN_NS)
            print(f"  {kind}: {n} samples")
        for path in refs.values():
            _write_jpeg(path, stamp_dir / path.name, _jpeg_size(size, rotation))
        print(f"  {len(refs)} reference JPEGs")

        _verify(files.video, clip_video, i0, i1, rotation, size is not None)
        stage.rename(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="capture root (or its stamp dir)")
    parser.add_argument("--out", type=Path, required=True, help="output capture root")
    parser.add_argument("--start", type=float, default=None,
                        help="window start in s from the first frame (default: first window with a dropped frame)")
    parser.add_argument("--duration", type=float, default=5.0, help="window length in s")
    parser.add_argument("--long-side", type=int, default=None,
                        help="shrink video and reference JPEGs to this long side in px (default: keep size)")
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists, pass --force to replace it")
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_clip(args.input, args.out, args.start, args.duration, args.long_side)


if __name__ == "__main__":
    main()
