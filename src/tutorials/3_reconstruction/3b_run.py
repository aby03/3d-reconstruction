from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys

import cv2
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "room2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "room2_reconstruction"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OPENMVS_TOOLS = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )


def _find_image_directory(data_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        image_dir = requested.expanduser().resolve()
        if not image_dir.is_dir() or not _image_files(image_dir):
            raise FileNotFoundError(f"No supported image files found in {image_dir}")
        return image_dir

    if _image_files(data_dir):
        return data_dir

    candidates = [child for child in sorted(data_dir.iterdir()) if child.is_dir() and _image_files(child)]
    if len(candidates) != 1:
        locations = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            f"Expected one image directory under {data_dir}, found {len(candidates)}: {locations}. "
            "Pass --image-dir to select one explicitly."
        )
    return candidates[0]


def _find_video(data_dir: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        video_path = requested.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
        return video_path

    candidates = sorted(data_dir.glob("*.mp4"))
    if len(candidates) > 1:
        locations = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"Found multiple MP4 files in {data_dir}: {locations}. Pass --video to select one.")
    return candidates[0] if candidates else None


def _new_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        output_dir = requested.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / run_id
    suffix = 1
    while output_dir.exists():
        output_dir = DEFAULT_OUTPUT_ROOT / f"{run_id}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def _stage_frames(image_dir: Path, staged_dir: Path, min_sharpness: float) -> dict[str, float]:
    staged_dir.mkdir(parents=True)
    scores: dict[str, float] = {}
    for image_path in _image_files(image_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scores[image_path.name] = score
        if score >= min_sharpness:
            shutil.copy2(image_path, staged_dir / image_path.name)

    if len(_image_files(staged_dir)) < 2:
        raise RuntimeError("Fewer than two usable frames remain; lower --min-sharpness.")
    return scores


def _stage_video(
    video_path: Path,
    staged_dir: Path,
    sample_fps: float,
    min_sharpness: float,
) -> tuple[dict[str, float], float, int]:
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or source_fps <= 0:
        capture.release()
        raise RuntimeError(f"OpenCV could not open video or read its frame rate: {video_path}")

    staged_dir.mkdir(parents=True)
    frame_stride = max(1, round(source_fps / sample_fps))
    scores: dict[str, float] = {}
    frame_index = 0
    while True:
        success, image = capture.read()
        if not success:
            break
        if frame_index % frame_stride == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            name = f"video_{frame_index:09d}.jpg"
            scores[name] = score
            if score >= min_sharpness and not cv2.imwrite(str(staged_dir / name), image):
                capture.release()
                raise RuntimeError(f"OpenCV failed to write sampled frame: {staged_dir / name}")
        frame_index += 1
    capture.release()

    if len(_image_files(staged_dir)) < 2:
        raise RuntimeError("Fewer than two usable frames remain; lower --min-sharpness or increase sampling rate.")
    return scores, source_fps, frame_stride


def _run(command: list[str], log_path: Path, label: str) -> None:
    print(f"Running: {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {shlex.join(command)}\n")
        log_file.flush()
        try:
            subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as error:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
            raise RuntimeError(f"{label} exited with status {error.returncode}; see {log_path}.") from error


def _make_obj_material_opaque(material_path: Path) -> None:
    if not material_path.is_file():
        raise RuntimeError(f"OpenMVS did not write the OBJ material file: {material_path}")
    material = material_path.read_text(encoding="utf-8")
    # OpenMVS writes Tr 1, which means fully transparent in standard MTL.
    material_path.write_text(material.replace("\nTr 1.000000\n", "\nTr 0.000000\n"), encoding="utf-8")


def _embed_glb_textures(source_path: Path, output_path: Path) -> None:
    scene = trimesh.load(source_path, force="scene", process=False)
    if not scene.geometry:
        raise RuntimeError(f"OpenMVS produced an empty textured GLB: {source_path}")

    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.glb")
    scene.export(temporary_path, file_type="glb")
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to package textured GLB: {temporary_path}")

    binary = temporary_path.read_bytes()
    json_length = struct.unpack_from("<I", binary, 12)[0]
    document = json.loads(binary[20 : 20 + json_length])
    if any("uri" in image for image in document.get("images", [])):
        temporary_path.unlink()
        raise RuntimeError("GLB still contains external texture references instead of embedded image data.")
    temporary_path.replace(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct room2 with OpenCV frame checks, COLMAP poses, and OpenMVS texturing."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None, help="Video to sample; defaults to the only MP4 in data-dir.")
    parser.add_argument("--images-only", action="store_true", help="Use extracted images instead of sampling video.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling rate when extracting video frames.")
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Optional variance-of-Laplacian threshold; 0 keeps every decodable frame.",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="OpenMVS dense resolution: 0 is full, 1 half, and 2 quarter resolution.",
    )
    parser.add_argument("--num-threads", type=int, default=-1)
    parser.add_argument("--gpu-index", default="-1")
    parser.add_argument("--cpu", action="store_true", help="Disable COLMAP SIFT GPU processing.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if args.min_sharpness < 0:
        raise RuntimeError("--min-sharpness must be non-negative.")
    if args.sample_fps <= 0:
        raise RuntimeError("--sample-fps must be positive.")

    use_video = not args.images_only and (args.video is not None or args.image_dir is None)
    video_path = _find_video(data_dir, args.video) if use_video else None
    image_dir = None if video_path is not None else _find_image_directory(data_dir, args.image_dir)
    colmap = shutil.which("colmap")
    openmvs = {tool: shutil.which(tool) for tool in OPENMVS_TOOLS}
    missing = [tool for tool, executable in openmvs.items() if executable is None]
    if colmap is None:
        raise FileNotFoundError("COLMAP CLI was not found; run through the Pixi environment.")
    if missing:
        raise FileNotFoundError(
            "OpenMVS tools are required and were not found on PATH: " + ", ".join(missing)
        )

    output_dir = _new_output_dir(args.output_dir)
    staged_dir = output_dir / "images"
    source_fps = None
    frame_stride = None
    if video_path is not None:
        frame_scores, source_fps, frame_stride = _stage_video(
            video_path, staged_dir, args.sample_fps, args.min_sharpness
        )
    else:
        frame_scores = _stage_frames(image_dir, staged_dir, args.min_sharpness)
    selected_images = _image_files(staged_dir)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir()
    openmvs_dir = output_dir / "openmvs"
    openmvs_dir.mkdir()
    intermediate_dir = openmvs_dir / "intermediate"
    intermediate_dir.mkdir()
    log_path = output_dir / "reconstruction.log"
    use_gpu = "0" if args.cpu else "1"

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "data_dir": str(data_dir),
        "source_image_dir": str(image_dir) if image_dir is not None else None,
        "source_video": str(video_path) if video_path is not None else None,
        "image_count": len(selected_images),
        "source_image_count": len(frame_scores),
        "selected_images": [path.name for path in selected_images],
        "opencv_version": cv2.__version__,
        "sharpness_metric": "variance_of_laplacian",
        "min_sharpness": args.min_sharpness,
        "frame_sharpness_scores": frame_scores,
        "video_source_fps": source_fps,
        "video_frame_stride": frame_stride,
        "video_sample_fps_requested": args.sample_fps if video_path is not None else None,
        "openmvs_resolution_level": args.resolution_level,
        "openmvs_texture_options": {
            "global_seam_leveling": False,
            "local_seam_leveling": False,
            "reason": "work around black texture sampling bug in OpenMVS 2.4.0",
        },
        "sensor_files_preserved_not_fused": [path.name for path in sorted(data_dir.rglob("*.csv"))],
        "method": "OpenCV frame validation + COLMAP SfM + OpenMVS MVS and texturing",
        "openmvs_tools": openmvs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _run(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(staged_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
            "--FeatureExtraction.use_gpu",
            use_gpu,
            "--FeatureExtraction.gpu_index",
            str(args.gpu_index),
            "--FeatureExtraction.num_threads",
            str(args.num_threads),
        ],
        log_path,
        "COLMAP feature extraction",
    )
    matcher_command = [
        colmap,
        "sequential_matcher" if video_path is not None else "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        use_gpu,
        "--FeatureMatching.gpu_index",
        str(args.gpu_index),
        "--FeatureMatching.num_threads",
        str(args.num_threads),
    ]
    if video_path is not None:
        matcher_command.extend(("--SequentialMatching.overlap", "10"))
    _run(
        matcher_command,
        log_path,
        "COLMAP exhaustive matching",
    )
    _run(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(staged_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.num_threads",
            str(args.num_threads),
        ],
        log_path,
        "COLMAP mapping",
    )

    sparse_models = sorted(
        path for path in sparse_dir.iterdir() if path.is_dir() and (path / "images.bin").is_file()
    )
    if not sparse_models:
        raise RuntimeError(f"COLMAP produced no sparse model; see {log_path}.")
    sparse_model = max(sparse_models, key=lambda path: (path / "images.bin").stat().st_size)
    undistorted_dir = intermediate_dir / "undistorted"
    _run(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(staged_dir),
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(undistorted_dir),
            "--output_type",
            "COLMAP",
        ],
        log_path,
        "COLMAP image undistortion",
    )

    scene_path = intermediate_dir / "scene.mvs"
    _run(
        [
            openmvs["InterfaceCOLMAP"],
            "--working-folder",
            str(intermediate_dir),
            "-i",
            str(undistorted_dir),
            "-o",
            str(scene_path),
            "--image-folder",
            str(undistorted_dir / "images"),
        ],
        log_path,
        "OpenMVS COLMAP import",
    )
    _run(
        [
            openmvs["DensifyPointCloud"],
            "--working-folder",
            str(intermediate_dir),
            str(scene_path),
            "--resolution-level",
            str(args.resolution_level),
            "--number-views",
            "8",
            "--number-views-fuse",
            "3",
        ],
        log_path,
        "OpenMVS densification",
    )
    dense_scene = intermediate_dir / "scene_dense.mvs"
    mesh_path = intermediate_dir / "scene_dense_mesh.ply"
    _run(
        [
            openmvs["ReconstructMesh"],
            "--working-folder",
            str(intermediate_dir),
            "-i",
            str(dense_scene),
            "-o",
            str(mesh_path),
        ],
        log_path,
        "OpenMVS mesh reconstruction",
    )
    refined_mesh_path = intermediate_dir / "scene_dense_mesh_refine.ply"
    _run(
        [
            openmvs["RefineMesh"],
            "--working-folder",
            str(intermediate_dir),
            "-i",
            str(dense_scene),
            "-m",
            str(mesh_path),
            "-o",
            str(refined_mesh_path),
        ],
        log_path,
        "OpenMVS mesh refinement",
    )
    textured_mesh_path = openmvs_dir / "scene_textured.obj"
    _run(
        [
            openmvs["TextureMesh"],
            "--working-folder",
            str(intermediate_dir),
            "-i",
            str(dense_scene),
            "-m",
            str(refined_mesh_path),
            "-o",
            str(textured_mesh_path),
            "--export-type",
            "obj",
            # OpenMVS 2.4.0's seam-leveling image sampler creates black texture
            # regions; keep both seam passes disabled until using a fixed build.
            "--global-seam-leveling",
            "0",
            "--local-seam-leveling",
            "0",
        ],
        log_path,
        "OpenMVS texturing",
    )
    textured_glb_path = openmvs_dir / "scene_textured.glb"
    _run(
        [
            openmvs["TextureMesh"],
            "--working-folder",
            str(intermediate_dir),
            "-i",
            str(dense_scene),
            "-m",
            str(refined_mesh_path),
            "-o",
            str(textured_glb_path),
            "--export-type",
            "glb",
            "--global-seam-leveling",
            "0",
            "--local-seam-leveling",
            "0",
        ],
        log_path,
        "OpenMVS GLB export",
    )
    _embed_glb_textures(textured_glb_path, textured_glb_path)
    _make_obj_material_opaque(openmvs_dir / "scene_textured.mtl")

    intermediate_outputs = sorted(
        path.relative_to(output_dir).as_posix()
        for path in intermediate_dir.rglob("*")
        if path.is_file()
    )
    final_outputs = sorted(
        path.relative_to(output_dir).as_posix()
        for path in openmvs_dir.iterdir()
        if path.is_file()
    )
    summary = {
        "image_count": len(selected_images),
        "sparse_model": str(sparse_model.relative_to(output_dir)),
        "undistorted_workspace": str(undistorted_dir.relative_to(output_dir)),
        "openmvs_scene": str(scene_path.relative_to(output_dir)),
        "intermediate_outputs": intermediate_outputs,
        "final_outputs": final_outputs,
        "run_log": str(log_path.relative_to(output_dir)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Reconstruction complete: {output_dir}")
    print(f"OpenMVS intermediate files: {len(intermediate_outputs)}")
    print(f"Textured OBJ: {textured_mesh_path}")
    print(f"Textured GLB: {textured_glb_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
