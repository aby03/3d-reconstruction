from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Sequence

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "room"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "room_reconstruction"


def _image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions),
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


def _new_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        output_dir = requested.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose a new path to preserve previous runs."
            )
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


def _run_colmap(command: Sequence[str], log_path: Path) -> None:
    print(f"Running: {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {shlex.join(command)}\n")
        log_file.flush()
        try:
            subprocess.run(
                command,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            log_file.flush()
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
            raise RuntimeError(
                f"COLMAP exited with status {error.returncode}; see {log_path} for the complete log."
            ) from error


def _export_point_cloud(ply_path: Path, glb_path: Path) -> int:
    cloud = trimesh.load(ply_path, force="pointcloud", process=False)
    vertices = np.asarray(cloud.vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"COLMAP produced an empty or invalid point cloud: {ply_path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"COLMAP point cloud contains non-finite coordinates: {ply_path}")

    cloud.export(glb_path, file_type="glb")
    if not glb_path.is_file() or glb_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write GLB point cloud: {glb_path}")
    return len(vertices)


def _export_mesh(ply_path: Path, glb_path: Path) -> tuple[int, int]:
    mesh = trimesh.load(ply_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"COLMAP produced an empty or invalid mesh: {ply_path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"COLMAP mesh has no triangular faces: {ply_path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"COLMAP mesh contains non-finite coordinates: {ply_path}")

    mesh.export(glb_path, file_type="glb")
    if not glb_path.is_file() or glb_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write GLB mesh: {glb_path}")
    return len(vertices), len(faces)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COLMAP sparse and dense reconstruction on an Open Camera Sensors capture."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing extracted capture frames; defaults to the only image subdirectory in data-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. It must be empty or not yet exist; otherwise a timestamped directory is used.",
    )
    parser.add_argument("--quality", choices=("low", "medium", "high", "extreme"), default="high")
    parser.add_argument(
        "--data-type",
        choices=("individual", "video"),
        default="individual",
        help="individual uses exhaustive matching; video uses sequential matching. Exhaustive is practical for this 46-frame capture.",
    )
    parser.add_argument(
        "--camera-model",
        choices=("SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_PINHOLE", "PINHOLE"),
        default="SIMPLE_RADIAL",
        help="Use a shared camera model; COLMAP self-calibrates because this capture has no intrinsics sidecar.",
    )
    parser.add_argument("--camera-params", default=None, help="Optional comma-separated known COLMAP camera parameters.")
    parser.add_argument("--feature", choices=("sift", "aliked", "loma", "loma128"), default="sift")
    parser.add_argument("--mapper", choices=("incremental", "hierarchical", "global"), default="incremental")
    parser.add_argument("--mesher", choices=("delaunay", "poisson"), default="delaunay")
    parser.add_argument("--num-threads", type=int, default=-1)
    parser.add_argument("--gpu-index", default="-1", help="COLMAP GPU index; -1 selects all available GPUs.")
    parser.add_argument("--cpu", action="store_true", help="Disable COLMAP GPU feature extraction and stereo.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    image_dir = _find_image_directory(data_dir, args.image_dir)
    images = _image_files(image_dir)
    if len(images) < 2:
        raise RuntimeError(f"At least two images are required, found {len(images)}")

    colmap = shutil.which("colmap")
    if colmap is None:
        raise RuntimeError("COLMAP CLI was not found. Run this entrypoint through the Pixi task.")

    output_dir = _new_output_dir(args.output_dir)
    log_path = output_dir / "colmap.log"
    database_path = output_dir / "database.db"
    use_gpu = "0" if args.cpu else "1"

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "data_dir": str(data_dir),
        "image_dir": str(image_dir),
        "image_count": len(images),
        "image_names": [path.name for path in images],
        "sensor_files_recorded_not_fused": [path.name for path in sorted(data_dir.rglob("*.csv"))],
        "method": "COLMAP automatic_reconstructor",
        "data_type": args.data_type,
        "feature": args.feature,
        "mapper": args.mapper,
        "mesher": args.mesher,
        "quality": args.quality,
        "camera_model": args.camera_model,
        "single_shared_camera": True,
        "camera_params": args.camera_params,
        "use_gpu": not args.cpu,
        "gpu_index": args.gpu_index,
        "note": "No Open Camera Sensors intrinsics sidecar was found; COLMAP self-calibrates the shared camera. IMU CSVs are preserved but not fused into poses.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    command = [
        colmap,
        "automatic_reconstructor",
        "--workspace_path",
        str(output_dir),
        "--image_path",
        str(image_dir),
        "--data_type",
        args.data_type,
        "--quality",
        args.quality,
        "--camera_model",
        args.camera_model,
        "--single_camera",
        "1",
        "--feature",
        args.feature,
        "--mapper",
        args.mapper,
        "--mesher",
        args.mesher,
        "--use_gpu",
        use_gpu,
        "--gpu_index",
        str(args.gpu_index),
        "--num_threads",
        str(args.num_threads),
        "--random_seed",
        "0",
    ]
    if args.camera_params:
        command.extend(("--camera_params", args.camera_params))

    print(f"Frames: {len(images)} in {image_dir}")
    print(f"Output directory: {output_dir}")
    mesher_warning = None
    try:
        _run_colmap(command, log_path)
    except RuntimeError as error:
        fused_candidates = sorted(output_dir.glob("**/fused.ply"))
        if not fused_candidates:
            raise
        mesher_warning = str(error)
        print(
            "COLMAP failed after dense fusion, but a fused point cloud exists; "
            "continuing with point-cloud export.",
            file=sys.stderr,
        )

    fused_candidates = sorted(output_dir.glob("**/fused.ply"))
    if not fused_candidates:
        raise RuntimeError(
            f"COLMAP completed but no fused.ply was found under {output_dir}. Check {log_path} and the sparse model."
        )
    fused_ply = max(fused_candidates, key=lambda path: path.stat().st_size)
    glb_path = output_dir / "fused_point_cloud.glb"
    point_count = _export_point_cloud(fused_ply, glb_path)

    mesh_candidates = sorted(output_dir.glob("**/meshed-delaunay.ply"))
    if not mesh_candidates:
        dense_workspace = fused_ply.parent
        mesh_ply = dense_workspace / "meshed-delaunay.ply"
        _run_colmap(
            [
                colmap,
                "delaunay_mesher",
                "--input_path",
                str(dense_workspace),
                "--input_type",
                "dense",
                "--output_path",
                str(mesh_ply),
                "--DelaunayMeshing.max_proj_dist",
                "30",
            ],
            log_path,
        )
        mesh_candidates = sorted(output_dir.glob("**/meshed-delaunay.ply"))
    if not mesh_candidates:
        raise RuntimeError(f"COLMAP did not produce a Delaunay mesh under {output_dir}")

    mesh_ply = max(mesh_candidates, key=lambda path: path.stat().st_size)
    mesh_glb = output_dir / "dense_mesh.glb"
    mesh_vertex_count, mesh_face_count = _export_mesh(mesh_ply, mesh_glb)

    summary = {
        "image_count": len(images),
        "fused_ply": str(fused_ply.relative_to(output_dir)),
        "glb_point_cloud": str(glb_path.relative_to(output_dir)),
        "point_count": point_count,
        "glb_size_bytes": glb_path.stat().st_size,
        "dense_mesh_ply": str(mesh_ply.relative_to(output_dir)),
        "dense_mesh_glb": str(mesh_glb.relative_to(output_dir)),
        "mesh_vertex_count": mesh_vertex_count,
        "mesh_face_count": mesh_face_count,
        "mesh_glb_size_bytes": mesh_glb.stat().st_size,
        "colmap_post_fusion_warning": mesher_warning,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nReconstruction complete.")
    print(f"Dense PLY: {fused_ply}")
    print(f"GLB point cloud: {glb_path}")
    print(f"Points: {point_count:,}")
    print(f"Dense mesh PLY: {mesh_ply}")
    print(f"Dense mesh GLB: {mesh_glb}")
    print(f"Mesh vertices/faces: {mesh_vertex_count:,}/{mesh_face_count:,}")
    print(f"Run log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error