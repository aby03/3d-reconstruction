"""OpenMVS: argv builders for each tool and the tool runs (through a ToolRunner), in one working folder.

Every tool runs with `cwd = work_dir` and `-w .`, so the paths OpenMVS stores in its `.mvs` files are relative
to `work_dir`: the working folder and the undistorted images next to it must not move.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from recon.io.tools import ToolResult, ToolRunner

log = logging.getLogger(__name__)

INTERFACE_COLMAP = "InterfaceCOLMAP"
DENSIFY = "DensifyPointCloud"
RECONSTRUCT_MESH = "ReconstructMesh"
SCENE = "scene.mvs"
SCENE_DENSE = "scene_dense.mvs"
DENSE_PLY = "scene_dense.ply"  # referenced by scene_dense.mvs
MESH_PLY = "scene_dense_mesh.ply"
CPU_DEVICE = -2  # --cuda-device: -2 CPU, -1 best GPU, >= 0 GPU index
DENSE_POINTS_PATTERN = r"Densifying point-cloud completed: (\d+) points"


@dataclass(slots=True)
class OpenMvsOptions:
    resolution_level: int = 1  # images scaled down by 2^level before patch match ...
    min_resolution: int = 640  # ... but the long side never below this
    max_resolution: int = 2560
    number_views: int = 6
    number_views_fuse: int = 3
    sub_resolution_levels: int = 2
    iters: int = 3
    geometric_iters: int = 2
    filter_point_cloud: int = 0
    crop_to_roi: bool = False  # OpenMVS's estimated ROI cuts the upper walls
    keep_dmaps: bool = False
    bin_dir: str | None = None  # None: OpenMVS from PATH


@dataclass(slots=True)
class MeshOptions:
    remove_spurious: float = 20.0
    remove_spikes: bool = True
    close_holes: int = 30
    smooth: int = 2
    decimate: float = 1.0


@dataclass(slots=True)
class DensifyResult:
    ply: Path
    num_points: int | None  # from the tool log
    effective_level: int
    effective_long_side: int
    cuda_device: int
    result: ToolResult


@dataclass(slots=True)
class MeshResult:
    ply: Path
    result: ToolResult


def cuda_device(backend: str, gpu_index: int = 0) -> int:
    """`--cuda-device` value: the GPU index on CUDA, -2 on the CPU (every tool needs it: most default to a GPU)."""
    return gpu_index if backend == "cuda" else CPU_DEVICE


def _flag(value: bool) -> int:
    return int(bool(value))


def _argv(*items: object) -> list[str]:
    return [str(item) for item in items]


def interface_colmap_args(undistorted_rel: str, images_rel: str) -> list[str]:
    return ["-w", ".", "-i", undistorted_rel, "-o", SCENE, "--image-folder", images_rel]


def densify_args(options: OpenMvsOptions, device: int) -> list[str]:
    o = options
    return _argv(
        "-w", ".", "-i", SCENE,
        "--resolution-level", o.resolution_level,
        "--min-resolution", o.min_resolution,
        "--max-resolution", o.max_resolution,
        "--number-views", o.number_views,
        "--number-views-fuse", o.number_views_fuse,
        "--sub-resolution-levels", o.sub_resolution_levels,
        "--iters", o.iters,
        "--geometric-iters", o.geometric_iters,
        "--filter-point-cloud", o.filter_point_cloud,
        "--crop-to-roi", _flag(o.crop_to_roi),
        "--tower-mode", 0,  # no synthetic tower points
        "--remove-dmaps", _flag(not o.keep_dmaps),
        "--cuda-device", device,
    )


def reconstruct_mesh_args(options: MeshOptions, device: int, crop_to_roi: bool = False) -> list[str]:
    o = options
    return _argv(
        "-w", ".", "-i", SCENE_DENSE,
        "--remove-spurious", o.remove_spurious,
        "--remove-spikes", _flag(o.remove_spikes),
        "--close-holes", o.close_holes,
        "--smooth", o.smooth,
        "--decimate", o.decimate,
        "--crop-to-roi", _flag(crop_to_roi),
        "--cuda-device", device,
    )


def effective_long_side(size: int, level: int, min_res: int, max_res: int) -> tuple[int, int]:
    """(long side, level) OpenMVS works at for images with long side `size` (its computeMaxResolution): the size
    halves per level, but a level that would go below `min_res` is raised back to the last one above it."""
    if level == 0:
        return min(size, max_res), 0
    side = size >> level
    if side < min_res:
        level = 0
        while (size >> (level + 1)) >= min_res:
            level += 1
        side = size >> level
    return min(side, max_res), level


def parse_count(log_text: str, pattern: str) -> int | None:
    """The number in the last match of `pattern` (one group), None when it doesn't occur."""
    matches = re.findall(pattern, log_text)
    return int(matches[-1]) if matches else None


def _check_output(tool: str, path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{tool} finished but wrote no {path.name}")
    return path


def import_colmap(runner: ToolRunner, work_dir: Path, undistorted_dir: Path) -> ToolResult:
    """`work_dir/scene.mvs` from the undistorted COLMAP workspace `undistorted_dir`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    rel = Path(os.path.relpath(undistorted_dir, work_dir))
    result = runner.run(INTERFACE_COLMAP, interface_colmap_args(str(rel), str(rel / "images")), work_dir)
    _check_output(INTERFACE_COLMAP, work_dir / SCENE)
    return result


def densify(
    runner: ToolRunner, work_dir: Path, options: OpenMvsOptions, device: int, long_side: int
) -> DensifyResult:
    """Depth maps and their fusion into `work_dir/scene_dense.{mvs,ply}`; `long_side` is the images' long side."""
    side, level = effective_long_side(
        long_side, options.resolution_level, options.min_resolution, options.max_resolution
    )
    log.info("densifying at level %d (long side %d px), --cuda-device %d", level, side, device)
    result = runner.run(DENSIFY, densify_args(options, device), work_dir)
    _check_output(DENSIFY, work_dir / SCENE_DENSE)
    num_points = parse_count(result.log_path.read_text(errors="replace"), DENSE_POINTS_PATTERN)
    return DensifyResult(_check_output(DENSIFY, work_dir / DENSE_PLY), num_points, level, side, device, result)


def reconstruct_mesh(
    runner: ToolRunner, work_dir: Path, options: MeshOptions, device: int, crop_to_roi: bool = False
) -> MeshResult:
    """Delaunay + graph-cut mesh of the dense cloud into `work_dir/scene_dense_mesh.ply`."""
    result = runner.run(RECONSTRUCT_MESH, reconstruct_mesh_args(options, device, crop_to_roi), work_dir)
    return MeshResult(_check_output(RECONSTRUCT_MESH, work_dir / MESH_PLY), result)
