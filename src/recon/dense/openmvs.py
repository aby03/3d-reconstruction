"""OpenMVS: argv builders for each tool and the tool runs (through a ToolRunner), in one working folder.

Every tool runs with `cwd = work_dir` and `-w .`, so the paths OpenMVS stores in its `.mvs` files are relative
to `work_dir`: the working folder and the undistorted images next to it must not move.
"""

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from recon.io.tools import ToolError, ToolResult, ToolRunner, ToolTimeout

log = logging.getLogger(__name__)

INTERFACE_COLMAP = "InterfaceCOLMAP"
DENSIFY = "DensifyPointCloud"
RECONSTRUCT_MESH = "ReconstructMesh"
REFINE_MESH = "RefineMesh"
TEXTURE_MESH = "TextureMesh"
SCENE = "scene.mvs"
SCENE_DENSE = "scene_dense.mvs"
DENSE_PLY = "scene_dense.ply"  # referenced by scene_dense.mvs
MESH_PLY = "scene_dense_mesh.ply"
REFINED_PLY = "scene_dense_mesh_refine.ply"
PREVIEW_OBJ = "mesh.obj"
CPU_DEVICE = -2  # --cuda-device: -2 CPU, -1 best GPU, >= 0 GPU index
DENSE_POINTS_PATTERN = r"Densifying point-cloud completed: (\d+) points"
OOM_PATTERN = r"CUDA_ERROR_OUT_OF_MEMORY|out of memory|cudaErrorMemoryAllocation"


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
class RefineOptions:
    enabled: bool = True
    use_cuda: bool = False  # True: follows the backend (room2: CPU 16.7 min, CUDA 20.3 min, same mesh)
    resolution_level: int = 1
    max_face_area: int = 16
    scales: int = 2
    regularity_weight: float = 0.2
    max_minutes: float = 45.0  # killed after this; the raw mesh is kept


@dataclass(slots=True)
class PreviewOptions:
    enabled: bool = True
    resolution_level: int = 2
    min_resolution: int = 320


@dataclass(slots=True)
class OomRetry:
    tool: str
    from_level: int | None  # resolution level of the call that ran out of memory (None: the tool has none)
    to_level: int | None
    cpu_fallback: bool  # the retry ran with --cuda-device -2


@dataclass(slots=True)
class DensifyResult:
    ply: Path
    num_points: int | None  # from the tool log
    effective_level: int
    effective_long_side: int
    cuda_device: int  # of the call that succeeded
    oom_retries: list[OomRetry] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


@dataclass(slots=True)
class MeshResult:
    ply: Path | None  # None: the tool was stopped by its time limit
    cuda_device: int
    timed_out: bool = False
    oom_retries: list[OomRetry] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


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


def reconstruct_mesh_args(
    options: MeshOptions, device: int, crop_to_roi: bool = False, points: str | None = None
) -> list[str]:
    o = options
    return _argv(
        "-w", ".", "-i", SCENE_DENSE,
        *(("--pointcloud-file", points) if points else ()),
        "--remove-spurious", o.remove_spurious,
        "--remove-spikes", _flag(o.remove_spikes),
        "--close-holes", o.close_holes,
        "--smooth", o.smooth,
        "--decimate", o.decimate,
        "--crop-to-roi", _flag(crop_to_roi),
        "--cuda-device", device,
    )


def refine_mesh_args(options: RefineOptions, device: int, mesh: str = MESH_PLY) -> list[str]:
    o = options
    return _argv(
        "-w", ".", "-i", SCENE_DENSE, "-m", mesh, "-o", REFINED_PLY,
        "--resolution-level", o.resolution_level,
        "--max-face-area", o.max_face_area,
        "--scales", o.scales,
        "--regularity-weight", o.regularity_weight,
        "--planar-vertex-ratio", 0,
        "--reduce-memory", 1,
        "--cuda-device", device,
    )


def texture_mesh_args(options: PreviewOptions, device: int, mesh: str, out_rel: str) -> list[str]:
    o = options
    return _argv(
        "-w", ".", "-i", SCENE_DENSE, "-m", mesh, "-o", f"{out_rel}/{PREVIEW_OBJ}",
        "--export-type", "obj",
        "--resolution-level", o.resolution_level,
        "--min-resolution", o.min_resolution,
        "--global-seam-leveling", 0,  # seam leveling is slow and not needed for a quick look
        "--local-seam-leveling", 0,
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


def is_oom(log_text: str) -> bool:
    """Whether an OpenMVS log reports a CUDA out-of-memory error."""
    return re.search(OOM_PATTERN, log_text) is not None


def oom_retry_options(options: OpenMvsOptions, long_side: int) -> OpenMvsOptions:
    """Densify options for a retry after a CUDA OOM: one resolution level more (with `min_resolution` lowered so
    the clamp of `effective_long_side` doesn't undo it) and two neighbour views fewer."""
    _, level = effective_long_side(long_side, options.resolution_level, options.min_resolution, options.max_resolution)
    level += 1
    return replace(
        options,
        resolution_level=level,
        min_resolution=min(options.min_resolution, long_side >> level),
        number_views=max(options.number_views - 2, 2),
    )


@dataclass(slots=True)
class _Attempt:
    args: list[str]
    device: int
    retry: OomRetry | None = None  # set on every attempt after the first


def _run_attempts(
    runner: ToolRunner,
    tool: str,
    attempts: list[_Attempt],
    work_dir: Path,
    timeout_s: float | None = None,
    before_retry: Callable[[], None] | None = None,
) -> tuple[_Attempt, list[ToolResult], list[OomRetry]]:
    """Run `attempts` in order until one succeeds; a failed attempt moves on to the next only on a CUDA OOM."""
    results, retries = [], []
    for i, attempt in enumerate(attempts):
        if attempt.retry is not None:
            log.warning(
                "%s ran out of GPU memory, retrying%s", tool, " on the CPU" if attempt.retry.cpu_fallback else ""
            )
            retries.append(attempt.retry)
            if before_retry is not None:
                before_retry()
        try:
            results.append(runner.run(tool, attempt.args, work_dir, timeout_s))
            return attempt, results, retries
        except ToolTimeout:
            raise
        except ToolError as e:
            if e.returncode is not None:  # None: the tool didn't start
                results.append(runner.results[-1])
            if i == len(attempts) - 1 or not is_oom(e.log_tail):
                raise
    raise AssertionError("no attempts")


def _cpu_attempt(args_for: Callable[[int], list[str]], tool: str, level: int | None) -> _Attempt:
    return _Attempt(args_for(CPU_DEVICE), CPU_DEVICE, OomRetry(tool, level, level, cpu_fallback=True))


def check_output(tool: str, path: Path) -> Path:
    """`path`, after checking that `tool` wrote it."""
    if not path.is_file():
        raise RuntimeError(f"{tool} finished but wrote no {path.name}")
    return path


def import_colmap(runner: ToolRunner, work_dir: Path, undistorted_dir: Path) -> ToolResult:
    """`work_dir/scene.mvs` from the undistorted COLMAP workspace `undistorted_dir`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    rel = Path(os.path.relpath(undistorted_dir, work_dir))
    result = runner.run(INTERFACE_COLMAP, interface_colmap_args(str(rel), str(rel / "images")), work_dir)
    check_output(INTERFACE_COLMAP, work_dir / SCENE)
    return result


def densify(
    runner: ToolRunner,
    work_dir: Path,
    options: OpenMvsOptions,
    device: int,
    long_side: int,
    cpu_fallback: bool = False,
) -> DensifyResult:
    """Depth maps and their fusion into `work_dir/scene_dense.{mvs,ply}`; `long_side` is the images' long side.

    On CUDA, an out-of-memory failure is retried once with `oom_retry_options`, then (with `cpu_fallback`) on the
    CPU; the retries are listed in the result."""
    side, level = effective_long_side(
        long_side, options.resolution_level, options.min_resolution, options.max_resolution
    )
    retry = oom_retry_options(options, long_side)
    retry_side, retry_level = effective_long_side(
        long_side, retry.resolution_level, retry.min_resolution, retry.max_resolution
    )
    attempts = [_Attempt(densify_args(options, device), device)]
    if device != CPU_DEVICE:
        attempts.append(_Attempt(densify_args(retry, device), device, OomRetry(DENSIFY, level, retry_level, False)))
        if cpu_fallback:
            attempts.append(_cpu_attempt(lambda d: densify_args(retry, d), DENSIFY, retry_level))
    log.info("densifying at level %d (long side %d px), --cuda-device %d", level, side, device)

    def clear_dmaps() -> None:  # depth maps of the failed call are at another resolution
        for path in work_dir.glob("*.dmap"):
            path.unlink()

    used, results, retries = _run_attempts(runner, DENSIFY, attempts, work_dir, before_retry=clear_dmaps)
    if used.retry is not None:
        side, level = retry_side, retry_level
    check_output(DENSIFY, work_dir / SCENE_DENSE)
    num_points = parse_count(results[-1].log_path.read_text(errors="replace"), DENSE_POINTS_PATTERN)
    ply = check_output(DENSIFY, work_dir / DENSE_PLY)
    return DensifyResult(ply, num_points, level, side, used.device, retries, results)


def run_with_cpu_fallback(
    runner: ToolRunner,
    tool: str,
    argv_for_device: Callable[[int], list[str]],
    work_dir: Path,
    device: int,
    cpu_fallback: bool,
    level: int | None = None,
    timeout_s: float | None = None,
) -> tuple[int, list[ToolResult], list[OomRetry]]:
    """Run a tool on `device` (argv from `argv_for_device(device)`); on a CUDA OOM, once more on the CPU when
    `cpu_fallback` (no lower resolution: these tools need little memory, so an OOM means the GPU is taken).
    Returns the device that succeeded, the tool results and the retries."""
    attempts = [_Attempt(argv_for_device(device), device)]
    if device != CPU_DEVICE and cpu_fallback:
        attempts.append(_cpu_attempt(argv_for_device, tool, level))
    used, results, retries = _run_attempts(runner, tool, attempts, work_dir, timeout_s)
    return used.device, results, retries


def reconstruct_mesh(
    runner: ToolRunner,
    work_dir: Path,
    options: MeshOptions,
    device: int,
    crop_to_roi: bool = False,
    cpu_fallback: bool = False,
    points: str | None = None,
) -> MeshResult:
    """Delaunay + graph-cut mesh of the dense cloud into `work_dir/scene_dense_mesh.ply`; `points` (a PLY with
    views in `work_dir`) replaces the cloud stored in the scene."""
    used, results, retries = run_with_cpu_fallback(
        runner, RECONSTRUCT_MESH, lambda d: reconstruct_mesh_args(options, d, crop_to_roi, points), work_dir, device,
        cpu_fallback,
    )
    return MeshResult(check_output(RECONSTRUCT_MESH, work_dir / MESH_PLY), used, False, retries, results)


def refine_mesh(
    runner: ToolRunner,
    work_dir: Path,
    options: RefineOptions,
    device: int,
    cpu_fallback: bool = False,
    mesh: str = MESH_PLY,
) -> MeshResult:
    """Photometric refinement of the mesh `work_dir/<mesh>` (default `scene_dense_mesh.ply`) into `work_dir/scene_dense_mesh_refine.ply`.
    Stopped after `options.max_minutes`: the result then has `ply=None, timed_out=True`."""
    try:
        used, results, retries = run_with_cpu_fallback(
            runner, REFINE_MESH, lambda d: refine_mesh_args(options, d, mesh), work_dir, device, cpu_fallback,
            options.resolution_level, options.max_minutes * 60,
        )
    except ToolTimeout:
        log.warning("%s stopped after %.0f min", REFINE_MESH, options.max_minutes)
        return MeshResult(None, device, True, [], [runner.results[-1]])
    return MeshResult(check_output(REFINE_MESH, work_dir / REFINED_PLY), used, False, retries, results)


def texture_preview(
    runner: ToolRunner,
    work_dir: Path,
    mesh: str,
    out_dir: Path,
    options: PreviewOptions,
    device: int,
    cpu_fallback: bool = False,
) -> MeshResult:
    """Low-resolution textured `out_dir/mesh.obj` (+ `.mtl` and the atlas) of `mesh` (a file in `work_dir`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rel = os.path.relpath(out_dir, work_dir)
    used, results, retries = run_with_cpu_fallback(
        runner, TEXTURE_MESH, lambda d: texture_mesh_args(options, d, mesh, out_rel), work_dir, device,
        cpu_fallback, options.resolution_level,
    )
    return MeshResult(check_output(TEXTURE_MESH, out_dir / PREVIEW_OBJ), used, False, retries, results)
