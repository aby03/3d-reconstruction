"""Figures of a run: offscreen renders of its outputs (VTK) and the camera path, joined into a few titled figures
(`overview`, `stages`, `path`, `details`, and `gaussians` under dense backend 2dgs) that the report links and the
README uses.

Rendering needs an OpenGL window (X, EGL or OSMesa through VTK) and can crash on a headless machine, so the report runs
`python -m recon.report.figures spec.json` in a subprocess (`build_figures`): a failure loses the figures, never the run.
Every view is derived from the run itself (room box from the mesh and its floor and ceiling planes, camera poses from
the aligned model), nothing is specific to a capture.
"""

import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import vtk
from PIL import Image, ImageDraw, ImageFont
from vtk.util.numpy_support import numpy_to_vtk

from recon.io.colmap import parse_image_name
from recon.io.ply import read_mesh, read_openmvs_points
from recon.report import FigureOptions
from recon.report.plots import plot_cameras_3d, plot_path_topdown

log = logging.getLogger(__name__)

GAP, TITLE_H = 8, 30
FIGURES = {  # figure -> (columns, panel width, [(panel, title)])
    "overview": (2, 700, [("dense_cloud", "Dense fused point cloud"), ("model", "Textured model (model.glb)")]),
    "stages": (
        3,
        560,
        [
            ("input_frames", "input: video keyframes + stills"),
            ("sfm_cameras", "sfm: sparse points, cameras"),
            ("dense_cloud", "dense: fused point cloud"),
            ("mesh_geometry", "mesh_post: flat planes, filled gaps"),
            ("textured_mesh", "export: textured model"),
            ("report", "report: checks and timings"),
        ],
    ),
    "path": (1, 1500, [("sfm_path", "SfM camera path from above, walked from waypoint 1 to the last")]),
    "details": (
        2,
        700,
        [
            ("detail_texture", "keyframe | model from the same camera (two views)"),
            ("detail_mono", "mono depth fitted to the SfM points"),
            ("detail_tracks", "SfM track lengths"),
            ("detail_reprojection", "SfM reprojection error"),
        ],
    ),
    "gaussians": (  # backend 2dgs only
        3,
        600,
        [
            ("gs_novel", "2DGS novel views: RGB over normals, along the walk lifted and pitched down"),
            ("gs_holdout", "2DGS held-out view: ground truth | render"),
            ("gs_mesh", "2DGS mesh: TSDF of the rendered depth (mesh_raw.ply)"),
        ],
    ),
}


# ---- offscreen renders -------------------------------------------------------------------------------------------


def _window(size: tuple[int, int]) -> tuple[vtk.vtkRenderer, vtk.vtkRenderWindow]:
    renderer, window = vtk.vtkRenderer(), vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(*size)
    renderer.SetBackground(1.0, 1.0, 1.0)
    return renderer, window


def _look(
    renderer: vtk.vtkRenderer, bounds: np.ndarray, up: int, azimuth: float, elevation: float, zoom: float,
    clip: float | None,
) -> None:
    """Camera on a sphere around the middle of `bounds` (2x3: min, max); `up` is the vertical axis (1: Y, 2: Z).
    Everything higher than `clip` along the vertical axis is cut away (the ceiling, to see into the room)."""
    lo, hi = bounds
    centre = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))
    axes = [a for a in range(3) if a != up]
    az, el = np.radians(azimuth), np.radians(elevation)
    offset = np.zeros(3)
    offset[axes[0]], offset[axes[1]], offset[up] = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*(centre + offset * 0.85 * diag / zoom))
    view_up = np.zeros(3)
    view_up[up] = 1.0
    camera.SetViewUp(*view_up)
    camera.SetClippingRange(0.01 * diag, 6 * diag)
    if clip is not None:
        plane = vtk.vtkPlane()
        origin, normal = np.zeros(3), np.zeros(3)
        origin[up], normal[up] = clip, -1.0
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)
        actors = renderer.GetActors()
        actors.InitTraversal()
        for _ in range(actors.GetNumberOfItems()):
            actors.GetNextActor().GetMapper().AddClippingPlane(plane)


def _save(window: vtk.vtkRenderWindow, path: Path) -> Path:
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()
    return path


def _glb_scene(glb: Path, size: tuple[int, int]) -> tuple[vtk.vtkRenderer, vtk.vtkRenderWindow]:
    renderer, window = _window(size)
    importer = vtk.vtkGLTFImporter()
    importer.SetFileName(str(glb))
    importer.SetRenderWindow(window)
    importer.Update()
    return renderer, window


def render_glb(glb: Path, path: Path, bounds: np.ndarray, size: tuple[int, int], **view: Any) -> Path:
    """The glb (Y up) from outside; `bounds` is the room box in the glb frame."""
    renderer, window = _glb_scene(glb, size)
    _look(renderer, bounds, 1, **view)
    return _save(window, path)


Pose = tuple[np.ndarray, np.ndarray, float, tuple[int, int]]  # cam_from_world R, centre, vertical fov (deg), (w, h)


def camera_pose(rec: pycolmap.Reconstruction, name: str, height: int) -> Pose:
    """The pose of the registered image `name`, at `height` pixels."""
    image = next(im for im in rec.images.values() if im.name == name)
    model = image.camera
    fov = float(np.degrees(2 * np.arctan(model.height / (2 * model.mean_focal_length()))))
    size = (round(height * model.width / model.height), height)
    return image.cam_from_world().rotation.matrix(), image.projection_center(), fov, size


def render_poses(glb: Path, poses: list[Pose], paths: list[Path]) -> list[Path]:
    """The glb (imported once) seen from each pose of the model's Z up frame, which becomes the glb's Y up:
    (x, y, z) -> (x, z, -y)."""
    renderer, window = _glb_scene(glb, poses[0][3])
    to_glb = lambda v: np.array([v[0], v[2], -v[1]])  # noqa: E731
    camera = renderer.GetActiveCamera()
    for (R, centre, fov, size), path in zip(poses, paths):
        window.SetSize(*size)
        camera.SetPosition(*to_glb(centre))
        camera.SetFocalPoint(*to_glb(centre + R[2]))
        camera.SetViewUp(*to_glb(-R[1]))
        camera.SetViewAngle(fov)
        camera.SetClippingRange(0.02, 200.0)
        _save(window, path)
    return paths


def render_from_camera(glb: Path, rec: pycolmap.Reconstruction, name: str, path: Path, height: int) -> Path:
    """The glb seen from the pose of the registered image `name`."""
    return render_poses(glb, [camera_pose(rec, name, height)], [path])[0]


def render_mesh(ply: Path, path: Path, bounds: np.ndarray, size: tuple[int, int], **view: Any) -> Path:
    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(ply))
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.78, 0.8, 0.84)
    actor.GetProperty().SetAmbient(0.25)
    renderer, window = _window(size)
    renderer.AddActor(actor)
    _look(renderer, bounds, 2, **view)
    return _save(window, path)


def render_points(
    ply: Path, path: Path, bounds: np.ndarray, size: tuple[int, int], count: int, **view: Any
) -> Path:
    xyz, _, rgb, _ = read_openmvs_points(ply)
    keep = np.random.default_rng(0).choice(len(xyz), min(count, len(xyz)), replace=False)
    xyz, rgb = xyz[keep].astype(np.float32), np.asarray(rgb)[keep]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(points)
    vertices = vtk.vtkVertexGlyphFilter()
    vertices.SetInputData(cloud)
    vertices.Update()
    out = vertices.GetOutput()
    colors = numpy_to_vtk(np.ascontiguousarray(rgb[:, :3]), deep=True)
    colors.SetName("rgb")
    out.GetPointData().SetScalars(colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(out)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(1.6)
    renderer, window = _window(size)
    renderer.AddActor(actor)
    _look(renderer, bounds, 2, **view)
    return _save(window, path)


# ---- the run's own geometry ------------------------------------------------------------------------------------


def room_box(run: Path) -> tuple[np.ndarray, float | None]:
    """(min, max) of the room in the aligned Z up frame and the height of the ceiling plane (None without one): x and
    y from the final mesh without its outer 1%, z from the floor plane to the ceiling plane (or the mesh's 1st to 99th
    percentile when a plane is missing, so a stray piece of geometry does not stretch the view)."""
    mesh_dir = run / "06_mesh_post"
    vertices = np.asarray(read_mesh(mesh_dir / "mesh_final_geometry.ply").vertices)
    lo, hi = np.percentile(vertices, [1, 99], axis=0)
    ceiling = None
    planes_file = mesh_dir / "planes.json"
    if planes_file.is_file():
        planes = json.loads(planes_file.read_text())
        floor = next((-p["d"] if p["normal"][2] > 0 else p["d"] for p in planes if p["label"] == "floor"), None)
        top = next((-p["d"] for p in planes if p["label"] == "ceiling"), None)
        lo[2] = lo[2] if floor is None else floor
        if top is not None:
            hi[2], ceiling = top, top
    return np.array([lo, hi]), ceiling


def _main_camera_names(rec: pycolmap.Reconstruction) -> list[str]:
    """Names of the registered images of the largest camera group (the video), in name order."""
    names = sorted(rec.images[i].name for i in rec.reg_image_ids())
    groups = [parse_image_name(n)[1] for n in names]
    main = max(set(groups), key=groups.count)
    return [n for n, g in zip(names, groups) if g == main]


# ---- picture assembly ------------------------------------------------------------------------------------------


def montage(images: list[Path], path: Path, columns: int = 3, width: int = 640) -> Path:
    tiles = []
    for image in images:
        im = Image.open(image).convert("RGB")
        tiles.append(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS))
    rows = [tiles[i : i + columns] for i in range(0, len(tiles), columns)]
    heights = [max(t.height for t in row) for row in rows]
    sheet = Image.new("RGB", (columns * width, sum(heights)), (255, 255, 255))
    y = 0
    for row, height in zip(rows, heights):
        for k, tile in enumerate(row):
            sheet.paste(tile, (k * width, y))
        y += height
    sheet.save(path)
    return path


def side_by_side(images: list[Path], path: Path, height: int = 640) -> Path:
    tiles = []
    for image in images:
        im = Image.open(image).convert("RGB")
        tiles.append(im.resize((round(im.width * height / im.height), height), Image.LANCZOS))
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + GAP * (len(tiles) - 1), height), (255, 255, 255))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + GAP
    sheet.save(path)
    return path


def _panel(path: Path, width: int) -> Image.Image:
    if path.is_file():
        image = Image.open(path).convert("RGB")
        return image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    image = Image.new("RGB", (width, round(width * 9 / 16)), (200, 200, 200))
    ImageDraw.Draw(image).text((width // 2 - 60, image.height // 2), "not available", fill=(90, 90, 90), font=ImageFont.load_default(20))
    return image


def compose(out: Path, columns: int, width: int, panels: list[tuple[Path, str]]) -> Path:
    """A grid of titled panels scaled to one width, so a few small images carry many views."""
    images = [_panel(path, width) for path, _ in panels]
    rows = [images[i : i + columns] for i in range(0, len(images), columns)]
    titles = [[t for _, t in panels[i : i + columns]] for i in range(0, len(panels), columns)]
    heights = [max(im.height for im in row) for row in rows]
    canvas = Image.new(
        "RGB", (columns * width + (columns + 1) * GAP, sum(heights) + len(rows) * (TITLE_H + GAP) + GAP), (255, 255, 255)
    )
    draw, font, y = ImageDraw.Draw(canvas), ImageFont.load_default(16), GAP
    for row, height, row_titles in zip(rows, heights, titles):
        for k, (image, title) in enumerate(zip(row, row_titles)):
            x = GAP + k * (width + GAP)
            draw.text((x + 4, y + 6), title, fill=(30, 30, 30), font=font)
            canvas.paste(image, (x, y + TITLE_H))
        y += TITLE_H + height + GAP
    canvas.save(out, optimize=True)
    return out


# ---- the run -----------------------------------------------------------------------------------------------------


def build(run: Path, sequence_dirs: dict[str, Path], out: Path, options: FigureOptions) -> dict[str, Any]:
    """Panels into `out/panels/`, figures into `out/`. A panel that cannot be made is skipped with a warning.
    Returns `{figures: [file names], warnings: [...]}`."""
    from recon.pipeline.stages.common import read_keyframe_inputs  # not at the top: the stages import this module

    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    size = (options.width, options.height)
    export = run / "07_export"
    glb = export / ("model_preview.glb" if (export / "model_preview.glb").is_file() else "model.glb")

    def attempt(name: str, make) -> None:
        t0 = time.perf_counter()
        try:
            make()
            log.info("figure panel %s in %.1f s", name, time.perf_counter() - t0)
        except Exception as error:  # a missing input or a graphics failure costs one panel
            warnings.append(f"figure panel {name} skipped: {type(error).__name__}: {error}")

    box, ceiling = room_box(run)
    clip = None if ceiling is None else ceiling - options.clip_frac * (box[1, 2] - box[0, 2])
    box_glb = np.array([[box[0, 0], box[0, 2], -box[1, 1]], [box[1, 0], box[1, 2], -box[0, 1]]])  # (x, z, -y)
    view = dict(azimuth=options.azimuth, elevation=options.elevation, zoom=options.zoom, clip=clip)
    attempt("model", lambda: render_glb(glb, panels / "model.png", box_glb, size, **{**view, "azimuth": options.azimuth - 95.0}))
    attempt("textured_mesh", lambda: render_glb(glb, panels / "textured_mesh.png", box_glb, size, **view))
    attempt("mesh_geometry", lambda: render_mesh(run / "06_mesh_post/mesh_final_geometry.ply", panels / "mesh_geometry.png", box, size, **view))
    attempt("dense_cloud", lambda: render_points(export / "point_clouds/dense.ply", panels / "dense_cloud.png", box, size, options.max_points, **view))

    sequence_id = next(iter(sequence_dirs))
    inputs = read_keyframe_inputs({sequence_id: sequence_dirs[sequence_id]})
    image_root = inputs.image_dirs[sequence_id]
    rec = pycolmap.Reconstruction(run / "04_alignment/sparse_aligned")

    def pairs() -> None:
        names = _main_camera_names(rec)
        parts = []
        for k, fraction in enumerate((0.3, 0.7)):
            name = names[int(fraction * (len(names) - 1))]
            temp = panels / f"_render_{k}.png"
            render_from_camera(glb, rec, name, temp, options.height)
            parts += [image_root / Path(name).parent.name / Path(name).name, temp]
        side_by_side(parts, panels / "detail_texture.png")
        for k in range(2):
            (panels / f"_render_{k}.png").unlink(missing_ok=True)

    attempt("detail_texture", pairs)

    def frames() -> None:
        cameras = sorted(p.name for p in image_root.iterdir() if p.is_dir())
        video = sorted((image_root / cameras[0]).glob("*.jpg"))
        picks = [video[i] for i in np.linspace(0, len(video) - 1, 4).astype(int)]
        for camera in cameras[1:]:
            stills = sorted((image_root / camera).glob("*.jpg"))
            picks += [stills[len(stills) // 4], stills[3 * len(stills) // 4]] if len(stills) >= 2 else []
        montage(picks, panels / "input_frames.png")

    attempt("input_frames", frames)

    images = sorted((rec.images[i] for i in rec.reg_image_ids()), key=lambda im: im.name)
    centres = np.array([im.projection_center() for im in images])
    forward = np.array([im.cam_from_world().rotation.matrix()[2] for im in images])  # camera z axis in the world
    groups = [f"camera {im.camera_id}" for im in images]
    points = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    t_ns = dict(zip(inputs.images["image_name"], inputs.images["t_ns"].astype(int)))
    times = [t_ns[im.name] * 1e-9 for im in images]
    attempt("sfm_path", lambda: plot_path_topdown(centres, forward, groups, points, panels / "sfm_path.png", times=times, waypoints=options.waypoints))
    attempt("sfm_cameras", lambda: plot_cameras_3d(centres, forward, groups, points, panels / "sfm_cameras.png"))

    def copy(source: Path, name: str) -> None:
        Image.open(source).convert("RGB").save(panels / name)

    for name, rel in (
        ("report.png", "08_report/plots/timings.png"),
        ("detail_mono.png", "05_dense/reports/plots/mono_example.png"),
        ("detail_tracks.png", "03_sfm/reports/plots/track_len_hist.png"),
        ("detail_reprojection.png", "03_sfm/reports/plots/reproj_hist.png"),
    ):
        attempt(name, lambda name=name, rel=rel: copy(run / rel, name))

    dense = run / "05_dense"
    if (dense / "gs" / "splats.ply").is_file():
        attempt("gs_novel", lambda: copy(dense / "gs/novel_strip.png", "gs_novel.png"))
        attempt("gs_holdout", lambda: copy(dense / "gs/holdout_pair.jpg", "gs_holdout.png"))
        attempt("gs_mesh", lambda: render_mesh(dense / "mesh_raw.ply", panels / "gs_mesh.png", box, size, **view))

    made = []
    for name, (columns, width, parts) in FIGURES.items():
        if any((panels / f"{stem}.png").is_file() for stem, _ in parts):
            compose(out / f"{name}.png", columns, width, [(panels / f"{stem}.png", title) for stem, title in parts])
            made.append(f"{name}.png")
    return {"figures": made, "warnings": warnings}


def build_figures(run: Path, sequence_dirs: dict[str, Path], out: Path, options: FigureOptions) -> dict[str, Any]:
    """`build` in a subprocess (a graphics failure must not take the run with it); returns its result, or the failure
    as a warning."""
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "_spec.json"
    result = out / "_result.json"
    spec.write_text(
        json.dumps(
            {"run": str(run), "out": str(out), "sequence_dirs": {k: str(v) for k, v in sequence_dirs.items()}, "options": asdict(options)}
        )
    )
    result.unlink(missing_ok=True)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "recon.report.figures", str(spec)], capture_output=True, text=True, timeout=options.timeout_s
        )
        if process.returncode != 0 or not result.is_file():
            tail = (process.stderr or process.stdout).strip().splitlines()[-3:]
            return {"figures": [], "warnings": [f"figures skipped: rendering exited with status {process.returncode}: {' | '.join(tail)}"]}
        return json.loads(result.read_text())
    except subprocess.TimeoutExpired:
        return {"figures": [], "warnings": [f"figures skipped: rendering took more than {options.timeout_s:.0f} s"]}
    finally:
        spec.unlink(missing_ok=True)
        result.unlink(missing_ok=True)


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    outcome = build(
        Path(spec["run"]), {k: Path(v) for k, v in spec["sequence_dirs"].items()}, Path(spec["out"]), FigureOptions(**spec["options"])
    )
    (Path(spec["out"]) / "_result.json").write_text(json.dumps(outcome))


if __name__ == "__main__":
    main()
