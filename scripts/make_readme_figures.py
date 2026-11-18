"""README figures. The report stage renders them for every run (`08_report/figures/`); this copies a run's figures
into `results/<scene>/` for the README, builds the tutorial figure, or compares the dense backends of two runs that
share the aligned model:

    pixi run python scripts/make_readme_figures.py --run outputs/hotel1/runs/<run_id>   # -> results/hotel1/*.png
    pixi run python scripts/make_readme_figures.py --run RUN_2DGS --suffix _2dgs          # -> results/hotel1/*_2dgs.png
    pixi run python scripts/make_readme_figures.py --tutorials                          # results/tutorial_*.png -> results/tutorials.png
    pixi run -e gsplat python scripts/make_readme_figures.py --backends RUN_OPENMVS RUN_2DGS
        # -> results/<scene>/backends_heldout.png, backends_novel.png, hero.png, inputs.png, sfm_path.png,
        #    openmvs.png, 2dgs.png

To make the figures of a finished run again: `pixi run recon run --input data/<scene> --resume <run_id> --only report --force report`.
"""

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from recon.report.figures import compose

RESULTS = Path(__file__).resolve().parents[1] / "results"
TUTORIALS = [("tutorial_1b_epipolar", "1b: epipolar geometry"), ("tutorial_1c_matches", "1c: feature matches")]
HELD_VIEWS = 3  # held-out cameras, evenly spaced over the 2DGS run's held-out list
NOVEL_FRAMES = (0, 1, 2, 5)  # of the 6 frames of the novel-view strip (linspace over the path)
LONG_SIDE = 640
PANEL_W = 260


def backends(run_a: Path, run_b: Path, out: Path) -> list[Path]:
    """Held-out cameras (frame | A mesh | B mesh | B splats) and novel views (A mesh | B mesh | B splats) of an OpenMVS
    run A and a 2DGS run B. Meshes are the textured `model.glb`; needs the gsplat env for the splats."""
    import cv2
    import numpy as np
    import pycolmap

    from recon.dense.gs_render import novel_path, render
    from recon.dense.gs_train import Splats, fit_size, scaled_K
    from recon.io.tables import read_json
    from recon.pipeline.stages import s05_dense as d
    from recon.report.figures import camera_pose, render_poses

    dense = run_b / d.DIR_NAME
    rec = pycolmap.Reconstruction(dense / d.ARTIFACTS["undistorted_model"])
    images = {im.name: im for im in rec.images.values()}
    splats = Splats.load_ply(dense / d.GS_ARTIFACTS["splats"])
    glbs = {tag: run / "07_export/model.glb" for tag, run in (("a", run_a), ("b", run_b))}
    tmp = Path(tempfile.mkdtemp(prefix="backends_"))

    def splat(K: np.ndarray, w2c: np.ndarray, size: tuple[int, int], path: Path) -> Path:
        rgb = render(splats, K, w2c, size)["rgb"].cpu().numpy()
        cv2.imwrite(str(path), (rgb[..., ::-1] * 255).round().astype(np.uint8))
        return path

    held = [v["name"] for v in read_json(dense / d.GS_ARTIFACTS["holdout"])["per_view"]]
    names = [held[i] for i in np.linspace(0, len(held) - 1, HELD_VIEWS).round().astype(int)]
    poses = [camera_pose(rec, n, LONG_SIDE) for n in names]
    renders = {t: render_poses(g, poses, [tmp / f"held_{t}_{k}.png" for k in range(len(names))]) for t, g in glbs.items()}
    splat_tiles = []
    for k, name in enumerate(names):
        im = images[name]
        size = fit_size(im.camera, LONG_SIDE)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = im.cam_from_world().matrix()
        splat_tiles.append(splat(scaled_K(im.camera, *size), w2c, size, tmp / f"held_s_{k}.png"))
    panels = []  # one row per camera, one column per source
    for k, n in enumerate(names):
        panels += [(dense / d.ARTIFACTS["undistorted_images"] / n, "photo (held out)"), (renders["a"][k], "OpenMVS mesh"),
                   (renders["b"][k], "2DGS mesh"), (splat_tiles[k], "2DGS splats")]
    files = [compose(out / "backends_heldout.png", 4, PANEL_W, panels)]

    frames = sorted((dense / d.GS_FIGURES["gs_novel"]).glob("*.jpg"))
    n = len(frames)
    path, size = novel_path(rec, n, max(cv2.imread(str(frames[0])).shape[:2]))
    picks = [int(i) for i in np.linspace(0, n - 1, 6).round().astype(int)[list(NOVEL_FRAMES)]]
    poses = []
    for K, T in (path[i] for i in picks):
        R = T[:3, :3].astype(float)
        poses.append((R, -R.T @ T[:3, 3], float(np.degrees(2 * np.arctan(size[1] / (2 * K[1, 1])))), size))
    renders = {t: render_poses(g, poses, [tmp / f"novel_{t}_{k}.png" for k in range(len(picks))]) for t, g in glbs.items()}
    panels = []  # one row per viewpoint, one column per source
    for k, i in enumerate(picks):
        panels += [(renders["a"][k], "OpenMVS mesh"), (renders["b"][k], "2DGS mesh"), (frames[i], "2DGS splats")]
    files.append(compose(out / "backends_novel.png", 3, PANEL_W, panels))
    shutil.rmtree(tmp)
    return files


HERO_SIZE = (1100, 720)
HERO_FOV = 75.0  # vertical, degrees
HERO_CORNER = (0.06, 0.06)  # camera at this fraction of the room box (x, y) from its min corner ...
HERO_HEIGHT = 0.92  # ... at this share of the ceiling height, looking at the room centre at HERO_TARGET of it
HERO_TARGET = 0.15


def hero(run_a: Path, run_b: Path, out: Path) -> Path:
    """From one viewpoint high in a room corner: the SfM points, the SfM points with the camera path, the OpenMVS
    textured mesh (run A) and the 2DGS splats (run B)."""
    import cv2
    import numpy as np
    import pycolmap

    from recon.dense.gs_render import render
    from recon.dense.gs_train import Splats
    from recon.pipeline.stages import s05_dense as d
    from recon.report.figures import render_poses

    dense = run_b / d.DIR_NAME
    rec = pycolmap.Reconstruction(dense / d.ARTIFACTS["undistorted_model"])  # aligned: Z up, floor at z = 0
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    lo, hi = np.percentile(xyz, 3, axis=0), np.percentile(xyz, 97, axis=0)
    centre = np.array([*(lo[:2] + np.array(HERO_CORNER) * (hi[:2] - lo[:2])), HERO_HEIGHT * hi[2]])
    target = np.array([*(lo[:2] + hi[:2]) / 2, HERO_TARGET * hi[2]])
    f = (target - centre) / np.linalg.norm(target - centre)
    y = np.array([0.0, 0.0, -1.0]) - f * -f[2]
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, f), y, f])  # rows: camera x (right), y (down), z (forward)
    w, h = HERO_SIZE
    tmp = Path(tempfile.mkdtemp(prefix="hero_"))
    mesh = render_poses(run_a / "07_export/model.glb", [(R, centre, HERO_FOV, HERO_SIZE)], [tmp / "mesh.png"])[0]
    fl = h / 2 / np.tan(np.radians(HERO_FOV) / 2)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3], w2c[:3, 3] = R, -R @ centre
    K = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]], np.float32)
    rgb = render(Splats.load_ply(dense / d.GS_ARTIFACTS["splats"]), K, w2c, HERO_SIZE)["rgb"].cpu().numpy()
    cv2.imwrite(str(tmp / "splats.png"), (rgb[..., ::-1] * 255).round().astype(np.uint8))

    # SfM points drawn far to near, then the time-ordered centres of the main camera as a path
    cam = w2c[:3, :3] @ xyz.T + w2c[:3, 3:]
    uv = (K @ cam)[:2] / cam[2]
    keep = (cam[2] > 0.05) & (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
    colors = np.array([p.color for p in rec.points3D.values()])[keep][:, ::-1]
    order = np.argsort(-cam[2, keep])
    sheet = np.full((h, w, 3), 255, np.uint8)
    for (u, v), c in zip(uv[:, keep].T[order], colors[order]):
        cv2.circle(sheet, (int(u), int(v)), 2, tuple(int(x) for x in c), -1, cv2.LINE_AA)
    cv2.imwrite(str(tmp / "sfm.png"), sheet)
    images = sorted((im for im in rec.images.values() if im.camera_id == Counter(i.camera_id for i in rec.images.values()).most_common(1)[0][0]), key=lambda im: im.name)
    centres = np.array([im.projection_center() for im in images])
    cc = w2c[:3, :3] @ centres.T + w2c[:3, 3:]
    puv = ((K @ cc)[:2] / np.maximum(cc[2], 1e-6)).T
    for k in range(len(puv) - 1):
        if cc[2, k] > 0.05 and cc[2, k + 1] > 0.05:
            t = k / (len(puv) - 1)  # blue (start) to red (end)
            cv2.line(sheet, tuple(puv[k].astype(int)), tuple(puv[k + 1].astype(int)), (int(255 * (1 - t)), 60, int(255 * t)), 3, cv2.LINE_AA)
    for k in range(0, len(images), 12):  # viewing direction of every 12th camera
        tip = centres[k] + 0.6 * images[k].cam_from_world().rotation.matrix()[2]
        a, b = (w2c[:3, :3] @ np.stack([centres[k], tip]).T + w2c[:3, 3:]).T
        if a[2] > 0.05 and b[2] > 0.05:
            pa, pb = ((K @ np.stack([a, b], 1))[:2] / np.array([a[2], b[2]])).T
            if np.linalg.norm(pb - pa) > 60:  # too close to the viewpoint
                continue
            cv2.arrowedLine(sheet, tuple(pa.astype(int)), tuple(pb.astype(int)), (40, 40, 40), 2, cv2.LINE_AA, tipLength=0.3)
    if cc[2, 0] > 0.05:
        cv2.circle(sheet, tuple(puv[0].astype(int)), 9, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(tmp / "path.png"), sheet)
    path = compose(out / "hero.png", 2, 700, [
        (mesh, "OpenMVS: textured mesh"), (tmp / "splats.png", "2DGS: splat render"),
        (tmp / "sfm.png", "SfM: sparse points"), (tmp / "path.png", "SfM: camera path (blue start, red end)")])
    shutil.rmtree(tmp)
    return path


def _trim(path: Path, pad: int = 6):
    """The image without its near-white border."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    ys, xs = np.nonzero((np.asarray(image) < 245).any(axis=2))
    if not len(xs):
        return image
    return image.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad + 1, image.width),
                       min(ys.max() + pad + 1, image.height)))


def tight(out: Path, rows: list[list[tuple[Path, str]]], width: int = 1400) -> Path:
    """Rows of titled panels, each trimmed of white border; a row's panels share one height and fill `width`."""
    from PIL import Image, ImageDraw, ImageFont

    gap = 6
    font = ImageFont.load_default(16)
    sheets = []
    for row in rows:
        images = [_trim(path) for path, _ in row]
        title_h = 26 if any(title for _, title in row) else 0
        inner = width - gap * (len(images) - 1)
        height = round(inner / sum(im.width / im.height for im in images))
        sheet = Image.new("RGB", (width, height + title_h), (255, 255, 255))
        draw, x = ImageDraw.Draw(sheet), 0
        for image, (_, title) in zip(images, row):
            tile = image.resize((round(image.width * height / image.height), height), Image.LANCZOS)
            sheet.paste(tile, (x, title_h))
            draw.text((x + 4, 4), title, fill=(30, 30, 30), font=font)
            x += tile.width + gap
        sheets.append(sheet)
    canvas = Image.new("RGB", (width, sum(sh.height for sh in sheets) + gap * (len(sheets) - 1)), (255, 255, 255))
    y = 0
    for sheet in sheets:
        canvas.paste(sheet, (0, y))
        y += sheet.height + gap
    canvas.save(out, optimize=True)
    return out


def sections(run_a: Path, run_b: Path, out: Path) -> list[Path]:
    """Tightly packed README figures from the report panels: inputs and SfM (shared), the SfM path from above, and
    the dense results of the OpenMVS run A and the 2DGS run B."""
    from PIL import Image

    a, b = (run / "08_report/figures/panels" for run in (run_a, run_b))
    oblique = Path(tempfile.mkdtemp(prefix="sections_")) / "oblique.png"
    cameras = Image.open(a / "sfm_cameras.png")
    cameras.crop((0, 0, round(cameras.width * 0.53), cameras.height)).save(oblique)  # the left plot; the view from above is sfm_path
    return [
        tight(out / "inputs.png", [[(a / "input_frames.png", "input: video keyframes and stills"),
                                    (oblique, "SfM: sparse points and cameras")]]),
        tight(out / "sfm_path.png", [[(a / "sfm_path.png", "")]]),
        tight(out / "openmvs.png", [[(a / "dense_cloud.png", "dense point cloud"),
                                     (a / "mesh_geometry.png", "mesh after plane snapping and gap filling"),
                                     (a / "textured_mesh.png", "textured model")]]),
        tight(out / "2dgs.png", [[(b / "gs_holdout.png", "held-out photo | splat render"),
                                  (b / "mesh_geometry.png", "mesh after plane snapping"),
                                  (b / "textured_mesh.png", "textured model")]]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="run directory whose 08_report/figures are copied to results/<scene>/")
    parser.add_argument("--suffix", default="", help="--run: appended to the copied file names, e.g. _2dgs")
    parser.add_argument("--tutorials", action="store_true", help="build only the tutorial figure")
    parser.add_argument("--backends", type=Path, nargs=2, metavar=("RUN_OPENMVS", "RUN_2DGS"),
                        help="build the backend comparison figures of two runs with the same aligned model")
    args = parser.parse_args()
    if args.tutorials:
        print(compose(RESULTS / "tutorials.png", 1, 900, [(RESULTS / f"{stem}.png", title) for stem, title in TUTORIALS]))
        return
    if args.backends:
        out = RESULTS / args.backends[1].parents[1].name
        out.mkdir(parents=True, exist_ok=True)
        print(*backends(*args.backends, out), hero(*args.backends, out), *sections(*args.backends, out), sep="\n")
        return
    if args.run is None:
        parser.error("give --run RUN_DIR, --tutorials or --backends RUN_OPENMVS RUN_2DGS")
    scene = args.run.parents[1].name
    out = RESULTS / scene
    out.mkdir(parents=True, exist_ok=True)
    for figure in sorted((args.run / "08_report/figures").glob("*.png")):
        print(shutil.copy(figure, out / f"{figure.stem}{args.suffix}{figure.suffix}"))


if __name__ == "__main__":
    main()
