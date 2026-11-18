"""Compare the dense backends of two runs that share the aligned model (e.g. OpenMVS vs 2DGS): time, hole share,
geometry and QA, mesh renders, textured renders at held-out cameras with PSNR against the frame, novel views and the
`recon compare` table. Writes `<out>/index.html` and its PNGs; whatever a run (e.g. an older one) lacks shows as n/a:

    pixi run python scripts/compare_dense.py --a RUN_OPENMVS --b RUN_2DGS --out <dir> [--views 24] [--skip-holes]
"""

import argparse
import hashlib
import json
import shutil
import time
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from matplotlib import colormaps
from matplotlib.figure import Figure

from recon.dense.gs_render import novel_path
from recon.mesh.coverage import CoverageOptions, hole_share, mask_montage, reference_views
from recon.pipeline.stages import s05_dense as d
from recon.pipeline.stages.s05_gs import STRIP_FRAMES
from recon.report import FigureOptions
from recon.report.figures import camera_pose, render_mesh, render_poses, room_box, side_by_side
from recon.report.html import STYLE, _fmt, _table
from recon.report.summary import compare

NA = '<span class="na">n/a</span>'
TEXTURED_VIEWS, HOLE_ROWS, HEIGHT = 4, 6, 360
HOLDOUT_EVERY = 8  # held-out cameras when no run has gs/holdout.json


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def dig(data: dict, *keys: str):
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def cell(value) -> str:
    return NA if value is None else escape(_fmt(value))


def model_hash(run: Path) -> str | None:
    """sha1 of the aligned model's files."""
    files = sorted((run / "04_alignment" / "sparse_aligned").glob("*"))
    if not files:
        return None
    sha = hashlib.sha1()
    for f in files:
        sha.update(f.name.encode() + f.read_bytes())
    return sha.hexdigest()


class Run:
    def __init__(self, path: Path, tag: str):
        self.dir, self.tag, self.dense_dir = path, tag, path / d.DIR_NAME
        self.dense = load(self.dense_dir / d.ARTIFACTS["report"])
        self.mesh = load(path / "06_mesh_post" / "reports" / "mesh_report.json")
        self.summary = load(path / "08_report" / "summary.json")
        self.holdout = load(self.dense_dir / d.GS_ARTIFACTS["holdout"])
        self.backend = self.dense.get("dense_backend") or ("openmvs" if self.dense else None)
        self.label = f"{tag}: {self.backend or 'n/a'} ({path.name})"
        self.glb = path / "07_export" / "model.glb"

    def metric(self, stage: str, key: str):
        return next((dig(e, "metrics", key) for e in self.summary.get("stages", []) if e["stage"] == stage), None)

    def durations(self) -> dict[str, float]:
        """Stage seconds from the summary, else from the run's stage manifests."""
        if self.summary:
            out: dict[str, float] = {}
            for e in self.summary["stages"]:
                out[e["stage"]] = out.get(e["stage"], 0.0) + (e.get("duration_s") or 0.0)
            return out
        manifests = (load(p) for p in sorted(self.dir.glob("0*_*/stage_manifest.json")))
        return {m["stage"]: m.get("duration_s") or 0.0 for m in manifests}


def table(rows, runs, header=("",)) -> str:
    return _table([(escape(k), *(cell(v) for v in vals)) for k, *vals in rows], (*header, *(r.label for r in runs)), raw=True)


def na_tile(path: Path, size: tuple[int, int]) -> Path:
    tile = np.full((size[1], size[0], 3), 200, np.uint8)
    cv2.putText(tile, "n/a", (size[0] // 2 - 30, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90, 90, 90), 2)
    cv2.imwrite(str(path), tile)
    return path


def img(name: str, alt: str = "") -> str:
    return f'<a href="{escape(name)}"><img src="{escape(name)}" alt="{escape(alt or name)}"></a>'


def time_section(runs: list[Run], out: Path) -> str:
    timings = [r.dense.get("timings") or {} for r in runs]
    keys = list(dict.fromkeys(k for t in timings for k in t))
    if keys:
        fig = Figure(figsize=(11, 2.4))
        ax, left = fig.subplots(), np.zeros(len(runs))
        colours = [colormaps["tab20"](i % 20) for i in range(len(keys))]
        for key, colour in zip(keys, colours):
            widths = np.array([t.get(key, 0.0) / 60 for t in timings])
            ax.barh([r.label for r in runs], widths, left=left, label=key, color=colour)
            left += widths
        for y, total in enumerate(left):
            ax.text(total, y, f" {total:.1f} min", va="center")
        ax.invert_yaxis()
        ax.set_xlabel("dense stage [min]")
        ax.set_xlim(0, left.max() * 1.15 or 1)
        ax.legend(ncol=2, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(out / "time_dense.png", dpi=100)
    durations = [r.durations() for r in runs]
    stages = list(dict.fromkeys(s for t in durations for s in t))
    rows = [(s, *(t.get(s) for t in durations)) for s in stages]
    rows.append(("whole run", *(sum(t.values()) if t else None for t in durations)))
    steps = [(f"dense: {k}", *(t.get(k) for t in timings)) for k in keys]
    return (img("time_dense.png") if keys else f"<p>dense timings: {NA}</p>") + table(rows, runs, ("stage [s]",)) \
        + table(steps, runs, ("dense step [s]",))


def holes_section(runs: list[Run], out: Path, count: int) -> str:
    ref = next((r for r in runs if (r.dense_dir / d.MONO_ARTIFACTS["mono"]).is_dir()), None)
    if ref is None:
        return f"<p>no mono depth in either run: {NA}</p>"
    depth_dir = ref.dense_dir / d.MONO_ARTIFACTS["mono"]
    every = max(1, len(list(depth_dir.rglob("*.npz"))) // count)
    options = CoverageOptions(every=every)
    views = reference_views(ref.dense_dir / d.ARTIFACTS["undistorted_model"], depth_dir, options)
    views = [views[i] for i in sorted(set(np.linspace(0, len(views) - 1, min(count, len(views))).round().astype(int)))]
    names = {"mesh_raw": "raw", "mesh_final_geometry": "final"}
    paths = {(r.tag, n): r.dir / (d.DIR_NAME if n == "mesh_raw" else "06_mesh_post") / f"{n}.ply" for r in runs for n in names}
    sources = {f"{tag} {names[n]}": p for (tag, n), p in paths.items() if p.is_file()}
    picks = sorted(set(np.linspace(0, len(views) - 1, min(HOLE_ROWS, len(views))).round().astype(int)))
    shares, masks = hole_share(views, sources, options, keep=picks)
    images = [cv2.imread(str(ref.dense_dir / d.ARTIFACTS["undistorted_images"] / views[i].name)) for i in picks]
    if sources:
        mask_montage(images, masks, out / "hole_masks.png")
    rows = [(n, *(shares.get(f"{r.tag} {short}") for r in runs)) for n, short in names.items()]
    rows = [(k, *(None if v is None else f"{100 * v:.1f} %" for v in vals)) for k, *vals in rows]
    note = f"<p>{len(views)} views, reference: mono depth of {escape(ref.label)} (red: hole)</p>"
    return note + table(rows, runs, ("hole share",)) + (img("hole_masks.png") if sources else "")


def geometry_section(runs: list[Run]) -> str:
    rows = [
        ("dense points", [r.metric("dense", "num_dense_points") or r.dense.get("num_dense_points") for r in runs]),
        ("mesh_raw faces", [dig(r.dense, "mesh_raw", "faces") for r in runs]),
        ("final faces", [dig(r.mesh, "final", "faces") for r in runs]),
        ("final components", [dig(r.mesh, "final", "components") for r in runs]),
        ("largest component", [dig(r.mesh, "final", "largest_component_frac") for r in runs]),
        ("QA unsupported area", [dig(r.mesh, "qa", "unsupported_area_frac") for r in runs]),
        ("QA p95 / diag", [dig(r.mesh, "qa", "p95_frac") for r in runs]),
        ("planes", [dig(r.mesh, "planes", "count") for r in runs]),
        ("wall rms max / diag", [r.metric("mesh_post", "wall_rms_max_frac") for r in runs]),
        ("untextured faces", [r.metric("mesh_post", "untextured_face_frac") for r in runs]),
        ("Gaussians", [dig(r.dense, "gs", "num_gaussians") for r in runs]),
        ("2DGS held-out PSNR (splats)", [dig(r.dense, "gs", "holdout_psnr") for r in runs]),
    ]
    return table([(k, *v) for k, v in rows], runs)


def mesh_section(runs: list[Run], out: Path) -> str:
    box_run = next(r for r in runs if (r.dir / "06_mesh_post/mesh_final_geometry.ply").is_file())
    box, ceiling = room_box(box_run.dir)
    o = FigureOptions()
    height = box[1, 2] - box[0, 2]  # <= 0: planes mislabelled (tiny scenes), no ceiling cut
    clip = None if ceiling is None or height <= 0 else ceiling - o.clip_frac * height
    size, parts = (o.width // 2, o.height // 2), []
    for k, (az, el) in enumerate(((o.azimuth, o.elevation), (o.azimuth + 120, 30.0), (o.azimuth + 240, 30.0))):
        tiles = []
        for r in runs:
            ply, tile = r.dir / "06_mesh_post/mesh_final_geometry.ply", out / f"_mesh_{r.tag}.png"
            view = dict(azimuth=az, elevation=el, zoom=o.zoom, clip=clip)
            tiles.append(render_mesh(ply, tile, box, size, **view) if ply.is_file() else na_tile(tile, size))
        side_by_side(tiles, out / f"mesh_{k}.png", HEIGHT)
        parts.append(img(f"mesh_{k}.png"))
    return f"<p>mesh_final_geometry.ply, left {escape(runs[0].label)}, right {escape(runs[1].label)}</p>" + "".join(parts)


def psnr(render: Path, truth: Path) -> float:
    a = cv2.imread(str(render)).astype(np.float32) / 255
    b = cv2.resize(cv2.imread(str(truth)), a.shape[1::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return float(-10 * np.log10(max(np.mean((a - b) ** 2), 1e-10)))


def glb_renders(r: Run, poses: list, stems: list[str], out: Path) -> list[Path]:
    """The run's model.glb from each pose (one import), or n/a tiles."""
    paths = [out / f"_{r.tag}_{s}.png" for s in stems]
    if r.glb.is_file() and poses:
        return render_poses(r.glb, poses, paths)
    return [na_tile(p, pose[3]) for p, pose in zip(paths, poses)]


def appearance_sections(runs: list[Run], out: Path) -> tuple[str, str]:
    """Textured renders at held-out cameras (with PSNR) and novel views along the 2DGS path."""
    cam_run = next(r for r in runs if (r.dense_dir / d.ARTIFACTS["undistorted_model"]).is_dir())
    urec = pycolmap.Reconstruction(cam_run.dense_dir / d.ARTIFACTS["undistorted_model"])
    splat_psnr = {v["name"]: v["psnr"] for r in runs for v in r.holdout.get("per_view", [])}
    names = list(splat_psnr) or sorted(im.name for im in urec.images.values())[::HOLDOUT_EVERY]
    names = [names[i] for i in sorted(set(np.linspace(0, len(names) - 1, min(TEXTURED_VIEWS, len(names))).round().astype(int)))]
    poses = [camera_pose(urec, n, HEIGHT) for n in names]

    gs_run = next((r for r in runs if (r.dense_dir / d.GS_FIGURES["gs_novel"]).is_dir()), runs[-1])
    frames = sorted((gs_run.dense_dir / d.GS_FIGURES["gs_novel"]).glob("*.jpg"))
    n, long_side = (len(frames), max(cv2.imread(str(frames[0])).shape[:2])) if frames else (120, 640)
    gs_dir = gs_run.dense_dir / d.ARTIFACTS["undistorted_model"]
    path, size = novel_path(pycolmap.Reconstruction(gs_dir) if gs_dir.is_dir() else urec, n, long_side)
    picks = np.linspace(0, n - 1, min(STRIP_FRAMES, n)).round().astype(int)
    for K, T in (path[i] for i in picks):
        R = T[:3, :3].astype(float)
        poses.append((R, -R.T @ T[:3, 3], float(np.degrees(2 * np.arctan(size[1] / (2 * K[1, 1])))), size))

    stems = [f"held_{k}" for k in range(len(names))] + [f"novel_{k}" for k in range(len(picks))]
    renders = {r.tag: glb_renders(r, poses, stems, out) for r in runs}
    rows, parts = [], []
    for k, name in enumerate(names):
        truth = cam_run.dense_dir / d.ARTIFACTS["undistorted_images"] / name
        tiles = [renders[r.tag][k] for r in runs]
        side_by_side([truth, *tiles], out / f"textured_{k}.png", HEIGHT)
        parts.append(img(f"textured_{k}.png", name))
        rows.append((name, *(psnr(t, truth) if r.glb.is_file() else None for r, t in zip(runs, tiles)), splat_psnr.get(name)))
    header = ("camera", *(f"{r.label} textured PSNR" for r in runs), "2DGS splats PSNR (training size)")
    textured = (f"<p>frame | {' | '.join(escape(r.label) for r in runs)} (model.glb, {HEIGHT} px)</p>"
                + _table([(escape(k), *map(cell, v)) for k, *v in rows], header, raw=True) + "".join(parts))

    novel = []
    for r in runs:
        side_by_side(renders[r.tag][len(names):], out / f"novel_{r.tag}.png", size[1])
        novel.append(f"<h3>{escape(r.label)}: model.glb along the path</h3>" + img(f"novel_{r.tag}.png"))
    strip = gs_run.dense_dir / d.GS_FIGURES["gs_novel_strip"]
    if strip.is_file():
        shutil.copy(strip, out / "novel_2dgs.png")
        novel.insert(0, f"<h3>{escape(gs_run.label)}: 2DGS splats (RGB over normals)</h3>" + img("novel_2dgs.png"))
    else:
        novel.insert(0, f"<h3>2DGS splats</h3><p>{NA}</p>")
    return textured, "".join(novel)


def compare_section(runs: list[Run]) -> str:
    a, b = (r.summary for r in runs)
    if not (a and b):
        return f"<p>{NA} (08_report/summary.json missing)</p>"
    rows = [(r["stage"], r["metric"], r["a"], r["b"], r["diff"]) for r in compare(a, b)]
    return _table(rows, ("stage", "metric", runs[0].label, runs[1].label, "diff"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="first run (e.g. OpenMVS)")
    parser.add_argument("--b", type=Path, required=True, help="second run (e.g. 2DGS)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--views", type=int, default=24, help="hole share: views, evenly spaced")
    parser.add_argument("--skip-holes", action="store_true", help="leave out the hole share (the slowest part)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    hashes = [model_hash(run) for run in (args.a, args.b)]
    if None in hashes or hashes[0] != hashes[1]:
        raise SystemExit(f"refusing: the aligned models differ or are missing ({hashes}); views need the same cameras")
    runs = [Run(args.a, "A"), Run(args.b, "B")]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    timed: dict[str, float] = {}

    def section(title: str, make) -> str:
        t = time.perf_counter()
        try:
            body = make()
        except Exception as error:  # an older run without some file costs one section, not the page
            body = f"<p>{NA} ({escape(type(error).__name__)}: {escape(str(error))})</p>"
        timed[title] = time.perf_counter() - t
        print(f"{title}: {timed[title]:.1f} s")
        return body if isinstance(body, tuple) else (body,)

    sections = [("time", *section("time", lambda: time_section(runs, out)))]
    holes = (f"<p>{NA} (--skip-holes)</p>",) if args.skip_holes else section("holes", lambda: holes_section(runs, out, args.views))
    sections += [("holes", *holes), ("geometry and QA", *section("geometry", lambda: geometry_section(runs))),
                 ("mesh renders", *section("mesh", lambda: mesh_section(runs, out)))]
    appearance = section("textured + novel", lambda: appearance_sections(runs, out))
    sections += [("textured at held-out cameras", appearance[0]), ("novel views", appearance[-1] if len(appearance) > 1 else appearance[0]),
                 ("recon compare", *section("compare", lambda: compare_section(runs)))]
    for tmp in out.glob("_*.png"):
        tmp.unlink()
    head = _table([(r.label, str(r.dir)) for r in runs]) + f"<p>aligned model sha1 {hashes[0][:12]} (same for both)</p>"
    body = "\n".join(f"<h2>{escape(t)}</h2>\n" + "\n".join(parts) for t, *parts in sections)
    page = (f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>dense backends</title><style>{STYLE}</style>'
            f"</head>\n<body>\n<h1>dense backends: {escape(runs[0].label)} vs {escape(runs[1].label)}</h1>\n{head}\n{body}\n</body></html>\n")
    (out / "index.html").write_text(page)
    print(f"{time.perf_counter() - t0:.1f} s -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
