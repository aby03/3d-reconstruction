"""Hole share of every dense and mesh_post geometry of a run, as the keyframes see it: the share of reference-depth
pixels (the fitted mono depth by default) where a view sees a surface and the geometry has nothing at or in front
of it. Writes `hole_share.json` and `hole_masks.png` (red: hole) into `--out`:

    pixi run python scripts/hole_share.py --run outputs/hotel1/runs/<run_id> --out <dir>
    pixi run python scripts/hole_share.py --run RUN --depth OTHER_NPZ_DIR --only mesh_final_geometry
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from recon.mesh.coverage import CoverageOptions, hole_share, mask_montage, reference_views
from recon.pipeline.stages import s05_dense, s06_mesh_post

# in pipeline order; missing files are skipped
STAGES = [
    (s05_dense.DIR_NAME, s05_dense.ARTIFACTS, "dense_points"),
    (s05_dense.DIR_NAME, s05_dense.ARTIFACTS, "mesh_raw"),
    (s05_dense.DIR_NAME, s05_dense.ARTIFACTS, "mesh_refined"),
    (s05_dense.DIR_NAME, s05_dense.FUSION_ARTIFACTS, "mesh_tsdf"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_merged"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_clean"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_filled"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_holes_filled"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_consistent"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_regularized"),
    (s06_mesh_post.DIR_NAME, s06_mesh_post.ARTIFACTS, "mesh_final_geometry"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument("--out", type=Path, help="output directory (default: RUN/hole_share)")
    parser.add_argument("--depth", type=Path, help="reference depth npz dir (default: RUN/05_dense/mono)")
    parser.add_argument("--only", nargs="+", help="geometry names to test (default: all present)")
    parser.add_argument("--every", type=int, default=CoverageOptions.every, help="every n-th view")
    parser.add_argument("--scale", type=float, default=CoverageOptions.scale, help="depth map resize factor")
    parser.add_argument("--depth-rel", type=float, default=CoverageOptions.depth_rel)
    parser.add_argument("--rows", type=int, default=6, help="views in the mask montage")
    args = parser.parse_args()

    t0 = time.perf_counter()
    dense = args.run / s05_dense.DIR_NAME
    depth_dir = args.depth or dense / s05_dense.MONO_ARTIFACTS["mono"]
    out = args.out or args.run / "hole_share"
    options = CoverageOptions(every=args.every, scale=args.scale, depth_rel=args.depth_rel)
    sources = {name: args.run / d / a[name] for d, a, name in STAGES if (args.run / d / a[name]).is_file()}
    if args.only:
        sources = {k: v for k, v in sources.items() if k in args.only}

    # reference depth files mirror the image names: <depth_dir>/<sequence>/<camera>/<frame>.npz
    views = reference_views(dense / s05_dense.ARTIFACTS["undistorted_model"], depth_dir, options)
    rows = min(args.rows, len(views))
    picks = sorted({round(i * (len(views) - 1) / max(rows - 1, 1)) for i in range(rows)})
    shares, masks = hole_share(views, sources, options, keep=picks)
    images_dir = dense / s05_dense.ARTIFACTS["undistorted_images"]
    images = [cv2.imread(str(images_dir / views[i].name)) for i in picks]
    mask_montage(images, masks, out / "hole_masks.png")

    result = {
        "run": str(args.run),
        "reference": str(depth_dir),
        "views": len(views),
        "options": vars(options),
        "columns": ["image", *sources],
        "hole_share": shares,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "hole_share.json").write_text(json.dumps(result, indent=2))
    width = max(map(len, shares), default=0)
    for name, share in shares.items():
        print(f"{name:<{width}}  {100 * share:5.1f}%")
    print(f"{len(views)} views, {result['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
