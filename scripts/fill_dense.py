"""Mono depth points where the dense cloud of a run has none, written next to a copy of the cloud:

    pixi run python scripts/fill_dense.py --run outputs/room2_tiny/runs/<run_id> --out <dir>

Writes `fill_points.ply` (the added points), `dense_points_filled.ply` (dense cloud and added points) and
`dense_points_filled_debug.ply` (added points in magenta) into `--out` and prints how much of the SfM points the
cloud covers before and after. The run is not modified."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial import cKDTree

from recon.dense.fill import FillOptions, FillView, fill_points
from recon.dense.fusion import FusionOptions
from recon.io.ply import read_openmvs_points, write_points
from recon.pipeline.stages import s05_dense


def load_views(dense: Path, min_inlier_ratio: float, max_residual: float) -> list[FillView]:
    fits = pd.read_csv(dense / s05_dense.MONO_ARTIFACTS["mono_fit"])
    views = s05_dense.load_mono_views(dense, fits, min_inlier_ratio, max_residual)
    good = fits[fits["fitted"] & (fits["inlier_ratio"] >= min_inlier_ratio) & (fits["median_rel_residual"] <= max_residual)]
    images = dense / s05_dense.ARTIFACTS["undistorted_images"]
    out = []
    for name, (depth, K, T) in zip(sorted(good["image"]), views):
        bgr = cv2.imread(str(images / name))
        rgb = cv2.cvtColor(cv2.resize(bgr, depth.shape[::-1], interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        out.append(FillView(depth, K, T, rgb))
    return out


def main() -> None:
    d = FillOptions()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for f in fields(FillOptions):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(getattr(d, f.name)), default=getattr(d, f.name))
    args = parser.parse_args()
    options = FillOptions(**{f.name: getattr(args, f.name) for f in fields(FillOptions)})

    dense = args.run / s05_dense.DIR_NAME
    report = json.loads((dense / s05_dense.ARTIFACTS["report"]).read_text())
    lo, hi = np.array(report["sparse_bbox"]["min"]), np.array(report["sparse_bbox"]["max"])
    diag, tau = float(np.linalg.norm(hi - lo)), float(report["fusion"]["tau"])
    fusion = FusionOptions()

    xyz, normals, rgb, _ = read_openmvs_points(dense / s05_dense.ARTIFACTS["dense_points"])
    views = load_views(dense, fusion.min_inlier_ratio, fusion.max_residual)
    added, added_normals, added_rgb, rep = fill_points(xyz, views, diag, tau, options)
    print(f"{len(views)} views, {rep['depth_points']} depth points, {rep['multi_view']} voxels with >= {options.min_views} views, "
          f"{rep['added']} added to {len(xyz)} dense points ({rep['added'] / len(xyz):.1%})")

    args.out.mkdir(parents=True, exist_ok=True)
    write_points(args.out / "fill_points.ply", added, added_normals, added_rgb)
    write_points(args.out / "dense_points_filled.ply", np.concatenate([xyz, added]), np.concatenate([normals, added_normals]), np.concatenate([rgb, added_rgb]))
    magenta = np.tile(np.array([255, 0, 255], np.uint8), (len(added), 1))
    write_points(args.out / "dense_points_filled_debug.ply", np.concatenate([xyz, added]), np.concatenate([normals, added_normals]), np.concatenate([rgb, magenta]))

    rec = pycolmap.Reconstruction(dense / s05_dense.ARTIFACTS["undistorted_model"])
    sparse = np.array([p.xyz for p in rec.points3D.values()])
    reach = options.radius_tau * tau
    near = lambda cloud: np.isfinite(cKDTree(cloud).query(sparse, distance_upper_bound=reach, workers=-1)[0])
    before = near(xyz)
    after = near(np.concatenate([xyz, added])) if len(added) else before
    print(f"SfM points within {options.radius_tau:g} tau of the cloud: {before.mean():.1%} -> {after.mean():.1%} ({len(sparse)} points)")
    only = ~before & after
    if only.any():
        dist = cKDTree(added).query(sparse[only], workers=-1)[0]
        print(f"{only.sum()} SfM points are covered only by added points, median distance {np.median(dist) / tau:.2f} tau")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
