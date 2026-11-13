"""Screened Poisson mesh of an oriented point cloud (x y z nx ny nz), the low-density surface trimmed away:

    pixi run python scripts/poisson_mesh.py --cloud dense_points_filled.ply --out mesh.ply --run RUN

The cloud is read like the dense stage's `dense_points.ply`. The octree depth gives cells of `--cell-tau` times the
run's geometry tolerance over the cloud's extent (a finer octree only reproduces the depth noise); the mesh is cropped
to the sparse bounding box of the run's dense stage."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from recon.io.ply import read_openmvs_points
from recon.pipeline.stages import s05_dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, help="octree depth; default: from --cell-tau")
    parser.add_argument("--cell-tau", type=float, default=1.0, help="finest octree cell in tau (the dense stage's geometry tolerance)")
    parser.add_argument("--trim", type=float, default=0.03, help="share of the lowest-density vertices removed")
    parser.add_argument("--run", type=Path, required=True, help="run directory: tau and the sparse bounding box of its dense stage")
    parser.add_argument("--no-crop", action="store_true", help="keep the mesh outside the sparse bounding box")
    args = parser.parse_args()

    report = json.loads((args.run / s05_dense.DIR_NAME / s05_dense.ARTIFACTS["report"]).read_text())
    lo, hi = np.array(report["sparse_bbox"]["min"]), np.array(report["sparse_bbox"]["max"])
    tau = float(report["fusion"]["tau"])
    t0 = time.perf_counter()
    xyz, normals, _, _ = read_openmvs_points(args.cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    print(f"{len(xyz)} points read in {time.perf_counter() - t0:.0f} s", flush=True)
    depth = args.depth or int(np.ceil(np.log2(float((xyz.max(axis=0) - xyz.min(axis=0)).max()) / (args.cell_tau * tau))))
    t0 = time.perf_counter()
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    density = np.asarray(density)
    print(f"poisson depth {depth}: {len(mesh.triangles)} faces in {time.perf_counter() - t0:.0f} s", flush=True)
    mesh.remove_vertices_by_mask(density < np.quantile(density, args.trim))
    if not args.no_crop:
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()  # trimming and cropping leave some; the mesh_post filters need manifold input
    mesh.remove_unreferenced_vertices()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), mesh)
    print(f"{len(mesh.triangles)} faces after trimming -> {args.out}")


if __name__ == "__main__":
    main()
