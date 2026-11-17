import numpy as np
import trimesh
from conftest import assert_stage_ok

from recon.mesh.holes import boundary_loops, fill_holes


def test_smoke(smoke_run):
    assert_stage_ok(smoke_run.stage_dir("mesh_post"))
    assert (smoke_run.run_dir / "logs" / "mesh_post" / "TextureMesh.log").is_file()


def test_holes_punched(tmp_path):
    """A hole punched into a closed sphere (the faces around one vertex and its neighbours) is closed."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    ring = [0, *mesh.vertex_neighbors[0]]
    punched = trimesh.Trimesh(mesh.vertices, mesh.faces[~np.isin(mesh.faces, ring).any(axis=1)], process=False)
    punched.remove_unreferenced_vertices()
    punched.export(tmp_path / "punched.ply")
    assert len(boundary_loops(punched)) == 1

    result = fill_holes(tmp_path / "punched.ply", tmp_path / "filled.ply")
    assert result["filled"] == 1
    filled = trimesh.load(tmp_path / "filled.ply", process=False, force="mesh")
    assert boundary_loops(filled) == []


def test_plane_fill_closes_an_enclosed_hole(tmp_path):
    """A square hole in a floor raster is filled with flat faces; nothing is added where there is no hole."""
    from recon.mesh.plane_fill import PlaneFillOptions, fill_plane_holes
    from recon.mesh.planes import Plane

    n = 60
    xs = np.linspace(-5, 5, n + 1)
    X, Y = np.meshgrid(xs, xs)
    vertices = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            if not (25 <= i < 30 and 25 <= j < 30):
                faces += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    src, out = tmp_path / "floor.ply", tmp_path / "filled.ply"
    trimesh.Trimesh(vertices, faces, process=False).export(src)
    floor = [Plane(0, "floor", [0, 0, 1.0], 0.0, 100.0, 1.0, len(faces), 0.0, 0.0, 0.0)]
    report = fill_plane_holes(src, out, floor, [], 0.05, 14.0, PlaneFillOptions(enabled=True))
    filled = trimesh.load(out, process=False, force="mesh")
    assert report["holes"] == report["holes_filled"] == 1 and report["cells_added"] > 100
    assert len(filled.faces) == len(faces) + 2 * report["cells_added"] and np.allclose(filled.vertices[:, 2], 0)


def test_hole_share_counts_a_hole_and_ignores_points_behind(tmp_path):
    """A camera looks at a wall with a hole: the hole share is the hole's share of the view, for the mesh and for
    points sampled on it; geometry behind the reference depth counts as a hole."""
    from recon.mesh.coverage import CoverageOptions, RefView, hole_share

    n, size = 40, 4.0
    xs = np.linspace(-size / 2, size / 2, n + 1)
    X, Y = np.meshgrid(xs, xs)
    vertices = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 2.0)]  # wall at z = 2 in front of a camera at the origin
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            if not (15 <= i < 25 and 15 <= j < 25):  # 1 x 1 hole in the middle
                faces += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    wall = tmp_path / "wall.ply"
    trimesh.Trimesh(vertices, faces, process=False).export(wall)
    far = tmp_path / "far.ply"
    trimesh.Trimesh(vertices + [0, 0, 1.0], faces, process=False).export(far)  # the same wall at z = 3
    points = tmp_path / "points.ply"
    trimesh.PointCloud(trimesh.Trimesh(vertices, faces, process=False).sample(400_000, seed=0)).export(points)

    w = 100  # a 100 px view whose frustum spans the 4 x 4 wall at depth 2: the hole is 1/16 of it
    K = np.array([[w / 2, 0, w / 2], [0, w / 2, w / 2], [0, 0, 1.0]])
    view = RefView("v", np.full((w, w), 2.0, np.float32), K, np.eye(4))
    shares, masks = hole_share([view], {"mesh": wall, "points": points, "far": far}, CoverageOptions(), keep=[0])
    assert abs(shares["mesh"] - 1 / 16) < 0.01
    assert abs(shares["points"] - 1 / 16) < 0.02
    assert shares["far"] == 1.0
    assert masks[0]["mesh"][w // 2, w // 2] and not masks[0]["mesh"][5, 5]
