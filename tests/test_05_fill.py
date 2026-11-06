import numpy as np

from recon.dense.fill import FillOptions, FillView, fill_points

K = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])


def view(x: float) -> FillView:
    """A camera at x looking down +z at the plane z = 2, 100x100 px."""
    T = np.eye(4)
    T[0, 3] = -x
    return FillView(np.full((100, 100), 2.0, np.float32), K, T, np.full((100, 100, 3), 200, np.uint8))


def test_only_uncovered_surface_is_added():
    xs, ys = np.meshgrid(np.linspace(-0.9, 0, 120), np.linspace(-0.9, 0.9, 240))
    dense = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)], axis=1)  # the plane for x < 0 only
    points, normals, rgb, report = fill_points(dense, [view(0.0), view(0.1)], diag=4.0, tau=0.1, options=FillOptions())
    assert report["added"] == len(points) > 100
    assert (points[:, 0] > 0.25).all()  # nothing where the dense cloud is
    assert np.allclose(points[:, 2], 2.0, atol=1e-6)
    assert (normals[:, 2] < 0).all()  # towards the cameras
    assert (rgb == 200).all()


def test_depth_seen_by_one_view_is_dropped():
    points, *_ = fill_points(np.zeros((0, 3)), [view(0.0)], diag=4.0, tau=0.1, options=FillOptions())
    assert len(points) == 0
