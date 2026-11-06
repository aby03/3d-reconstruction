import numpy as np

from recon.dense.mono import MonoOptions
from recon.dense.simplerecon import (
    SimpleReconOptions,
    confidence,
    network_size,
    point_agreement,
    rotate_view,
    scale_pose,
    search_scale,
    source_views,
)


def _pose(x: float, yaw: float = 0.0) -> np.ndarray:
    """cam_from_world of a camera at (x, 0, 0) turned by `yaw` about y."""
    c, s = np.cos(yaw), np.sin(yaw)
    world_from_cam = np.eye(4)
    world_from_cam[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    world_from_cam[0, 3] = x
    return np.linalg.inv(world_from_cam)


def test_rotate_view_keeps_the_projection_of_a_point():
    K = np.array([[500.0, 0, 300], [0, 520, 400], [0, 0, 1]])
    T = _pose(0.3, 0.2)
    image = np.zeros((800, 600, 3), np.uint8)
    point = np.array([0.4, -0.2, 3.0, 1.0])
    x = K @ (T @ point)[:3]
    u, v = x[:2] / x[2]
    rotated, K2, T2 = rotate_view(image, K, T)
    x2 = K2 @ (T2 @ point)[:3]
    u2, v2 = x2[:2] / x2[2]
    assert rotated.shape == (600, 800, 3)
    np.testing.assert_allclose((u2, v2), (800 - v, u))


def test_source_views_prefer_the_optimal_baseline_and_skip_the_reference():
    options = SimpleReconOptions(num_sources=2, optimal_baseline_m=0.15)
    poses = np.stack([_pose(x) for x in [0.0, 0.01, 0.15, 0.3, 2.0]])
    assert source_views(poses, 0, 1.0, options) == [2, 3]
    # the same path in model units that are 10 times a metre: the baselines are measured in metres
    scaled = np.stack([_pose(10 * x) for x in [0.0, 0.01, 0.15, 0.3, 2.0]])
    assert source_views(scaled, 0, 10.0, options) == [2, 3]


def test_scale_pose_divides_the_translation_only():
    T = _pose(3.0, 0.4)
    out = scale_pose(T, 2.0)
    np.testing.assert_allclose(out[:3, :3], T[:3, :3])
    np.testing.assert_allclose(out[:3, 3], T[:3, 3] / 2.0)


def test_network_size_is_a_multiple_of_32_with_the_image_aspect():
    options = SimpleReconOptions(long_side=672)
    assert network_size(1043, 1857, options) == (384, 672)
    assert network_size(1857, 1043, options) == (672, 384)


def test_confidence_drops_depth_jumps_and_invalid_pixels():
    depth = np.full((20, 20), 2.0, np.float32)
    depth[:, 10:] = 4.0
    valid = np.ones_like(depth, bool)
    valid[0] = False
    conf = confidence(depth, valid, 0.05)
    assert conf[5, 3] == 1 and conf[5, 15] == 1
    assert conf[5, 9] == 0 and conf[5, 10] == 0
    assert conf[0].sum() == 0


def test_point_agreement_scores_depth_against_sfm_points():
    depth = np.full((40, 40), 3.0, np.float32)
    uv = np.stack(np.meshgrid(np.arange(5, 35, 2), np.arange(5, 35, 2)), -1).reshape(-1, 2).astype(float)
    options = MonoOptions()
    fit, ratio = point_agreement(depth, uv, np.full(len(uv), 3.0), options)
    assert fit.fitted and fit.inlier_ratio == 1.0 and fit.median_rel_residual == 0.0 and ratio == 1.0
    fit, ratio = point_agreement(depth, uv, np.full(len(uv), 2.0), options)  # depth 50% too far
    assert not fit.fitted and abs(ratio - 1.5) < 1e-9


def _valley(true: float):
    """A score with its minimum at `true` that fits only near it."""

    def score(s: float) -> tuple[float, float]:
        err = abs(np.log(s / true))
        return err, float(err < 0.3)

    return score


def test_search_scale_finds_the_minimum_from_a_start_off_by_a_factor_of_two():
    options = SimpleReconOptions()
    best, curve = search_scale(_valley(5.3), 5.3 * 2.2, options)
    assert abs(np.log(best / 5.3)) < 0.1
    assert len(curve) >= 11 and curve == sorted(curve, key=lambda c: c["scale"])


def test_search_scale_shifts_the_grid_when_the_best_is_at_its_edge():
    options = SimpleReconOptions()
    best, curve = search_scale(_valley(5.3), 5.3 * 5.0, options)  # the grid reaches 0.4x of the start = 2x too high
    assert abs(np.log(best / 5.3)) < 0.1
    assert len(curve) > 11
    best, _ = search_scale(_valley(5.3), 5.3 * 100, options)  # out of reach: the grid stops after `search_expand` shifts
    assert best > 5.3 * 3
