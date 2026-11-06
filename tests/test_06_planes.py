import numpy as np

from recon.mesh.planes import MIN_PAIR_ANGLE_DEG, _axis_angle_deg


def test_opposite_normals_are_parallel():
    up, down, side = np.array([0.0, 0, 1]), np.array([0.0, 0, -1]), np.array([1.0, 0, 0])
    assert _axis_angle_deg(up, down) < MIN_PAIR_ANGLE_DEG
    assert _axis_angle_deg(up, side) >= MIN_PAIR_ANGLE_DEG
