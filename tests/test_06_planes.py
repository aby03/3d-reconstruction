import numpy as np

from recon.mesh.planes import MIN_PAIR_ANGLE_DEG, _axis_angle_deg, _offset_peaks


def test_opposite_normals_are_parallel():
    up, down, side = np.array([0.0, 0, 1]), np.array([0.0, 0, -1]), np.array([1.0, 0, 0])
    assert _axis_angle_deg(up, down) < MIN_PAIR_ANGLE_DEG
    assert _axis_angle_deg(up, side) >= MIN_PAIR_ANGLE_DEG


def test_offset_peak_with_empty_window_is_skipped():
    # the peak bin sees the area at 2.4, but the ±tau window around the bin centre holds none
    assert _offset_peaks(np.array([0.0, 2.4]), np.array([0.0, 1.0]), tau=1.0, min_area=1.0) == []


def test_offset_peak_with_fewer_bins_than_the_window():
    peaks = _offset_peaks(np.array([0.0, 0.6]), np.array([1.0, 1.0]), tau=1.0, min_area=1.0)  # 2 bins
    assert np.allclose(peaks, [0.3])
