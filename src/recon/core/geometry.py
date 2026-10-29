"""Rotation helpers. Quaternions are xyzw (scipy order); rotation vectors are axis * angle in radians."""

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw -> (..., 3, 3)."""
    return Rotation.from_quat(q).as_matrix()


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) xyzw."""
    return Rotation.from_matrix(R).as_quat()


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 3, 3) rotation matrix."""
    return Rotation.from_rotvec(rotvec).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 3) rotation vector."""
    return Rotation.from_matrix(R).as_rotvec()


def skew(v: np.ndarray) -> np.ndarray:
    """(3,) -> (3, 3) with skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
