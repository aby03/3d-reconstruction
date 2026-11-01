"""Rotation and similarity helpers. Quaternions are xyzw (scipy order); rotation vectors are axis * angle in radians."""

from dataclasses import dataclass, field
from typing import Any

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


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between vectors `a` and `b` (..., 3), row-wise; they need not be unit length."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cos = (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation (3, 3) with `R @ a` parallel to `b`; a half turn about an axis normal to `a` when they
    point in opposite directions."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    sin, cos = float(np.linalg.norm(axis)), float(a @ b)
    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        normal = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return so3_exp(np.pi * normal / np.linalg.norm(normal))
    return so3_exp(axis / sin * np.arctan2(sin, cos))


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane `n . x + d = 0` through `points` (N, 3), N >= 3; returns the unit normal and d."""
    points = np.asarray(points, dtype=float)
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    n = vt[-1]
    return n, float(-n @ centre)


@dataclass(slots=True)
class Sim3:
    """Similarity `x -> scale * R @ x + t`."""

    scale: float = 1.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Transform a point (3,) or the rows of (N, 3)."""
        return self.scale * np.asarray(x, dtype=float) @ self.R.T + self.t

    def __matmul__(self, other: "Sim3") -> "Sim3":
        """Composition: `(a @ b).apply(x) == a.apply(b.apply(x))`."""
        return Sim3(self.scale * other.scale, self.R @ other.R, self.scale * self.R @ other.t + self.t)

    def inverse(self) -> "Sim3":
        R_inv = self.R.T
        return Sim3(1.0 / self.scale, R_inv, -R_inv @ self.t / self.scale)

    def matrix(self) -> np.ndarray:
        """(4, 4) homogeneous matrix [scale * R | t]."""
        M = np.eye(4)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M

    def to_dict(self) -> dict[str, Any]:
        return {"scale": float(self.scale), "R": self.R.tolist(), "t": self.t.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sim3":
        R, t = np.asarray(data["R"], dtype=float), np.asarray(data["t"], dtype=float)
        return cls(float(data.get("scale", 1.0)), R, t)
