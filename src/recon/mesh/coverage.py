"""Hole share of a mesh or point cloud as the keyframes see it: the share of reference-depth pixels where a view sees
a surface and the geometry has nothing at or in front of it."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap


@dataclass
class RefView:
    """One keyframe: reference depth (0: no reference), intrinsics at the depth map's size, 4x4 cam_from_world."""

    name: str
    depth: np.ndarray
    K: np.ndarray
    T: np.ndarray


@dataclass
class CoverageOptions:
    every: int = 4  # every n-th view with a reference depth
    scale: float = 0.5  # depth maps are resized by this factor before the test
    depth_rel: float = 1.25  # geometry farther than this times the reference depth does not count
    max_points: int = 4_000_000  # point clouds are subsampled (seeded) to this many points


def reference_views(model_dir: Path, depth_dir: Path, options: CoverageOptions | None = None) -> list[RefView]:
    """Reference depths `depth_dir/<image>.npz` (`depth`, and `confidence`: 0 drops the pixel) of the PINHOLE model
    in `model_dir`, every `options.every`-th in name order, resized by `options.scale`."""
    options = options or CoverageOptions()
    rec = pycolmap.Reconstruction(model_dir)
    views = []
    images = [im for im in sorted(rec.images.values(), key=lambda im: im.name) if (depth_dir / Path(im.name).with_suffix(".npz")).is_file()]
    for image in images[:: options.every]:
        data = np.load(depth_dir / Path(image.name).with_suffix(".npz"))
        depth = data["depth"].astype(np.float32)
        if "confidence" in data:
            depth[data["confidence"] == 0] = 0.0
        h, w = round(depth.shape[0] * options.scale), round(depth.shape[1] * options.scale)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        cam = rec.cameras[image.camera_id]
        fx, fy, cx, cy = cam.params
        sx, sy = w / cam.width, h / cam.height
        T = np.eye(4)
        T[:3] = image.cam_from_world().matrix()
        views.append(RefView(image.name, depth, np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]), T))
    return views


def _rays(view: RefView) -> np.ndarray:
    """(H*W, 6) rays through the pixel centres; the direction has unit z in the camera, so a hit's t is its depth."""
    h, w = view.depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    K, R, t = view.K, view.T[:3, :3], view.T[:3, 3]
    dirs = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ R  # R^T d
    origin = np.broadcast_to(-R.T @ t, dirs.shape)
    return np.concatenate([origin, dirs], -1).reshape(-1, 6).astype(np.float32)


def mesh_depth(scene: o3d.t.geometry.RaycastingScene, view: RefView) -> np.ndarray:
    """Depth of the first mesh hit per pixel (inf: none)."""
    hit = scene.cast_rays(o3d.core.Tensor(_rays(view)))["t_hit"].numpy()
    return hit.reshape(view.depth.shape)


def points_depth(points: np.ndarray, view: RefView) -> np.ndarray:
    """Depth of the nearest point per pixel (inf: none)."""
    h, w = view.depth.shape
    cam = points @ view.T[:3, :3].T + view.T[:3, 3]
    front = cam[:, 2] > 1e-6
    cam = cam[front]
    px = (cam[:, :2] / cam[:, 2:]) @ view.K[:2, :2].T + view.K[:2, 2]
    x, y = np.floor(px[:, 0]).astype(int), np.floor(px[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full((h, w), np.inf)
    np.minimum.at(out, (y[inside], x[inside]), cam[inside, 2])
    return out


def hole_mask(depth: np.ndarray, view: RefView, depth_rel: float) -> np.ndarray:
    """Pixels with a reference depth where `depth` (geometry) is missing or beyond `depth_rel` times it."""
    ref = view.depth > 0
    return ref & ~(depth <= depth_rel * view.depth)


def _ply_faces(path: Path) -> int:
    """The face count in a PLY header (0: none)."""
    with path.open("rb") as f:
        for line in f:
            words = line.split()
            if words[:2] == [b"element", b"face"]:
                return int(words[2])
            if words[:1] == [b"end_header"]:
                return 0
    return 0


def load_geometry(path: Path, max_points: int, seed: int = 0) -> o3d.t.geometry.RaycastingScene | np.ndarray:
    """A raycasting scene for a PLY with faces, or the (N, 3) points (subsampled to `max_points`) for one without."""
    if _ply_faces(path):
        mesh = o3d.t.io.read_triangle_mesh(str(path))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh.vertex.positions.to(o3d.core.float32), mesh.triangle.indices.to(o3d.core.uint32))
        return scene
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if len(points) > max_points:
        points = points[np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))]
    return points


def hole_share(
    views: list[RefView], sources: dict[str, Path], options: CoverageOptions | None = None, keep: Sequence[int] = ()
) -> tuple[dict[str, float], list[dict[str, np.ndarray]]]:
    """Hole share (0-1) of each geometry in `sources` (name -> mesh or point-cloud PLY) over all reference pixels of
    `views`, and the hole masks of the views at the indices `keep` ({name: mask} per kept view, in `keep` order)."""
    options = options or CoverageOptions()
    ref = sum(int((v.depth > 0).sum()) for v in views)
    shares, masks = {}, [{} for _ in keep]
    slot = {index: k for k, index in enumerate(keep)}
    for name, path in sources.items():
        geometry = load_geometry(path, options.max_points)
        holes = 0
        for i, view in enumerate(views):
            depth = points_depth(geometry, view) if isinstance(geometry, np.ndarray) else mesh_depth(geometry, view)
            mask = hole_mask(depth, view, options.depth_rel)
            holes += int(mask.sum())
            if i in slot:
                masks[slot[i]][name] = mask
        shares[name] = holes / ref if ref else float("nan")
    return shares, masks


def mask_montage(images: list[np.ndarray], masks: list[dict[str, np.ndarray]], path: Path) -> Path:
    """A header row of column names, then one row per view: the image, then each mask (red: hole), all at the mask
    size."""
    pad = lambda t: cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # noqa: E731
    h, w = next(iter(masks[0].values())).shape
    header = []
    for name in ["image", *masks[0]]:
        tile = np.full((24, w, 3), 255, np.uint8)
        cv2.putText(tile, name.removeprefix("mesh_"), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        header.append(tile)
    rows = [cv2.hconcat([pad(t) for t in header])]
    for image, view_masks in zip(images, masks):
        row = [cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)]
        for mask in view_masks.values():
            tile = np.full((h, w, 3), 235, np.uint8)
            tile[mask] = (40, 40, 220)  # BGR red
            row.append(tile)
        rows.append(cv2.hconcat([pad(t) for t in row]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.vconcat(rows))
    return path
