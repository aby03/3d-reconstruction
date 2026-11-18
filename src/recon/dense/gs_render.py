"""Render trained 2DGS splats: held-out metrics, TSDF mesh of the rendered depth, centres as points, novel views."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from recon.dense import GsOptions
from recon.dense.fusion import FusionOptions, FusionView, fuse
from recon.dense.gs_train import C0, Splats, View, fit_size, rasterize, scaled_K, ssim

LIFT_FRAC = 0.1  # novel path: lifted by this share of the room height
PITCH_DEG = 10.0  # ... and pitched down by this angle
VIEWS_PER_CONTROL = 6  # novel path: one Catmull-Rom control camera per this many frames


def _to(splats: Splats, device: str) -> Splats:
    """The splats with every tensor on `device`."""
    return Splats(*(t.to(device) for t in (splats.means, splats.quats, splats.scales, splats.opacities, splats.sh)))


def render(
    splats: Splats, K: np.ndarray, w2c: np.ndarray, size: tuple[int, int], depth: str = "median", device: str = "cuda"
) -> dict[str, Any]:
    """One view (no grad) as tensors on `device`: rgb (H,W,3) in [0,1], depth (H,W), normal (H,W,3) world, alpha (H,W)."""
    import torch

    w, h = size
    with torch.no_grad():
        Kt, Tt = (torch.as_tensor(np.asarray(a, np.float32), device=device) for a in (K, w2c))
        out, alpha, normals, _, _, median, _ = rasterize(
            _to(splats, device), Kt, Tt, w, h, splats.sh_degree, render_mode="RGB+ED")
    d = median[0, ..., 0] if depth == "median" else out[0, ..., 3]
    return {"rgb": out[0, ..., :3].clamp(0, 1), "depth": d, "normal": normals[0], "alpha": alpha[0, ..., 0]}


def holdout_metrics(splats: Splats, views: list[View], device: str = "cuda") -> dict[str, Any]:
    """PSNR / SSIM of the held-out views at training size: means and one {name, psnr, ssim} per view."""
    import torch

    splats, per_view = _to(splats, device), []
    for v in (v for v in views if v.holdout):
        rgb = render(splats, v.K, v.viewmat, v.size, device=device)["rgb"]
        gt = torch.from_numpy(v.image).to(device).float() / 255
        psnr = float(-10 * torch.log10(((rgb - gt) ** 2).mean()))
        per_view.append({"name": v.name, "psnr": psnr, "ssim": float(ssim(rgb, gt))})
    mean = lambda k: float(np.mean([p[k] for p in per_view])) if per_view else float("nan")
    return {"psnr": mean("psnr"), "ssim": mean("ssim"), "per_view": per_view}


def fuse_mesh(
    splats: Splats, views: list[View], fusion: FusionOptions, tau: float, diag: float, gs: GsOptions,
    device: str = "cuda",
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """TSDF (`fusion.fuse`) of the `gs.depth` depth of every view at `gs.render_long_side`, alpha < `gs.min_alpha` cut."""
    t0 = time.perf_counter()
    splats, fviews = _to(splats, device), []
    for v in views:
        w, h = v.size
        s = gs.render_long_side / max(w, h)
        rw, rh = round(w * s), round(h * s)
        K = v.K.astype(np.float64).copy()
        K[0] *= rw / w
        K[1] *= rh / h
        r = render(splats, K, v.viewmat, (rw, rh), gs.depth, device)
        d = (r["depth"] * (r["alpha"] >= gs.min_alpha)).cpu().numpy().astype(np.float32)
        fviews.append(FusionView(d, K, v.viewmat.astype(np.float64)))
    t_render = time.perf_counter() - t0
    vertices, faces, _, report = fuse(fviews, diag, tau, fusion)
    report |= {"tau": tau, "render_s": round(t_render, 2), "fuse_s": round(time.perf_counter() - t0 - t_render, 2)}
    return trimesh.Trimesh(vertices, faces, process=False), report


def _rotmat(quats: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices of (N,4) wxyz quaternions (normalised here)."""
    w, x, y, z = (quats / np.linalg.norm(quats, axis=1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def _centre(w2c: np.ndarray) -> np.ndarray:
    """Camera centre in the world of a 4x4 cam_from_world."""
    return -w2c[:3, :3].T @ w2c[:3, 3]


def splat_points(
    splats: Splats, views: list[View], min_opacity: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centres, surfel normals (towards the nearest camera) and SH DC colours (uint8) of splats above `min_opacity`."""
    keep = 1 / (1 + np.exp(-np.asarray(splats.opacities, np.float64))) > min_opacity
    xyz = np.asarray(splats.means, np.float32)[keep]
    normals = _rotmat(np.asarray(splats.quats, np.float64)[keep])[:, :, 2]
    centres = np.array([_centre(v.viewmat) for v in views])
    to_cam = centres[cKDTree(centres).query(xyz)[1]] - xyz
    normals *= np.where((normals * to_cam).sum(1) < 0, -1, 1)[:, None]
    rgb = (np.asarray(splats.sh, np.float32)[keep, 0] * C0 + 0.5).clip(0, 1) * 255
    return xyz, normals.astype(np.float32), rgb.round().astype(np.uint8)


def _catmull_rom(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Uniform Catmull-Rom through the control points `p` (K,3) at parameters `t` in [0, K-1]."""
    q = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    i = np.minimum(t.astype(int), len(p) - 2)
    u = (t - i)[:, None]
    p0, p1, p2, p3 = q[i], q[i + 1], q[i + 2], q[i + 3]
    return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2 + (3 * p1 - p0 - 3 * p2 + p3) * u**3)


def novel_path(
    rec: pycolmap.Reconstruction, n: int, long_side: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]]:
    """n (K, w2c) along the time-ordered centres of the main camera, lifted and pitched down; and the (w, h) of K."""
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    cam_id = Counter(im.camera_id for im in images).most_common(1)[0][0]
    images = [im for im in images if im.camera_id == cam_id]
    cam = rec.cameras[cam_id]
    size = fit_size(cam, long_side)
    R = np.array([im.cam_from_world().rotation.matrix() for im in images])
    centres = np.array([im.projection_center() for im in images])
    up = -R[:, 1].mean(0)  # camera y points down
    up /= np.linalg.norm(up)
    xyz = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    heights = (xyz if len(xyz) else centres) @ up
    lift = LIFT_FRAC * float(np.subtract(*np.percentile(heights, [98, 2])))
    if n <= 0:
        return [], size
    idx = np.linspace(0, len(images) - 1, int(np.clip(n // VIEWS_PER_CONTROL, 2, len(images)))).round().astype(int)
    t = np.linspace(0, len(idx) - 1, n)
    if len(idx) > 1:
        pos, rots = _catmull_rom(centres[idx], t), Slerp(np.arange(len(idx)), Rotation.from_matrix(R[idx]))(t)
    else:
        pos, rots = np.repeat(centres, n, 0), Rotation.from_matrix(np.repeat(R, n, 0))
    a = np.radians(PITCH_DEG)
    pitch = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])  # forward tilts to +y (down)
    K = scaled_K(cam, *size)
    path = []
    for c, Rw in zip(pos + lift * up, rots.as_matrix()):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = pitch @ Rw
        T[:3, 3] = -T[:3, :3] @ c
        path.append((K, T))
    return path, size


def render_path(
    splats: Splats, path: list[tuple[np.ndarray, np.ndarray]], size: tuple[int, int], out_dir: Path,
    device: str = "cuda",
) -> list[Path]:
    """Render each (K, w2c) of `path` at `size` (the size K belongs to) to `out_dir/NNNN.jpg`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splats, files = _to(splats, device), []
    for i, (K, w2c) in enumerate(path):
        rgb = render(splats, K, w2c, size, device=device)["rgb"]
        files.append(out_dir / f"{i:04d}.jpg")
        cv2.imwrite(str(files[-1]), (rgb.cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8))
    return files
