"""2D Gaussian Splatting training (gsplat, imported lazily: training needs the gsplat env) with a mono depth prior."""

from __future__ import annotations

import csv
import time
from collections.abc import Collection
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

from recon.dense import GsOptions
from recon.io.ply import read_openmvs_points

C0 = 0.28209479177387814  # SH band-0 constant
LOG_EVERY = 100  # train_log.csv row every n iterations (and the last one)
SH_EVERY = 1000  # one more SH degree every n iterations
DIST_FROM = 0.10  # distortion loss from this share of the iterations
NORMAL_FROM = 0.25  # normal consistency loss from this share
REFINE_START, REFINE_EVERY, REFINE_STOP = 500, 100, 0.8  # MCMC relocation / growth schedule
OPACITY_REG, SCALE_REG = 0.01, 0.01  # MCMC regularisers
INIT_OPACITY, INIT_SCALE = 0.5, 0.5  # init opacity, init scale as a share of the knn distance
FLAT_SCALE = 1e-3  # third scale (ignored by 2DGS, shapes the MCMC noise) as a share of the knn distance
LR = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
MEANS_LR_END = 0.01  # means lr decays exponentially to this share
PLY_FLAT_LOG_SCALE = float(np.log(1e-7))  # scale_2 in the PLY: 3DGS viewers show flat discs
LOG_COLUMNS = ["iter", "loss", "l1", "ssim", "dist", "normal", "mono", "num_gaussians", "psnr_train", "sec", "vram_mb"]


@dataclass(slots=True)
class View:
    """One registered image at the training size: RGB uint8 on the CPU, K at that size, 4x4 cam_from_world."""

    name: str
    image_id: int
    camera_id: int
    K: np.ndarray
    viewmat: np.ndarray
    image: np.ndarray
    holdout: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


@dataclass(slots=True)
class Splats:
    """Raw 2DGS parameters (CPU tensors): means, wxyz quats, log scales (N,2), logit opacities, SH (N,K,3)."""

    means: Any
    quats: Any
    scales: Any
    opacities: Any
    sh: Any

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return round(self.sh.shape[1] ** 0.5) - 1

    def save_ply(self, path: Path) -> Path:
        """Standard 3DGS PLY (x y z nx ny nz f_dc_* f_rest_* opacity scale_0..2 rot_0..3, float32)."""
        means, quats, scales, opacities, sh = (np.asarray(getattr(self, f.name), np.float32) for f in fields(self))
        n, k = sh.shape[:2]
        cols = {"x": means[:, 0], "y": means[:, 1], "z": means[:, 2]} | {f"n{a}": np.zeros(n) for a in "xyz"}
        cols |= {f"f_dc_{c}": sh[:, 0, c] for c in range(3)}
        rest = sh[:, 1:].transpose(0, 2, 1).reshape(n, 3 * (k - 1))  # channel-major, as 3DGS
        cols |= {f"f_rest_{i}": rest[:, i] for i in range(rest.shape[1])}
        cols["opacity"] = opacities
        cols |= {"scale_0": scales[:, 0], "scale_1": scales[:, 1], "scale_2": np.full(n, PLY_FLAT_LOG_SCALE)}
        cols |= {f"rot_{i}": quats[:, i] for i in range(4)}
        data = np.empty(n, np.dtype([(c, "<f4") for c in cols]))
        for c, v in cols.items():
            data[c] = np.asarray(v)
        header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
        header += "".join(f"property float {c}\n" for c in cols) + "end_header\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header.encode("ascii") + data.tobytes())
        return path

    @classmethod
    def load_ply(cls, path: Path) -> Splats:
        """Read a PLY written by `save_ply` (float32: equal to the saved splats up to float32 rounding)."""
        import torch

        buf = Path(path).read_bytes()
        end = buf.index(b"end_header\n") + len(b"end_header\n")
        lines = buf[:end].decode("ascii").splitlines()
        n = int(next(ln for ln in lines if ln.startswith("element vertex")).split()[2])
        names = [ln.split()[2] for ln in lines if ln.startswith("property float")]
        data = np.frombuffer(buf, np.dtype([(c, "<f4") for c in names]), n, end)
        col = lambda *cs: torch.from_numpy(np.stack([data[c] for c in cs], 1).astype(np.float32))
        n_rest = sum(c.startswith("f_rest_") for c in names)
        rest = torch.zeros(n, 0, 3)
        if n_rest:
            rest = col(*[f"f_rest_{i}" for i in range(n_rest)]).reshape(n, 3, n_rest // 3).transpose(1, 2)
        sh = torch.cat([col("f_dc_0", "f_dc_1", "f_dc_2")[:, None], rest], 1)
        return cls(col("x", "y", "z"), col(*[f"rot_{i}" for i in range(4)]), col("scale_0", "scale_1"),
                   col("opacity")[:, 0], sh)

    def save_ckpt(self, path: Path) -> Path:
        """Exact torch checkpoint."""
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({f.name: getattr(self, f.name) for f in fields(self)}, path)
        return path

    @classmethod
    def load_ckpt(cls, path: Path) -> Splats:
        """Read a checkpoint written by `save_ckpt`."""
        import torch

        return cls(**torch.load(path, map_location="cpu"))


def fit_size(cam: pycolmap.Camera, long_side: int) -> tuple[int, int]:
    """(width, height) of the camera's image scaled to `long_side`."""
    s = long_side / max(cam.width, cam.height)
    return round(cam.width * s), round(cam.height * s)


def scaled_K(cam: pycolmap.Camera, width: int, height: int) -> np.ndarray:
    """3x3 K of a PINHOLE camera at the image size (width, height)."""
    fx, fy, cx, cy = cam.params
    sx, sy = width / cam.width, height / cam.height
    return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float32)


def load_views(urec: pycolmap.Reconstruction, image_dir: Path, long_side: int, holdout_every: int) -> list[View]:
    """Every registered image of the undistorted model, by name, at `long_side`; every n-th is held out (0: none)."""
    images = sorted((im for im in urec.images.values() if im.has_pose), key=lambda im: im.name)
    views = []
    for i, im in enumerate(images):
        cam = urec.cameras[im.camera_id]
        w, h = fit_size(cam, long_side)
        bgr = cv2.imread(str(image_dir / im.name))
        if bgr is None:
            raise FileNotFoundError(image_dir / im.name)
        rgb = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
        T = np.eye(4, dtype=np.float32)
        T[:3] = im.cam_from_world().matrix()
        holdout = holdout_every > 0 and i % holdout_every == holdout_every // 2
        views.append(View(im.name, im.image_id, im.camera_id, scaled_K(cam, w, h), T, rgb, holdout))
    return views


def load_mono_depth(
    mono_dir: Path, views: list[View], keep: Collection[str] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(depth float16, valid) at view size from the fitted `<name>.npz` (SfM scale), for views in `keep` (None: all)."""
    out = {}
    for v in views:
        path = mono_dir / Path(v.name).with_suffix(".npz")
        if (keep is not None and v.name not in keep) or not path.is_file():
            continue
        data = np.load(path)
        depth = np.where(data["confidence"] > 0, data["depth"].astype(np.float32), 0)
        depth = cv2.resize(depth, v.size, interpolation=cv2.INTER_NEAREST)
        if (valid := depth > 0).any():
            out[v.name] = (depth.astype(np.float16), valid)
    return out


def init_points(
    rec: pycolmap.Reconstruction, views: list[View], mono_points_ply: Path | None, n_mono: int, seed: int,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """xyz float32, rgb uint8: SfM points seen by the views + up to `n_mono` in-view mono points, ≤ `max_points`."""
    ids = {v.image_id for v in views}
    pts = [p for p in rec.points3D.values() if any(el.image_id in ids for el in p.track.elements)]
    xyz = np.array([p.xyz for p in pts], np.float32).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], np.uint8).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if max_points is not None and len(xyz) > max_points:
        pick = rng.choice(len(xyz), max_points, replace=False)
        return xyz[pick], rgb[pick]
    n_mono = min(n_mono, max_points - len(xyz)) if max_points is not None else n_mono
    if n_mono <= 0 or mono_points_ply is None or not mono_points_ply.is_file():
        return xyz, rgb
    mxyz = read_openmvs_points(mono_points_ply)[0].astype(np.float32)
    mxyz = mxyz[rng.choice(len(mxyz), min(len(mxyz), 2 * n_mono), replace=False)]
    mrgb, seen = np.zeros((len(mxyz), 3), np.uint8), np.zeros(len(mxyz), bool)
    for v in views:
        cam = mxyz @ v.viewmat[:3, :3].T + v.viewmat[:3, 3]
        front = cam[:, 2] > 1e-6
        px = cam[:, :2] / np.where(front, cam[:, 2], 1)[:, None] @ v.K[:2, :2].T + v.K[:2, 2]
        w, h = v.size
        hit = front & ~seen & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        mrgb[hit] = v.image[px[hit, 1].astype(int), px[hit, 0].astype(int)]
        seen |= hit
    idx = np.flatnonzero(seen)[:n_mono]
    return np.concatenate([xyz, mxyz[idx]]), np.concatenate([rgb, mrgb[idx]])


def ssim(a: Any, b: Any, win: int = 11, sigma: float = 1.5) -> Any:
    """Mean SSIM of two (H,W,3) images in [0, 1] (Gaussian window)."""
    import torch
    import torch.nn.functional as F

    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    x = torch.arange(win, device=a.device, dtype=a.dtype) - win // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum())[None, None]
    k = (g.transpose(-1, -2) @ g).expand(3, 1, win, win)
    f = lambda t: F.conv2d(t, k, padding=win // 2, groups=3)
    mu_a, mu_b = f(a), f(b)
    s_a, s_b, s_ab = f(a * a) - mu_a**2, f(b * b) - mu_b**2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_a + s_b + c2))).mean()


def rasterize(params: Any, K: Any, viewmat: Any, width: int, height: int, sh_degree: int, **kwargs: Any) -> tuple:
    """gsplat's `rasterization_2dgs` 7-tuple of one view from raw parameters (Splats, or a dict with sh or sh0+shN)."""
    import torch
    from gsplat import rasterization_2dgs

    p = {f.name: getattr(params, f.name) for f in fields(params)} if isinstance(params, Splats) else params
    sh = p["sh"] if "sh" in p else torch.cat([p["sh0"], p["shN"]], 1)
    scales = torch.exp(p["scales"])
    if scales.shape[1] == 2:
        scales = torch.cat([scales, torch.full_like(scales[:, :1], float(np.exp(PLY_FLAT_LOG_SCALE)))], 1)
    return rasterization_2dgs(
        means=p["means"], quats=p["quats"], scales=scales, opacities=torch.sigmoid(p["opacities"]),
        colors=sh, viewmats=viewmat[None], Ks=K[None], width=width, height=height, sh_degree=sh_degree,
        packed=False, **kwargs,
    )


def _init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, scene_scale: float, diag: float, seed: int) -> dict:
    """Initial raw parameters and their learning rates."""
    import torch

    n = len(xyz)
    dist = np.sqrt((cKDTree(xyz).query(xyz, 4)[0][:, 1:] ** 2).mean(1)).clip(1e-4 * diag, 0.02 * diag)
    log_s = np.log(dist * INIT_SCALE)[:, None].repeat(3, 1)
    log_s[:, 2] = np.log(dist * FLAT_SCALE)
    sh = np.zeros((n, (sh_degree + 1) ** 2, 3), np.float32)
    sh[:, 0] = (rgb / 255.0 - 0.5) / C0
    gen = torch.Generator().manual_seed(seed)
    return {
        "means": (torch.from_numpy(xyz), LR["means"] * scene_scale),
        "scales": (torch.tensor(log_s, dtype=torch.float32), LR["scales"]),
        "quats": (torch.rand(n, 4, generator=gen), LR["quats"]),
        "opacities": (torch.logit(torch.full((n,), INIT_OPACITY)), LR["opacities"]),
        "sh0": (torch.from_numpy(sh[:, :1].copy()), LR["sh0"]),
        "shN": (torch.from_numpy(sh[:, 1:].copy()), LR["shN"]),
    }


def train(
    views: list[View], init: tuple[np.ndarray, np.ndarray], mono: dict[str, tuple[np.ndarray, np.ndarray]],
    options: GsOptions, device: str, log_csv: Path, seed: int = 0,
) -> Splats:
    """Train on the non-held-out views with MCMC up to `max_gaussians`; LOG_COLUMNS rows to `log_csv`."""
    import torch
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_views = [v for v in views if not v.holdout] or views
    xyz, rgb = init
    lo, hi = np.percentile(xyz, [0.5, 99.5], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    centers = np.array([-v.viewmat[:3, :3].T @ v.viewmat[:3, 3] for v in views])
    scene_scale = max(1.1 * float(np.linalg.norm(centers - centers.mean(0), axis=1).max()), 0.05 * diag)
    init_params = _init_params(xyz, rgb, options.sh_degree, scene_scale, diag, seed)
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, (v, _) in init_params.items()})
    opts = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
            for k, (_, lr) in init_params.items()}
    iters = options.iterations
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=MEANS_LR_END ** (1.0 / max(iters, 1)))
    strategy = MCMCStrategy(cap_max=options.max_gaussians, refine_start_iter=REFINE_START,
                            refine_stop_iter=int(REFINE_STOP * iters), refine_every=REFINE_EVERY)
    state = strategy.initialize_state()
    strategy.check_sanity(splats, opts)
    dist_from, normal_from = int(DIST_FROM * iters), int(NORMAL_FROM * iters)
    use_dist = options.lambda_distortion > 0
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    zero = torch.zeros((), device=device)
    with log_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)
        t0 = time.perf_counter()
        for step in range(iters):
            v = train_views[rng.integers(len(train_views))]
            w, h = v.size
            K, T = (torch.from_numpy(a).to(device) for a in (v.K, v.viewmat))
            gt = torch.from_numpy(v.image).to(device).float() / 255
            out, alpha, normals, normals_d, distort, _, info = rasterize(
                splats, K, T, w, h, min(step // SH_EVERY, options.sh_degree), render_mode="RGB+ED", distloss=use_dist)
            colors, depth = out[0, ..., :3], out[0, ..., 3]
            l1 = (colors - gt).abs().mean()
            s = ssim(colors, gt)
            loss = (1 - options.lambda_dssim) * l1 + options.lambda_dssim * (1 - s)
            l_normal = l_dist = l_mono = zero
            if options.lambda_normal > 0 and step >= normal_from:
                l_normal = (1 - (normals[0] * (normals_d * alpha[0].detach())).sum(-1)).mean()
                loss = loss + options.lambda_normal * l_normal
            if use_dist and step >= dist_from:
                l_dist = distort.mean()
                loss = loss + options.lambda_distortion * l_dist
            if options.mono_weight > 0 and v.name in mono:
                m, valid = (torch.from_numpy(a).to(device) for a in mono[v.name])
                m = m[valid].float()
                l_mono = ((depth[valid] - m).abs() / m).mean()
                loss = loss + options.mono_weight * l_mono
            loss = loss + OPACITY_REG * torch.sigmoid(splats["opacities"]).mean()
            loss = loss + SCALE_REG * torch.exp(splats["scales"][:, :2]).mean() / scene_scale
            loss.backward()
            strategy.step_post_backward(splats, opts, state, step, info, lr=sched.get_last_lr()[0])
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            if step % LOG_EVERY == 0 or step == iters - 1:
                psnr = -10 * torch.log10(((colors.detach() - gt) ** 2).mean())
                vram = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else 0.0
                writer.writerow([step, *(f"{float(x.detach()):.6g}" for x in (loss, l1, s, l_dist, l_normal, l_mono)),
                                 len(splats["means"]), f"{float(psnr):.4f}", f"{time.perf_counter() - t0:.3f}",
                                 f"{vram:.0f}"])
                f.flush()
    p = {k: t.detach().cpu() for k, t in splats.items()}
    return Splats(p["means"], p["quats"], p["scales"][:, :2].contiguous(), p["opacities"],
                  torch.cat([p["sh0"], p["shN"]], 1))
