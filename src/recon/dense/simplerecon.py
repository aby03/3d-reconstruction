"""Learned multi-view depth per keyframe: SimpleRecon (Niantic, ECCV 2022; non-commercial licence), a plane-sweep cost
volume over a reference view and its best source views, decoded to a depth map. It expects metric poses, so the poses
are scaled by a factor `s` (model units per metre) before the network and the depth is divided by `s` after.

The upstream repository is fetched at a fixed commit into `$SIMPLERECON_DIR` (`pixi run fetch-simplerecon`) and is not
part of this repository; the weights are downloaded once into the cache directory and checked against their SHA-256."""

import contextlib
import functools
import hashlib
import importlib.util
import logging
import os
import subprocess
import sys
import time
import types
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap

from recon.dense.mono import MonoOptions, ViewFit, cache_dir, view_points

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/nianticlabs/simplerecon"
REPO_COMMIT = "113b825084cc62166c6f5584d526650928127317"
WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=1hCuKZjEq-AghrYAmFxJs_4eeixIlP488&export=download&confirm=t"
WEIGHTS_SHA256 = "4752de233de1253934434141c2f713d7e0282326457386604477640d90028b74"
WEIGHTS_FILE = "simplerecon_hero_model.ckpt"
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float32)
STRIDE = 32  # the network input sides are multiples of it
SCALES = 5  # intrinsics pyramid levels K_s0..K_s4 the network reads
# 90 degrees clockwise: x' = -y, y' = x, z' = z in the camera frame
ROT90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class SimpleReconOptions:
    repo: str | None = None  # None: $SIMPLERECON_DIR
    cache_dir: str | None = None  # None: as for the mono model
    num_sources: int = 7  # source views per reference view (the network was trained with 7)
    long_side: int = 672  # network input long side (rounded to a multiple of 32); the short side keeps the image aspect
    rotate: bool = False  # rotate portrait views by 90 degrees (landscape input; K and pose rotated with them)
    optimal_baseline_m: float = 0.15  # source-view rule: preferred translation between reference and source, metres
    min_depth_m: float = 0.25  # the cost volume's depth planes (metres)
    max_depth_m: float = 5.0
    edge_rel: float = 0.05  # depth jumps above this fraction of the depth between neighbour pixels have no confidence
    fp16: bool = False
    scale: float | None = None  # model units per metre; None: searched on a sample of views
    typical_depth_m: float = 2.5  # start of the search without an accelerometer scale: the median SfM depth is this
    search_views: int = 16  # views the scale search runs on
    search_factor: float = 1.36  # grid ratio: 7 values from start / factor^3 to start * factor^3 (0.4x - 2.5x), then 4 between the best's neighbours
    search_expand: int = 3  # a best at the grid's edge shifts the grid by one value, at most this many times
    min_fitted_frac: float = 0.7  # the search fails (the caller falls back) when fewer sampled views fit at the best scale


def repo_dir(options: SimpleReconOptions) -> Path:
    path = options.repo or os.environ.get("SIMPLERECON_DIR")
    if not path:
        raise RuntimeError("SimpleRecon: set $SIMPLERECON_DIR or the `repo` option (see `pixi run fetch-simplerecon`)")
    return Path(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_repo(options: SimpleReconOptions | None = None) -> Path:
    """The upstream repository at `REPO_COMMIT`, cloned if missing."""
    path = repo_dir(options or SimpleReconOptions())
    if not (path / "experiment_modules").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--quiet", REPO_COMMIT], check=True)
    return path


def fetch_weights(options: SimpleReconOptions | None = None) -> tuple[Path, dict[str, Any]]:
    """The `hero_model` checkpoint, downloaded once into the cache directory and checked against its SHA-256, and the
    record of where it came from."""
    options = options or SimpleReconOptions()
    path = cache_dir(MonoOptions(cache_dir=options.cache_dir)) / WEIGHTS_FILE
    if not path.is_file() or _sha256(path) != WEIGHTS_SHA256:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("downloading the SimpleRecon weights to %s", path)
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        if (digest := _sha256(tmp)) != WEIGHTS_SHA256:
            tmp.unlink()
            raise RuntimeError(f"SimpleRecon weights: SHA-256 {digest}, expected {WEIGHTS_SHA256}")
        tmp.replace(path)
    return path, {"url": WEIGHTS_URL, "sha256": WEIGHTS_SHA256, "license": "non-commercial", "path": str(path)}


@contextlib.contextmanager
def _upstream_imports(repo: Path) -> Iterator[None]:
    """Import context for the upstream code: its directory on the path, TorchScript off (its helpers do not compile
    with the kornia here), no download of pretrained backbones (the checkpoint overrides them) and
    a stub for the video library it imports for visualisation."""
    import antialiased_cnns
    import timm
    import torch

    saved = (torch.jit.script, torch.jit.ScriptModule, torch.jit.script_method, timm.create_model, antialiased_cnns.resnet18)
    sys.path.insert(0, str(repo))
    torch.jit.script, torch.jit.ScriptModule, torch.jit.script_method = (lambda f, *a, **k: f), torch.nn.Module, (lambda f: f)
    create, resnet18 = timm.create_model, antialiased_cnns.resnet18
    timm.create_model = lambda *a, **k: create(*a, **{**k, "pretrained": False})
    antialiased_cnns.resnet18 = lambda *a, **k: resnet18(pretrained=False, **{x: y for x, y in k.items() if x != "pretrained"})
    if "moviepy" not in sys.modules:
        sys.modules["moviepy"] = types.ModuleType("moviepy")
        sys.modules["moviepy.editor"] = types.ModuleType("moviepy.editor")
    try:
        yield
    finally:
        torch.jit.script, torch.jit.ScriptModule, torch.jit.script_method, timm.create_model, antialiased_cnns.resnet18 = saved
        sys.path.remove(str(repo))


def network_size(width: int, height: int, options: SimpleReconOptions) -> tuple[int, int]:
    """(width, height) of the network input for an image (after the optional rotation): long side `long_side`, both
    sides multiples of 32."""
    scale = options.long_side / max(width, height)
    return max(STRIDE, round(width * scale / STRIDE) * STRIDE), max(STRIDE, round(height * scale / STRIDE) * STRIDE)


class SimpleRecon:
    """The network on `device`, built for one input size (the cost volume keeps projection grids of that size)."""

    def __init__(self, size: tuple[int, int], device: str, options: SimpleReconOptions) -> None:
        import torch

        self.torch, self.options, self.device, self.size = torch, options, device, size
        weights, _ = fetch_weights(options)
        repo = repo_dir(options)
        with _upstream_imports(repo):
            from experiment_modules.depth_model import DepthModel

            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            opts = checkpoint["hyper_parameters"]["opts"]
            opts.image_width, opts.image_height = size
            opts.min_matching_depth, opts.max_matching_depth = options.min_depth_m, options.max_depth_m
            model = DepthModel(opts)
        # the projection grids are buffers of the training size; the rest are weights and must all be there
        current = model.state_dict()
        state = {k: v for k, v in checkpoint["state_dict"].items() if k in current and current[k].shape == v.shape}
        if missing := [k for k in current if k not in state and not k.endswith("pix_coords_13N")]:
            raise RuntimeError(f"SimpleRecon checkpoint does not match the model: {missing[:3]}")
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()
        self.num_views = opts.model_num_views

    def pyramid(self, K: np.ndarray, shape: tuple[int, int]) -> dict[str, Any]:
        """K_s{0..4} and inverses (4x4, batch of 1) for an image of `shape` = (h, w) whose 3x3 intrinsics are K;
        level 0 is the depth map (half the network input), each next level half the size of the previous."""
        torch = self.torch
        w, h = self.size
        out = {}
        for i in range(SCALES):
            k = np.eye(4, dtype=np.float32)
            k[:3, :3] = K
            k[0] *= w / shape[1] / 2**i / 2
            k[1] *= h / shape[0] / 2**i / 2
            out[f"K_s{i}_b44"] = torch.from_numpy(k)[None].to(self.device)
            out[f"invK_s{i}_b44"] = torch.from_numpy(np.linalg.inv(k))[None].to(self.device)
        return out

    def predict(
        self,
        images: list[np.ndarray],
        Ks: list[np.ndarray],
        cam_from_world: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Depth (metres, float32 (h/2, w/2) of the network input) and the source-view coverage mask (bool, same
        shape) of the first view, the reference, from the others, its sources. `images` are BGR, `Ks` 3x3 and
        `cam_from_world` 4x4, all already in the network's orientation and metric scale."""
        torch, device = self.torch, self.device
        w, h = self.size

        def prepare(image: np.ndarray, K: np.ndarray, T: np.ndarray) -> dict[str, Any]:
            rgb = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(((rgb.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1).copy())
            T = T.astype(np.float32)
            return {
                "image_b3hw": x[None].to(device),
                "cam_T_world_b44": torch.from_numpy(T)[None].to(device),
                "world_T_cam_b44": torch.from_numpy(np.linalg.inv(T).astype(np.float32))[None].to(device),
                **self.pyramid(K, image.shape[:2]),
            }

        views = [prepare(*v) for v in zip(images, Ks, cam_from_world)]
        cur, src = views[0], views[1:]
        src_data = {k: torch.stack([s[k][0] for s in src])[None] for k in src[0]}  # (1, M, ...)
        with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=self.options.fp16):
            out = self.model("test", cur, src_data, unbatched_matching_encoder_forward=True, return_mask=True)
        depth = out["depth_pred_s0_b1hw"][0, 0].float().cpu().numpy()
        mask = out["overall_mask_bhw"][0].cpu().numpy().astype(np.uint8)  # at the cost volume's (coarser) size
        mask = cv2.resize(mask, depth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        return depth, mask


def source_views(cam_from_world: np.ndarray, ref: int, scale: float, options: SimpleReconOptions) -> list[int]:
    """Indices of the `num_sources` best source views of view `ref` among `cam_from_world` (N, 4, 4, model units),
    by the rule of the network's training data (DeepVideoMVS): a penalty on the rotation angle measure plus the
    translation's distance to `optimal_baseline_m` (nearer than that counts five times as much). `scale` is model
    units per metre."""
    world_from_cam = np.linalg.inv(cam_from_world)
    rel = cam_from_world[ref][None] @ world_from_cam  # source camera -> reference camera
    t = np.linalg.norm(rel[:, :3, 3], axis=1) / scale
    trace = np.trace(rel[:, :3, :3], axis1=1, axis2=2)
    rot = np.sqrt(np.maximum(2.0 * (1.0 - np.minimum(3.0, trace) / 3.0), 0.0))
    diff = t - options.optimal_baseline_m
    penalty = rot**2 + np.where(diff < 0, 5.0, 1.0) * diff**2
    penalty[ref] = np.inf
    return [int(i) for i in np.argsort(penalty)[: min(options.num_sources, len(penalty) - 1)]]


def rotate_view(image: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The view turned 90 degrees clockwise: image, 3x3 intrinsics and 4x4 cam_from_world."""
    h = image.shape[0]
    K2 = np.array([[K[1, 1], 0, h - K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])
    T2 = np.eye(4)
    T2[:3] = ROT90 @ T[:3]
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), K2, T2


def scale_pose(T: np.ndarray, scale: float) -> np.ndarray:
    """cam_from_world with the translation divided by `scale` (model units per metre): the pose in metres."""
    out = T.copy()
    out[:3, 3] /= scale
    return out


def confidence(depth: np.ndarray, valid: np.ndarray, edge_rel: float) -> np.ndarray:
    """uint8 mask: 1 where the depth is positive, `valid` and not on a depth jump."""
    gy, gx = np.gradient(depth)
    jump = np.hypot(gx, gy) > edge_rel * np.maximum(depth, 1e-9)
    return ((depth > 0) & valid & ~jump).astype(np.uint8)


def point_agreement(
    depth: np.ndarray, uv: np.ndarray, z: np.ndarray, options: MonoOptions, valid: np.ndarray | None = None
) -> tuple[ViewFit, float]:
    """How well a depth map (SfM units) agrees with the SfM depths `z` (N,) at the pixels `uv` (N, 2; in the depth
    map's coordinates), as a `ViewFit` of the identity map in inverse depth (scale 1, shift 0): inliers have a
    relative error below `options.inlier_rel`; pixels outside `valid` or without depth count as misses. Also the
    median ratio of the map's depth to the SfM depth (1: the scale is right)."""
    n = len(z)
    skip = ViewFit(1.0, 0.0, n, 0, 0.0, float("nan"), float("nan"), False)
    if n < options.min_points:
        skip.reason = f"{n} points (< {options.min_points})"
        return skip, float("nan")
    h, w = depth.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    d = depth[y, x].astype(float)
    ok = d > 0 if valid is None else (d > 0) & valid[y, x]
    err = np.full(n, np.inf)
    err[ok] = np.abs(d[ok] - z[ok]) / z[ok]
    inliers = err < options.inlier_rel
    ratio = float(inliers.mean())
    fit = ViewFit(
        1.0, 0.0, n, int(inliers.sum()), ratio,
        float(np.median(err[inliers])) if inliers.any() else float("nan"),
        float(np.median(err[np.isfinite(err)])) if np.isfinite(err).any() else float("nan"),
        True,
    )
    ratio_to_sfm = float(np.median(d[ok] / z[ok])) if ok.any() else float("nan")
    if fit.inliers < options.min_inliers:
        fit.fitted, fit.reason = False, f"{fit.inliers} inliers (< {options.min_inliers})"
    elif ratio < options.min_inlier_ratio:
        fit.fitted, fit.reason = False, f"inlier ratio {ratio:.2f} (< {options.min_inlier_ratio})"
    return fit, ratio_to_sfm



ERROR_CAP = 0.2  # relative error of a view that does not fit, and the most any view adds to a scale's score


class SimpleReconError(RuntimeError):
    """SimpleRecon cannot be used on this input (not installed, or no metric scale makes it agree with the SfM)."""


def check_installed(options: SimpleReconOptions) -> None:
    """Raises `SimpleReconError` unless the Python packages and the upstream checkout are there."""
    missing = [m for m in ("timm", "pytorch_lightning", "kornia", "antialiased_cnns", "einops") if importlib.util.find_spec(m) is None]
    if missing:
        raise SimpleReconError(f"missing packages {', '.join(missing)} (use the `simplerecon` pixi environment)")
    try:
        path = repo_dir(options)
    except RuntimeError as e:
        raise SimpleReconError(str(e)) from None
    if not (path / "experiment_modules").is_dir():
        raise SimpleReconError(f"no SimpleRecon checkout at {path} (run `pixi run -e simplerecon fetch-simplerecon`)")


@dataclass
class ViewDepth:
    depth: np.ndarray  # float32, SfM units, in the image's own orientation
    confidence: np.ndarray  # uint8
    fit: ViewFit
    depth_ratio: float  # median depth / SfM depth at the SfM points


class DepthRunner:
    """SimpleRecon depth of the views of an undistorted (PINHOLE) model: sources chosen by pose over all its views,
    one network per input size (views of other sizes or orientations get theirs when first needed)."""

    def __init__(
        self, rec: pycolmap.Reconstruction, images_dir: Path, options: SimpleReconOptions, device: str, mono: MonoOptions
    ) -> None:
        self.rec, self.images_dir, self.options, self.device, self.mono = rec, images_dir, options, device, mono
        self.images = sorted(rec.images.values(), key=lambda im: im.name)
        self.names = [im.name for im in self.images]
        self.cam_from_world = np.stack([_pose(im) for im in self.images])
        self.nets: dict[tuple[int, int], SimpleRecon] = {}
        self.seconds: list[float] = []
        self._view = functools.lru_cache(maxsize=64)(self._load)

    def _turned(self, i: int) -> bool:
        cam = self.rec.cameras[self.images[i].camera_id]
        return self.options.rotate and cam.height > cam.width

    def _size(self, i: int) -> tuple[int, int]:
        cam = self.rec.cameras[self.images[i].camera_id]
        w, h = (cam.height, cam.width) if self._turned(i) else (cam.width, cam.height)
        return network_size(w, h, self.options)

    def _load(self, i: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image at the network size, its 3x3 intrinsics and cam_from_world (model units), turned when asked."""
        image = cv2.imread(str(self.images_dir / self.names[i]))
        cam = self.rec.cameras[self.images[i].camera_id]
        fx, fy, cx, cy = cam.params
        K, T = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), self.cam_from_world[i]
        if self._turned(i):
            image, K, T = rotate_view(image, K, T)
        K = K * np.array([[size[0] / image.shape[1]], [size[1] / image.shape[0]], [1.0]])
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA), K, T

    def depth(self, i: int, scale: float) -> ViewDepth:
        """Depth of view `i` for `scale` model units per metre."""
        size, options = self._size(i), self.options
        if size not in self.nets:
            self.nets[size] = SimpleRecon(size, self.device, options)
        net = self.nets[size]
        # every source is resized to the reference's input size, with its intrinsics scaled to match
        views = [self._view(j, size) for j in [i, *source_views(self.cam_from_world, i, scale, options)]]
        t0 = time.perf_counter()
        depth_m, valid = net.predict(
            [v[0] for v in views], [v[1] for v in views], [scale_pose(v[2], scale) for v in views]
        )
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        self.seconds.append(time.perf_counter() - t0)
        valid &= (depth_m >= options.min_depth_m) & (depth_m <= options.max_depth_m)
        depth = (depth_m * scale).astype(np.float32)
        conf = confidence(depth, valid, options.edge_rel)
        uv, z = view_points(self.rec, self.images[i])
        cam = self.rec.cameras[self.images[i].camera_id]
        if self._turned(i):
            uv = np.stack([cam.height - uv[:, 1], uv[:, 0]], axis=1)
        src = (cam.height, cam.width) if self._turned(i) else (cam.width, cam.height)
        fit, ratio = point_agreement(depth, uv * np.array([depth.shape[1] / src[0], depth.shape[0] / src[1]]), z, self.mono, valid)
        if self._turned(i):
            depth, conf = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(conf, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return ViewDepth(depth, conf, fit, ratio)

    def peak_vram_gb(self) -> float:
        if not self.device.startswith("cuda"):
            return 0.0
        import torch

        return torch.cuda.max_memory_allocated() / 1e9


def _pose(image: pycolmap.Image) -> np.ndarray:
    T = np.eye(4)
    T[:3] = image.cam_from_world().matrix()
    return T


def start_scale(rec: pycolmap.Reconstruction, accelerometer: float | None, options: SimpleReconOptions) -> tuple[float, str]:
    """Where the scale search starts (model units per metre) and why: the accelerometer scale of the SfM when there is
    one, else the median SfM depth over `typical_depth_m`."""
    if accelerometer and accelerometer > 0:
        return float(accelerometer), "accelerometer"
    depths = [np.median(view_points(rec, im)[1]) for im in rec.images.values() if im.num_points3D >= 30]
    if not depths:
        raise SimpleReconError("no view has enough SfM points to start the scale search")
    return float(np.median(depths) / options.typical_depth_m), "median SfM depth"


def search_scale(
    score: Callable[[float], tuple[float, float]], start: float, options: SimpleReconOptions
) -> tuple[float, list[dict[str, float]]]:
    """The scale with the lowest error of `score(s) -> (error, fitted share)`: a geometric grid around `start`
    (shifted while its best value is at an edge), then a finer pass around the best. Returns it and every tried
    value as {scale, error, fitted}."""
    tried: dict[float, tuple[float, float]] = {}

    def evaluate(s: float) -> float:
        s = float(s)
        if s not in tried:
            tried[s] = score(s)
        return tried[s][0]

    f = options.search_factor
    lo, hi = -3, 3
    for s in start * f ** np.arange(lo, hi + 1):
        evaluate(s)
    for _ in range(options.search_expand):
        best = min(tried, key=lambda s: tried[s][0])
        if best >= start * f**hi * 0.999:
            hi += 1
            evaluate(start * f**hi)
        elif best <= start * f**lo * 1.001:
            lo -= 1
            evaluate(start * f**lo)
        else:
            break
    best = min(tried, key=lambda s: tried[s][0])
    for e in (-0.5, -0.25, 0.25, 0.5):
        evaluate(best * f**e)
    best = min(tried, key=lambda s: tried[s][0])
    return best, [{"scale": s, "error": e, "fitted": fitted} for s, (e, fitted) in sorted(tried.items())]


def find_scale(
    runner: DepthRunner, start: float, options: SimpleReconOptions
) -> tuple[float, list[dict[str, float]]]:
    """`search_scale` on `options.search_views` evenly spaced views of `runner`, scored by the median over them of each
    view's median relative error against its SfM points: the mean over them with each view's error capped at
    `ERROR_CAP` and views that do not fit counted at the cap, so a scale that loses views cannot win. Raises `SimpleReconError` when
    fewer than `min_fitted_frac` of the views fit at the best scale."""
    usable = [i for i, im in enumerate(runner.images) if im.num_points3D >= runner.mono.min_points]
    if not usable:
        raise SimpleReconError("no view has enough SfM points to score the scale")
    picks = [usable[k] for k in np.unique(np.linspace(0, len(usable) - 1, min(options.search_views, len(usable))).round().astype(int))]

    def score(s: float) -> tuple[float, float]:
        fits = [runner.depth(i, s).fit for i in picks]
        err = np.array([f.median_rel_residual_all if f.fitted else np.inf for f in fits])
        return float(np.mean(np.minimum(err, ERROR_CAP))), float(np.mean([f.fitted for f in fits]))

    best, curve = search_scale(score, start, options)
    fitted = next(c["fitted"] for c in curve if c["scale"] == best)
    if fitted < options.min_fitted_frac:
        raise SimpleReconError(f"best scale {best:.3g}: only {fitted:.0%} of the sampled views agree with the SfM points")
    return best, curve


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        logging.basicConfig(level=logging.INFO)
        print(fetch_repo())
        print(fetch_weights()[0])
    else:
        raise SystemExit("usage: python -m recon.dense.simplerecon fetch")
