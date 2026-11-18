import time

import numpy as np
import pandas as pd
import pycolmap
import pytest

pytest.importorskip("gsplat")
torch = pytest.importorskip("torch")

from recon.dense.gs_render import fuse_mesh, holdout_metrics, novel_path, render_path, splat_points
from recon.dense.gs_train import LOG_COLUMNS, Splats, init_points, load_mono_depth, load_views, train
from recon.io.ply import read_openmvs_points, write_points
from recon.mesh.qa import tau as geometry_tau

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


@pytest.fixture(scope="module")
def trained(smoke_run, tmp_path_factory):
    """50 iterations at 160 px on the smoke run's undistorted model: (rec, views, splats, out dir)."""
    dense = smoke_run.run_dir / "05_dense"
    options = smoke_run.config.dense.gs
    rec = pycolmap.Reconstruction(dense / "undistorted" / "sparse")
    views = load_views(rec, dense / "undistorted" / "images", options.long_side, options.holdout_every)
    assert views and any(v.holdout for v in views) and max(views[0].size) == options.long_side
    mono = load_mono_depth(dense / "mono", views)
    sizes = {v.name: v.size for v in views}
    assert mono and all(d.dtype == np.float16 and d.shape[::-1] == sizes[n] and ok.any() for n, (d, ok) in mono.items())
    xyz, rgb = init_points(rec, views, dense / "mono_points.ply", options.init_mono_points, 0, options.max_gaussians)
    assert len(xyz) == len(rgb) <= options.max_gaussians and rgb.dtype == np.uint8
    out = tmp_path_factory.mktemp("gs")
    t0 = time.perf_counter()
    splats = train(views, (xyz, rgb), mono, options, "cuda", out / "train_log.csv", seed=0)
    print(f"train {options.iterations} it: {time.perf_counter() - t0:.2f} s, {len(splats)} Gaussians")
    return rec, views, splats, out


def test_gs_train(smoke_run, trained):
    options = smoke_run.config.dense.gs
    _, _, splats, out = trained
    log = pd.read_csv(out / "train_log.csv")
    assert list(log.columns) == LOG_COLUMNS and len(log) >= 1 and log["iter"].iloc[-1] == options.iterations - 1
    assert np.isfinite(log["loss"]).all() and 0 < len(splats) <= options.max_gaussians

    back = Splats.load_ply(splats.save_ply(out / "splats.ply"))
    for name in ("means", "quats", "scales", "opacities", "sh"):
        assert torch.equal(getattr(back, name), getattr(splats, name).float()), name
    ckpt = Splats.load_ckpt(splats.save_ckpt(out / "splats.pt"))
    assert torch.equal(ckpt.sh, splats.sh) and back.sh_degree == options.sh_degree


def test_gs_render(smoke_run, trained):
    gs, fusion = smoke_run.config.dense.gs, smoke_run.config.dense.fusion
    rec, views, splats, out = trained
    t0 = time.perf_counter()
    metrics = holdout_metrics(splats, views)
    assert metrics["per_view"] and np.isfinite([metrics["psnr"], metrics["ssim"]]).all()

    lo, hi = np.percentile(splats.means.numpy(), [0.5, 99.5], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    mesh, report = fuse_mesh(splats, views, fusion, geometry_tau("arbitrary", diag), diag, gs)
    assert report["views"] == len(views) and len(mesh.faces) == report["faces"]

    xyz, normals, rgb = splat_points(splats, views)
    assert len(xyz) == len(splats) and np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-4)
    pts = read_openmvs_points(write_points(out / "dense_points.ply", xyz, normals, rgb))
    assert np.allclose(pts[0], xyz) and (pts[2] == rgb).all()

    path, size = novel_path(rec, gs.novel_views, gs.render_long_side)
    assert len(path) == gs.novel_views and max(size) == gs.render_long_side
    files = render_path(splats, path, size, out / "novel_views")
    assert len(files) == gs.novel_views and all(f.stat().st_size > 0 for f in files)
    print(f"render test {time.perf_counter() - t0:.2f} s (fuse {report['fuse_s']} s), {report['faces']} faces")
