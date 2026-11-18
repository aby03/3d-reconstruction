import time

import numpy as np
import pandas as pd
import pycolmap
import pytest

pytest.importorskip("gsplat")
torch = pytest.importorskip("torch")

from recon.dense.gs_train import LOG_COLUMNS, Splats, init_points, load_mono_depth, load_views, train


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
def test_gs_train(smoke_run, tmp_path):
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

    t0 = time.perf_counter()
    splats = train(views, (xyz, rgb), mono, options, "cuda", tmp_path / "train_log.csv", seed=0)
    print(f"train {options.iterations} it: {time.perf_counter() - t0:.2f} s, {len(splats)} Gaussians")
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert list(log.columns) == LOG_COLUMNS and len(log) >= 1 and log["iter"].iloc[-1] == options.iterations - 1
    assert np.isfinite(log["loss"]).all() and 0 < len(splats) <= options.max_gaussians

    back = Splats.load_ply(splats.save_ply(tmp_path / "splats.ply"))
    for name in ("means", "quats", "scales", "opacities", "sh"):
        assert torch.equal(getattr(back, name), getattr(splats, name).float()), name
    ckpt = Splats.load_ckpt(splats.save_ckpt(tmp_path / "splats.pt"))
    assert torch.equal(ckpt.sh, splats.sh) and back.sh_degree == options.sh_degree
