import runpy
import socket
import subprocess
import sys
import time
import urllib.request
from unittest import mock
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pycolmap
import pytest
from conftest import assert_stage_ok, cli, rerun_copy

pytest.importorskip("gsplat")
torch = pytest.importorskip("torch")

from recon.dense.gs_render import fuse_mesh, holdout_metrics, novel_path, render_path, splat_points
from recon.dense.gs_train import LOG_COLUMNS, Splats, init_points, load_mono_depth, load_views, train
from recon.io.ply import read_openmvs_points, write_points
from recon.io.tables import read_json
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
    splats, info = train(views, (xyz, rgb), mono, options, "cuda", out / "train_log.csv", seed=0)
    assert info == {"step": options.iterations, "finished": True, "resumed_from": None, "elapsed_s": info["elapsed_s"]}
    print(f"train {options.iterations} it: {time.perf_counter() - t0:.2f} s, {len(splats)} Gaussians")
    return rec, views, splats, out, (xyz, rgb), mono


def test_gs_train(smoke_run, trained):
    options = smoke_run.config.dense.gs
    _, _, splats, out, _, _ = trained
    log = pd.read_csv(out / "train_log.csv")
    assert list(log.columns) == LOG_COLUMNS and len(log) >= 1 and log["iter"].iloc[-1] == options.iterations - 1
    assert np.isfinite(log["loss"]).all() and 0 < len(splats) <= options.max_gaussians

    back = Splats.load_ply(splats.save_ply(out / "splats.ply"))
    for name in ("means", "quats", "scales", "opacities", "sh"):
        assert torch.equal(getattr(back, name), getattr(splats, name).float()), name
    ckpt = Splats.load_ckpt(splats.save_ckpt(out / "splats.pt"))
    assert torch.equal(ckpt.sh, splats.sh) and back.sh_degree == options.sh_degree


def test_gs_resume(smoke_run, trained, tmp_path):
    """Stopped after 20 iterations and continued from the checkpoint, the training reaches the full schedule."""
    options = smoke_run.config.dense.gs
    _, views, _, _, init, mono = trained
    t0 = time.perf_counter()
    log, ckpt = tmp_path / "train_log.csv", tmp_path / "ckpt.pt"
    _, first = train(views, init, mono, options, "cuda", log, seed=0, ckpt=ckpt, stop_after=20)
    assert first["step"] == 20 and not first["finished"] and first["resumed_from"] is None
    assert "optimizers" in torch.load(ckpt, weights_only=False)
    splats, second = train(views, init, mono, options, "cuda", log, seed=0, ckpt=ckpt)
    assert second["step"] == options.iterations and second["finished"] and second["resumed_from"] == 20
    assert second["elapsed_s"] >= first["elapsed_s"] and 0 < len(splats) <= options.max_gaussians
    saved = torch.load(ckpt, weights_only=False)
    assert saved["step"] == options.iterations and "optimizers" not in saved
    iters = pd.read_csv(log)["iter"]
    assert iters.is_monotonic_increasing and iters.iloc[-1] == options.iterations - 1 and (iters < 20).any()
    print(f"resume test {time.perf_counter() - t0:.2f} s")


def test_gs_render(smoke_run, trained):
    gs, fusion = smoke_run.config.dense.gs, smoke_run.config.dense.fusion
    rec, views, splats, out, _, _ = trained
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


@pytest.fixture(scope="module")
def backend_run(smoke_run, output_root, tmp_path_factory):
    """The smoke run from dense on with backend 2dgs: stopped at once by max_minutes, then continued with
    --force dense to the end. Returns (run dir, outputs root, stopped dense report, undistorted mtime, timings)."""
    capture, tmp_path = next(iter(smoke_run.inputs.values())), tmp_path_factory.mktemp("gs_backend")
    t0 = time.perf_counter()
    run = rerun_copy(smoke_run, output_root, tmp_path, "--from", "dense", "--to", "dense",
                     "--set", "dense.backend=2dgs", "--set", "dense.gs.max_minutes=0")
    dense = run / "05_dense"
    stopped = read_json(dense / "reports" / "dense_report.json")
    undistorted_at = (dense / "undistorted" / "sparse" / "cameras.bin").stat().st_mtime_ns
    t_stopped = time.perf_counter() - t0

    t0 = time.perf_counter()
    assert cli(capture, tmp_path, "run", "--resume", run.name, "--force", "dense", "--set", "dense.backend=2dgs") == 0
    return run, tmp_path, stopped, undistorted_at, (t_stopped, time.perf_counter() - t0)


def test_gs_backend(smoke_run, backend_run):
    """The stopped run warns, the continued one finishes and keeps the prep; a re-run that changes only render
    options then keeps the training."""
    capture = next(iter(smoke_run.inputs.values()))
    run, tmp_path, stopped, undistorted_at, (t_stopped, t_first) = backend_run
    assert not stopped["gs"]["finished"] and stopped["gs"]["step"] == 0 and stopped["gs"]["resumed_from"] is None
    assert any("rerun dense with --force dense" in w for w in stopped["warnings"])
    assert not stopped["gs"]["prep_reused"]
    dense = run / "05_dense"
    sparse = dense / "undistorted" / "sparse" / "cameras.bin"
    for stage in ("05_dense", "06_mesh_post", "07_export", "08_report"):
        assert_stage_ok(run / stage)
    mesh = read_json(run / "06_mesh_post" / "reports" / "mesh_report.json")
    report = read_json(dense / "reports" / "dense_report.json")
    assert report["dense_backend"] == "2dgs" and not report["gs"]["reused"] and report["gs"]["num_gaussians"] > 0
    assert report["gs"]["finished"] and report["gs"]["step"] == report["gs"]["iterations"]
    assert report["gs"]["resumed_from"] == 0
    # the second chunk keeps the undistortion, import and mono depth of the first
    assert report["gs"]["prep_reused"] and sparse.stat().st_mtime_ns == undistorted_at
    assert (dense / "mono_points.ply").is_file() and any((dense / "mono").rglob("*.npz"))
    gs_novel = smoke_run.config.dense.gs.novel_views
    for name in ("gs_training", "gs_holdout", "gs_depth_vs_mono"):
        assert (dense / "reports" / f"{name}.png").stat().st_size > 0, name
    assert (dense / "gs" / "novel_strip.png").is_file() and len(list((dense / "gs" / "novel").glob("*.jpg"))) == gs_novel
    assert _frames(dense / "gs" / "novel.mp4") == gs_novel == report["gs"]["novel_views"]
    assert (run / "08_report" / "figures" / "gaussians.png").is_file()
    assert '<video src="../05_dense/gs/novel.mp4"' in (run / "08_report" / "index.html").read_text()
    assert report["mesh_raw"]["faces"] > 0 and report["timings"]["train"] > 0
    # nothing of the OpenMVS run is left for mesh_post to read
    assert not any((dense / rel).exists() for rel in ("mesh_tsdf.ply", "chain.json", "openmvs/scene_dense.ply"))
    assert (dense / "openmvs" / "scene_dense.mvs").is_file()
    assert mesh["dense_backend"] == "2dgs" and not mesh["merge"]["enabled"] and not mesh["fill"]["enabled"]
    assert mesh["texture"]["faces"] == mesh["final"]["faces"]
    assert (run / "07_export" / "splats" / "splats.ply").stat().st_size > 0
    print(f"qa_p95_tau {mesh['qa']['dense_to_mesh']['p95'] / mesh['qa']['tau']:.2f}, {report['num_dense_points']} points")

    t0 = time.perf_counter()
    assert cli(capture, tmp_path, "run", "--resume", run.name, "--force", "dense", "--to", "dense",
               "--set", "dense.backend=2dgs", "--set", "dense.gs.novel_views=2", "--set", "report.figures.width=200") == 0
    again = read_json(dense / "reports" / "dense_report.json")
    assert again["gs"]["reused"] and again["gs"]["train_s"] == report["gs"]["train_s"]
    assert again["mesh_raw"]["faces"] == report["mesh_raw"]["faces"] and again["gs"]["prep_reused"]
    assert _frames(dense / "gs" / "novel.mp4") == 2 == len(list((dense / "gs" / "novel").glob("*.jpg")))
    print(f"2dgs dense stopped at once: {t_stopped:.1f} s, continued from dense on: {t_first:.1f} s,"
          f" again (training reused): {time.perf_counter() - t0:.1f} s")


def test_view_splats(backend_run, capsys):
    """The viewer renders every mode offscreen (with the mesh) and its server answers HTTP."""
    run = backend_run[0]
    script = Path(__file__).parents[1] / "scripts" / "view_splats.py"
    t0 = time.perf_counter()
    with mock.patch.object(sys, "argv", ["view_splats.py", "--run", str(run), "--check", "--mesh"]):
        runpy.run_path(str(script), run_name="__main__")  # in-process: torch and gsplat are loaded already
    assert "ok:" in capsys.readouterr().out
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, script, "--run", run, "--port", str(port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                assert urllib.request.urlopen(f"http://localhost:{port}", timeout=1).status == 200
                break
            except OSError:
                assert server.poll() is None, "viewer exited"
                time.sleep(0.5)
        else:
            pytest.fail("viewer did not listen")
    finally:
        server.kill()
        server.wait()
    print(f"viewer check + serve {time.perf_counter() - t0:.1f} s")


def _frames(video) -> int:
    with av.open(str(video)) as container:
        return sum(1 for _ in container.decode(video=0))
