from conftest import assert_stage_ok, cli
from recon.io.tables import read_json
from recon.pipeline.cli import main
from recon.pipeline.run_dir import MANIFEST_NAME, read_manifest
from recon.pipeline.stages.s08_report import ARTIFACTS


def test_smoke(smoke_run):
    report = smoke_run.stage_dir("report")
    assert_stage_ok(report)
    figures = report / ARTIFACTS["figures"]
    assert (figures / "path.png").is_file()
    # the 3D renders need an OpenGL window; without one they are warnings
    if not [w for w in read_manifest(report).warnings if w.startswith("figure")]:
        assert all((figures / f"{name}.png").is_file() for name in ("overview", "stages", "path", "details"))
    summary = read_json(report / ARTIFACTS["summary"])
    mesh_post = next(e for e in summary["stages"] if e["stage"] == "mesh_post")
    if (smoke_run.stage_dir("dense") / "mono").is_dir():
        assert 0 <= mesh_post["metrics"]["hole_share_final"] <= 1


def test_compare(smoke_run, capsys):
    assert main(["compare", str(smoke_run.run_dir), str(smoke_run.run_dir)]) == 0
    assert capsys.readouterr().out.split()[:2] == ["stage", "metric"]


def test_cli_resume_skips_all(smoke_run, smoke_capture, output_root, capsys):
    manifests = sorted(smoke_run.scene_dir.glob(f"**/{MANIFEST_NAME}"))
    before = {p: p.stat().st_mtime_ns for p in manifests}
    capsys.readouterr()
    assert cli(smoke_capture, output_root, "run", "--resume", smoke_run.run_dir.name) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if " in " in ln and " s -> " in ln]
    assert lines and all(ln.split()[1] == "skipped" for ln in lines), lines
    assert {p: p.stat().st_mtime_ns for p in manifests} == before
