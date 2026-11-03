from html.parser import HTMLParser

from recon.io.tables import read_json
from recon.pipeline.cli import main
from recon.pipeline.config import PipelineConfig, load_config
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s08_report import ARTIFACTS
from recon.report.summary import compare

STAGES = ["ingest", "keyframes", "sfm", "align", "dense", "mesh_post", "export"]


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs)["src"])


def test_manifest(report_run):
    manifest = read_manifest(report_run)
    assert manifest.status == "success" and manifest.artifacts == ARTIFACTS
    for rel in ARTIFACTS.values():
        assert (report_run / rel).exists()
    assert list(manifest.upstream) == ["export"]
    assert manifest.metrics["stages"] == len(STAGES) and manifest.metrics["checks"] == 10


def test_summary(report_run, mesh_post_run):
    summary = read_json(report_run / ARTIFACTS["summary"])
    assert summary["scene"] == "room2_tiny" and summary["run_id"] == report_run.parent.name
    assert [s["stage"] for s in summary["stages"]] == STAGES
    assert all(s["status"] == "success" and s["duration_s"] > 0 for s in summary["stages"])
    assert summary["total_s"] > 0 and summary["tools"]["OpenMVS"]
    mesh_post = next(s for s in summary["stages"] if s["stage"] == "mesh_post")
    derived = {"qa_p95_tau": mesh_post["metrics"]["qa_p95_tau"]}
    assert mesh_post["metrics"] == {**read_manifest(mesh_post_run).metrics, **derived}
    assert mesh_post["planes"] and mesh_post["qa"]["tau"] > 0
    assert next(s for s in summary["stages"] if s["stage"] == "export")["files"]

    checks = {(c["stage"], c["metric"]): c for c in summary["checks"]}
    assert checks[("sfm", "registered_ratio")]["op"] == ">=" and checks[("sfm", "mean_reproj_px")]["op"] == "<="
    assert checks[("export", "all_valid")] == {**checks[("export", "all_valid")], "value": True, "pass": True}
    assert summary["pass"] == all(c["pass"] for c in summary["checks"])
    # threshold failures are warnings, never stage failures
    failed = [c for c in summary["checks"] if c["pass"] is not True]
    assert len(read_manifest(report_run).warnings) == len(failed)


def test_html(report_run):
    page = report_run / ARTIFACTS["html"]
    links = _Links()
    links.feed(page.read_text())
    assert "plots/timings.png" in links.images and len(links.images) > 5
    assert all((page.parent / src).is_file() for src in links.images)


def test_config():
    assert load_config().report == PipelineConfig().report


def test_compare(report_run, capsys):
    summary = read_json(report_run / ARTIFACTS["summary"])
    rows = compare(summary, summary)
    assert rows and all(r["diff"] == 0 for r in rows)
    assert main(["compare", str(report_run.parent), str(report_run.parent)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:2] == ["stage", "metric"] and len(out) == len(rows) + 1


def test_cli_skips(tiny_capture, output_root, report_run):
    mtime = (report_run / "stage_manifest.json").stat().st_mtime_ns
    args = ["--input", str(tiny_capture), "--output-root", str(output_root), "--resume", report_run.parent.name]
    assert main(["report", *args]) == 0
    assert (report_run / "stage_manifest.json").stat().st_mtime_ns == mtime
