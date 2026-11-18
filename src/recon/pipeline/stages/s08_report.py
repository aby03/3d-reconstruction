"""Step 8: one report per run in `08_report/`: `summary.json` (every stage's manifest metrics, checked against
`report.thresholds`) and a self-contained `index.html` linking the stages' plots."""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recon.io.tables import read_json, write_json
from recon.pipeline.config import PipelineConfig, config_dict, load_config
from recon.pipeline.run_dir import RUN_MANIFEST_NAME, now_iso, read_manifest
from recon.pipeline.stage import Stage, StageOutput, register_stage, registered_stages
from recon.pipeline.stages import s05_dense, s06_mesh_post, s07_export
from recon.report import ReportOptions
from recon.report.figures import build_figures
from recon.report.html import render
from recon.report.plots import plot_timings
from recon.report.summary import check, summarize

log = logging.getLogger(__name__)

STAGE = "report"
DIR_NAME = "08_report"

ARTIFACTS = {
    "summary": "summary.json",
    "html": "index.html",
    "plots": "plots",
    "plot_timings": "plots/timings.png",
    "figures": "figures",
}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict) and obj:
        return {k: v for key, value in obj.items() for k, v in _flatten(value, f"{prefix}{key}.").items()}
    return {prefix.rstrip("."): obj}


def config_diff(config: PipelineConfig, defaults: PipelineConfig) -> dict[str, dict[str, Any]]:
    """Every config key whose value differs from the defaults: `{key: {default, value}}`."""
    ours, theirs = _flatten(config_dict(config)), _flatten(config_dict(defaults))
    return {k: {"default": theirs.get(k), "value": v} for k, v in ours.items() if theirs.get(k) != v}


GS_KEYS = ("iterations", "step", "finished", "num_gaussians", "train_s", "holdout_psnr", "holdout_ssim", "peak_vram_mb",
           "novel_views")
GS_IMAGES = ("plot_gs_training", "plot_gs_holdout", "plot_gs_depth_vs_mono", "gs_novel_strip")


def _extras(name: str, path: Path, out: Path) -> dict[str, Any]:
    """Report fields beyond the manifest: the 2DGS block of dense (its numbers, figures and video, paths relative to
    the report dir `out`), the planes and the geometry check of mesh_post, the export's files."""
    if name == s05_dense.STAGE:
        report = read_json(path / s05_dense.ARTIFACTS["report"])
        if report.get("dense_backend") != "2dgs":
            return {}
        rel = lambda key: os.path.relpath(path / s05_dense.GS_FIGURES[key], out)  # noqa: E731
        video = path / s05_dense.GS_FIGURES["gs_novel_video"]
        return {"gs": {
            **{k: report["gs"].get(k) for k in GS_KEYS},
            "images": [rel(k) for k in GS_IMAGES if (path / s05_dense.GS_FIGURES[k]).is_file()],
            "video": rel("gs_novel_video") if video.is_file() else None,
        }}
    if name == s06_mesh_post.STAGE:
        report = read_json(path / s06_mesh_post.ARTIFACTS["report"])
        qa = report["qa"]
        return {
            "planes": read_json(path / s06_mesh_post.ARTIFACTS["planes"]),
            "qa": qa,
            "derived": {"qa_p95_tau": qa["dense_to_mesh"]["p95"] / qa["tau"]},
        }
    if name == s07_export.STAGE:
        return {"files": read_json(path / s07_export.ARTIFACTS["metadata"])["files"]}
    return {}


def stage_entry(name: str, sequence_id: str | None, path: Path, out: Path) -> dict[str, Any]:
    """The summary entry of one stage dir `path` (paths relative to the report dir `out`); `missing` when the stage
    has no manifest."""
    entry: dict[str, Any] = {"stage": name, "sequence_id": sequence_id, "dir": os.path.relpath(path, out)}
    manifest = read_manifest(path)
    if manifest is None:
        return {**entry, "status": "missing", "duration_s": None, "metrics": {}, "warnings": []}
    entry |= {
        "status": manifest.status,
        "duration_s": manifest.duration_s,
        "finished": manifest.finished,
        "metrics": dict(manifest.metrics),
        "warnings": list(manifest.warnings),
        "backends": dict(manifest.backends),
        "tools": dict(manifest.tools),
        "images": [
            os.path.relpath(path / rel, out) for rel in manifest.artifacts.values() if rel.endswith(".png")
        ],
    }
    if manifest.status == "success":
        extras = _extras(name, path, out)
        entry["metrics"] |= extras.pop("derived", {})
        entry |= extras
    if name == s07_export.STAGE:
        entry["metrics"]["all_valid"] = manifest.status == "success"  # the stage fails on any invalid file
    return entry


def run_report(
    entries: list[dict[str, Any]],
    run: dict[str, Any],
    out: Path,
    options: ReportOptions | None,
    make_figures: Callable[[], dict[str, Any]] | None = None,
) -> StageOutput:
    """Write `summary.json`, `plots/timings.png` and `index.html` into the stage dir `out` from the stage `entries`
    (`stage_entry`) and the run info `run`; `make_figures` builds the figures once the timings plot exists (they are linked from the page).
    Threshold failures are warnings."""
    options = options or ReportOptions()
    t0 = time.perf_counter()
    summary = summarize(run, entries)
    summary["checks"] = check(summary, options.thresholds)
    summary["pass"] = all(c["pass"] is True for c in summary["checks"])
    several = len({e["sequence_id"] for e in entries} - {None}) > 1
    durations = {
        f"{e['stage']} ({e['sequence_id']})" if several and e["sequence_id"] else e["stage"]: e["duration_s"]
        for e in entries
        if e["duration_s"] is not None
    }
    plot_timings(durations, out / ARTIFACTS["plot_timings"])
    figures = make_figures() if make_figures else None
    write_json(summary, out / ARTIFACTS["summary"])
    figure_files = [f"{ARTIFACTS['figures']}/{name}" for name in (figures or {}).get("figures", [])]
    render(summary, out / ARTIFACTS["html"], ARTIFACTS["plot_timings"], figure_files)

    warnings = list((figures or {}).get("warnings", [])) + [
        f"threshold: {c['stage']}.{c['metric']} = {c['value']} (needs {c['op']} {c['threshold']})"
        for c in summary["checks"]
        if c["pass"] is not True
    ]
    for warning in warnings:
        log.warning(warning)
    failed = sum(c["pass"] is not True for c in summary["checks"])
    log.info(
        "report: %d stages, %d of %d thresholds pass in %.1f s",
        len(entries),
        len(summary["checks"]) - failed,
        len(summary["checks"]),
        time.perf_counter() - t0,
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "pass": summary["pass"],
            "checks": len(summary["checks"]),
            "checks_failed": failed,
            "stages": len(entries),
            "total_s": round(summary["total_s"], 1),
        },
        warnings=warnings,
    )


@register_stage
class ReportStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s07_export.STAGE,)
    config_section = "report"

    def run(self) -> StageOutput:
        ctx = self.ctx
        entries = []
        for cls in registered_stages():
            if cls.name == STAGE:
                continue
            for sequence_id in ctx.sequence_ids if cls.scope == "sequence" else [None]:
                entries.append(stage_entry(cls.name, sequence_id, ctx.stage_dir(cls.name, sequence_id), self.out))
        record = read_json(ctx.run_dir / RUN_MANIFEST_NAME) if (ctx.run_dir / RUN_MANIFEST_NAME).is_file() else {}
        run = {
            "scene": ctx.scene_dir.name,
            "run_id": ctx.run_dir.name,
            "sequence_ids": ctx.sequence_ids,
            "hardware": record.get("hardware"),
            "pipeline_version": s07_export.git_version(),
            "created": now_iso(),
            "config_diff": config_diff(ctx.config, load_config()),
        }
        (self.out / ARTIFACTS["figures"]).mkdir(parents=True, exist_ok=True)
        sequence_dirs = {sequence_id: ctx.sequence_dir(sequence_id) for sequence_id in ctx.sequence_ids}
        make_figures = (
            (lambda: build_figures(ctx.run_dir, sequence_dirs, self.out / ARTIFACTS["figures"], self.options.figures))
            if self.options.figures.enabled
            else None
        )
        return run_report(entries, run, self.out, self.options, make_figures)

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
