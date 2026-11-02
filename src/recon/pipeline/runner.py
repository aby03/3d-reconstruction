"""Runs the selected stages in order, skipping those whose results are up to date."""

import logging
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from recon.capture.open_camera_sensors import discover_files
from recon.io.tables import read_json, write_json
from recon.pipeline.config import PipelineConfig, dump_config
from recon.pipeline.context import RunContext
from recon.pipeline.run_dir import (
    RUN_MANIFEST_NAME,
    StageManifest,
    latest_run_dir,
    manifest_hash,
    new_run_dir,
    now_iso,
    read_manifest,
    scene_dir,
    setup_logging,
    write_manifest,
)
from recon.pipeline.stage import Stage, get_stage, registered_stages

log = logging.getLogger("recon.pipeline")

TRACEBACK_LINES = 20


class PipelineError(RuntimeError):
    pass


@dataclass(slots=True)
class StageRecord:
    stage: str
    sequence_id: str | None
    status: str  # "success" | "skipped" | "failed" | "interrupted"
    duration_s: float = 0.0
    config_hash: str = ""


@dataclass(slots=True)
class RunResult:
    status: str  # "success" | "failed" | "interrupted"
    stages: list[StageRecord] = field(default_factory=list)
    error: str | None = None


def select_stages(from_: str | None = None, to: str | None = None, only: str | None = None) -> list[type[Stage]]:
    """The stages to run, in canonical order: just `only`, or the range `from_` … `to` (both inclusive)."""
    stages = registered_stages()
    names = [cls.name for cls in stages]
    if only is not None:
        if from_ is not None or to is not None:
            raise PipelineError("--only can't be combined with --from or --to")
        return [get_stage(only)]
    start = names.index(get_stage(from_).name) if from_ else 0
    stop = names.index(get_stage(to).name) + 1 if to else len(names)
    if start >= stop:
        raise PipelineError(f"--from {from_} comes after --to {to}")
    return stages[start:stop]


def scene_path(input_dir: Path | str, config: PipelineConfig, scene: str | None = None) -> Path:
    """`<output_root>/<scene>/` of a capture; the scene defaults to the capture folder name."""
    return scene_dir(config.run.output_root, scene or discover_files(input_dir).root.name)


def _resumed_run_dir(scene_root: Path, resume: str, inputs: dict[str, Path]) -> Path:
    run_path = latest_run_dir(scene_root) if resume == "latest" else scene_root / "runs" / resume
    if run_path is None or not (run_path / RUN_MANIFEST_NAME).is_file():
        what = "no run" if resume == "latest" else f"no run {resume!r}"
        raise PipelineError(f"{what} to resume in {scene_root / 'runs'}")
    sequences = set(read_json(run_path / RUN_MANIFEST_NAME)["sequences"])
    if sequences != set(inputs):
        raise PipelineError(
            f"run {run_path.name} has sequences {', '.join(sorted(sequences))}, not {', '.join(inputs)}"
        )
    return run_path


def open_run(
    input_dir: Path | str,
    config: PipelineConfig,
    scene: str | None = None,
    resume: str | None = None,
    hardware: str | None = None,
) -> RunContext:
    """Open a run dir for one capture under `<output_root>/<scene>/` and log to its `logs/pipeline.log`.

    Without `resume` the run is new. `resume` (a run id, or "latest" for the newest run) reuses that run dir,
    so its scene stages can be skipped when up to date; it must have been made for the same sequences.
    `hardware` is the name of the preset in `config` (recorded in the run manifest and `config.resolved.yaml`).
    """
    files = discover_files(input_dir)
    scene_root = scene_dir(config.run.output_root, scene or files.root.name)
    inputs = {files.sequence_id: files.root}
    run_path = new_run_dir(scene_root) if resume is None else _resumed_run_dir(scene_root, resume, inputs)
    setup_logging(run_path / "logs" / "pipeline.log")
    if resume is not None:
        log.info("resuming run %s", run_path.name)
    header = f"# hardware preset: {f'configs/hardware/{hardware}.yaml' if hardware else 'none'}\n"
    (run_path / "config.resolved.yaml").write_text(header + dump_config(config))
    return RunContext(
        scene_dir=scene_root, run_dir=run_path, inputs=inputs, config=config, logger=log, hardware=hardware
    )


def _upstream_hashes(stage: Stage) -> dict[str, str | None]:
    hashes = {}
    for name in stage.requires:
        if get_stage(name).scope == "scene" or stage.scope == "sequence":
            hashes[name] = manifest_hash(stage.ctx.stage_dir(name, stage.sequence_id))
        else:
            for sid in stage.ctx.sequence_ids:
                hashes[f"{name}/{sid}"] = manifest_hash(stage.ctx.stage_dir(name, sid))
    return hashes


def _missing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.exists()]


def is_up_to_date(stage: Stage, config_hash: str, upstream: dict[str, str | None]) -> bool:
    """Finished successfully with this config on these upstream results, and every artifact is still there."""
    manifest = read_manifest(stage.out)
    return (
        manifest is not None
        and manifest.status == "success"
        and manifest.config_hash == config_hash
        and manifest.upstream == upstream
        and not _missing(stage.expected_artifacts() + [stage.out / rel for rel in manifest.artifacts.values()])
    )


def _run_stage(stage: Stage, config_hash: str, upstream: dict[str, str]) -> StageManifest:
    out = stage.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started, t_start = now_iso(), time.perf_counter()

    def manifest(status: str, **fields: Any) -> StageManifest:
        result = StageManifest(
            stage=stage.name,
            status=status,
            started=started,
            finished=now_iso(),
            duration_s=round(time.perf_counter() - t_start, 3),
            config_hash=config_hash,
            upstream=upstream,
            **fields,
        )
        write_manifest(result, out)
        return result

    try:
        output = stage.run()
        if missing := _missing(stage.expected_artifacts() + [out / rel for rel in output.artifacts.values()]):
            raise RuntimeError(f"missing artifacts: {', '.join(str(p.relative_to(out)) for p in missing)}")
    except KeyboardInterrupt:
        manifest("interrupted", warnings=["interrupted"])
        raise
    except Exception as e:
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-TRACEBACK_LINES:])
        return manifest("failed", warnings=[f"{type(e).__name__}: {e}"], error=tail)
    return manifest(
        "success",
        artifacts=output.artifacts,
        metrics=output.metrics,
        warnings=output.warnings,
        backends=output.backends,
        tools=output.tools,
    )


def _check_requirements(stage: Stage, selected: set[str]) -> None:
    """Upstream stages outside this run must already have finished successfully."""
    for name in stage.requires:
        if name in selected:
            continue
        if get_stage(name).scope == "scene":
            manifest = read_manifest(stage.ctx.stage_dir(name))
            if manifest is None or manifest.status != "success":
                raise PipelineError(
                    f"{stage.name} needs a successful {name} in run {stage.ctx.run_dir.name}: run `recon {name}` first"
                )
            continue
        sids = [stage.sequence_id] if stage.scope == "sequence" else stage.ctx.sequence_ids
        for sid in sids:
            manifest = read_manifest(stage.ctx.stage_dir(name, sid))
            if manifest is None or manifest.status != "success":
                raise PipelineError(f"{stage.name} needs a successful {name} of {sid}: run `recon {name}` first")


def run_stages(ctx: RunContext, stages: list[type[Stage]], force: str | None = None) -> RunResult:
    """Run `stages` in order; `force` re-runs that stage and everything after it. Stops at the first failure."""
    order = [cls.name for cls in registered_stages()]
    forced = set(order[order.index(get_stage(force).name) :]) if force else set()
    selected = {cls.name for cls in stages}
    result = RunResult(status="success")
    started = now_iso()

    def write_run_manifest() -> None:
        write_json(
            {
                "scene": ctx.scene_dir.name,
                "run_id": ctx.run_dir.name,
                "status": result.status,
                "started": started,
                "finished": now_iso(),
                "sequences": {sid: {"input": path} for sid, path in ctx.inputs.items()},
                "hardware": ctx.hardware,
                "stages": [asdict(r) for r in result.stages],
                "error": result.error,
            },
            ctx.run_dir / RUN_MANIFEST_NAME,
        )

    for cls in stages:
        for sid in ctx.sequence_ids if cls.scope == "sequence" else [None]:
            stage = cls(ctx, sid)
            label = f"{cls.name} [{sid}]" if sid else cls.name
            try:
                _check_requirements(stage, selected)
            except PipelineError as e:
                result.status, result.error = "failed", str(e)
                log.error("%s", e)
                write_run_manifest()
                return result

            config_hash, upstream = stage.config_hash(), _upstream_hashes(stage)
            if cls.name not in forced and is_up_to_date(stage, config_hash, upstream):
                log.info("%s is up to date, skipping (%s)", label, stage.out)
                result.stages.append(StageRecord(cls.name, sid, "skipped", 0.0, config_hash))
                write_run_manifest()
                continue

            log.info("running %s", label)
            try:
                manifest = _run_stage(stage, config_hash, upstream)
            except KeyboardInterrupt:
                result.status, result.error = "interrupted", f"{label} interrupted"
                result.stages.append(StageRecord(cls.name, sid, "interrupted", 0.0, config_hash))
                write_run_manifest()
                raise
            result.stages.append(StageRecord(cls.name, sid, manifest.status, manifest.duration_s, config_hash))
            if manifest.status != "success":
                result.status, result.error = "failed", f"{label} failed: {manifest.warnings[0]}"
                log.error("%s\n%s", result.error, manifest.error)
                write_run_manifest()
                return result
            log.info("%s finished in %.1f s: %s", label, manifest.duration_s, stage.out)
            write_run_manifest()
    write_run_manifest()
    return result
