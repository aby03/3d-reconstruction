"""Command line entry point: `python -m recon.pipeline.cli <command>`."""

import argparse
import sys
from dataclasses import asdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from recon.capture.open_camera_sensors import OpenCameraSensorsLoader
from recon.io.tables import to_jsonable
from recon.pipeline.config import ConfigError, PipelineConfig, load_config
from recon.pipeline.device import CAPABILITIES, probe_device, resolve_backend, tool_versions
from recon.pipeline.run_dir import read_manifest, setup_logging
from recon.pipeline.runner import PipelineError, open_run, run_stages, select_stages
from recon.pipeline.stage import registered_stages
from recon.pipeline.stages import s01_ingest, s02_keyframes
from recon.sensors.sync import synchronize


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(obj, dict) and obj:
        return [row for k, v in obj.items() for row in _flatten(v, f"{prefix}{k}.")]
    return [(prefix.rstrip("."), obj)]


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list) and all(isinstance(v, float) for v in value):
        return "[" + ", ".join(f"{v:.3f}" for v in value) + "]"
    return str(value)


def _print_table(values: dict) -> int:
    """Print flattened key/value rows; returns the key column width."""
    rows = [(k, _format(v)) for k, v in _flatten(to_jsonable(values))]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")
    return width


def _print_report(report: dict) -> None:
    width = _print_table({k: v for k, v in report.items() if k != "warnings"})
    warnings = report.get("warnings", [])
    print(f"  {'warnings':<{width}}  {len(warnings)}")
    for w in warnings:
        print(f"    - {w}")


def _overrides(args: argparse.Namespace) -> list[str]:
    """`--set` values followed by the shortcut flags that were given (later wins)."""
    shortcuts = {
        "output_root": "run.output_root",
        "time_offset_ns": "ingest.camera_time_offset_ns",
        "gravity_method": "ingest.gravity_method",
        "uniform_fps": "keyframes.uniform_fps",
    }
    overrides = list(args.set)
    overrides += [f"{key}={getattr(args, a)}" for a, key in shortcuts.items() if getattr(args, a, None) is not None]
    if getattr(args, "no_verify_refs", False):
        overrides.append("loader.verify_reference_jpegs=false")
    return overrides


def _config(args: argparse.Namespace) -> PipelineConfig:
    return load_config(args.config, _overrides(args))


def _run(args: argparse.Namespace, from_: str | None, to: str | None, only: str | None, force: str | None) -> int:
    config = _config(args)
    stages = select_stages(from_, to, only)
    ctx = open_run(args.input, config, args.scene)
    try:
        result = run_stages(ctx, stages, force)
    except KeyboardInterrupt:
        print(f"interrupted; run record in {ctx.run_dir}")
        return 130

    for record in result.stages:
        out = ctx.stage_dir(record.stage, record.sequence_id)
        print(f"{record.stage} {record.status} in {record.duration_s:.1f} s -> {out}")
        if record.status == "success":
            manifest = read_manifest(out)
            _print_report({**manifest.metrics, "warnings": manifest.warnings})
    if result.error:
        print(f"error: {result.error}")
    print(f"run record: {ctx.run_dir}")
    return 0 if result.status == "success" else 1


def cmd_run(args: argparse.Namespace) -> int:
    return _run(args, args.from_, args.to, args.only, args.force)


def cmd_stage(args: argparse.Namespace) -> int:
    return _run(args, None, None, args.stage, args.stage if args.force else None)


def cmd_list_stages(args: argparse.Namespace) -> int:
    for cls in registered_stages():
        requires = ", ".join(cls.requires) or "-"
        print(f"  {cls.name:<12} {cls.dir_name:<14} {cls.scope:<9} {cls.config_section:<10} requires: {requires}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    info = probe_device(args.gpu_index)
    versions = tool_versions()
    print("device:")
    _print_table(asdict(info))
    print("backends (device.preference=auto):")
    _print_table({cap: resolve_backend(cap, "auto", info) for cap in CAPABILITIES})
    print("versions:")
    _print_table({name: version or "MISSING" for name, version in versions.items()})
    if missing := [name for name, version in versions.items() if version is None]:
        print(f"missing: {', '.join(missing)}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    setup_logging()
    config = _config(args)
    sequence = OpenCameraSensorsLoader(args.input, config.loader).load()
    sync = synchronize(sequence, config.ingest)
    print(f"{sequence.sequence_id} ({sequence.source_type}) at {sequence.root}")
    _print_report(s01_ingest.sync_report(sequence, sync))
    return 0


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, action="append", default=[], help="YAML config file layered on the defaults"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config key, e.g. keyframes.uniform_fps=2",
    )


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="capture directory (root or stamp dir)")
    _add_config_args(parser)
    parser.add_argument("--time-offset-ns", type=int, help="added to frame times before sensor lookup")
    parser.add_argument("--gravity-method", choices=["complementary", "lowpass"])
    parser.add_argument("--no-verify-refs", action="store_true", help="skip the reference JPEG check")


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene", help="scene name (default: the capture folder name)")
    parser.add_argument("--output-root", type=Path, help="same as --set run.output_root=DIR (default: outputs)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recon", description="Phone video + IMU to a textured room mesh.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the pipeline stages, skipping those that are up to date")
    _add_capture_args(run)
    _add_output_args(run)
    names = [cls.name for cls in registered_stages()]
    run.add_argument("--from", dest="from_", choices=names, metavar="STAGE", help="first stage to run")
    run.add_argument("--to", choices=names, metavar="STAGE", help="last stage to run")
    run.add_argument("--only", choices=names, metavar="STAGE", help="run just this stage")
    run.add_argument("--force", choices=names, metavar="STAGE", help="re-run this stage and everything after it")
    run.set_defaults(func=cmd_run)

    ingest = commands.add_parser(
        "ingest", help="load, validate and sync a capture; write 01_ingest/ (run --only ingest)"
    )
    _add_capture_args(ingest)
    _add_output_args(ingest)
    ingest.add_argument("--force", action="store_true", help="re-run even if the result is up to date")
    ingest.set_defaults(func=cmd_stage, stage=s01_ingest.STAGE)

    keyframes = commands.add_parser(
        "keyframes", help="select keyframes, extract them, write IMU priors; 02_keyframes/ (run --only keyframes)"
    )
    keyframes.add_argument("--input", required=True, type=Path, help="capture directory (root or stamp dir)")
    _add_config_args(keyframes)
    _add_output_args(keyframes)
    keyframes.add_argument("--uniform-fps", type=float, help="same as --set keyframes.uniform_fps=F (default: 3)")
    keyframes.add_argument("--force", action="store_true", help="re-run even if the result is up to date")
    keyframes.set_defaults(func=cmd_stage, stage=s02_keyframes.STAGE)

    inspect = commands.add_parser("inspect", help="load and check a capture, print a summary; writes nothing")
    _add_capture_args(inspect)
    inspect.set_defaults(func=cmd_inspect)

    doctor = commands.add_parser("doctor", help="print the GPU, which operations can use CUDA, and tool versions")
    doctor.add_argument("--gpu-index", type=int, default=0, help="GPU to probe (default: 0)")
    doctor.set_defaults(func=cmd_doctor)

    list_stages = commands.add_parser("list-stages", help="print the stages in pipeline order")
    list_stages.set_defaults(func=cmd_list_stages)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, PipelineError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
