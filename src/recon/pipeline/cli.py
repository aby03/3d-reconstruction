"""Command line entry point: `python -m recon.pipeline.cli <command>`."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from recon.capture.open_camera_sensors import LoaderOptions, OpenCameraSensorsLoader, discover_files
from recon.io.tables import to_jsonable
from recon.pipeline.run_dir import read_manifest, sequence_dir, setup_logging, stage_dir
from recon.keyframes.select import KeyframeOptions
from recon.pipeline.stages import s01_ingest, s02_keyframes
from recon.sensors.sync import SyncOptions, synchronize

log = logging.getLogger("recon.cli")


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


def _print_report(report: dict) -> None:
    rows = [(k, _format(v)) for k, v in _flatten(to_jsonable({k: v for k, v in report.items() if k != "warnings"}))]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")
    warnings = report.get("warnings", [])
    print(f"  {'warnings':<{width}}  {len(warnings)}")
    for w in warnings:
        print(f"    - {w}")


def _loader_options(args: argparse.Namespace) -> LoaderOptions:
    return LoaderOptions(verify_reference_jpegs=not args.no_verify_refs)


def _sync_options(args: argparse.Namespace) -> SyncOptions:
    return SyncOptions(camera_time_offset_ns=args.time_offset_ns, gravity_method=args.gravity_method)


def cmd_ingest(args: argparse.Namespace) -> int:
    files = discover_files(args.input)
    seq_dir = sequence_dir(args.output_root, args.scene or files.root.name, files.sequence_id)
    out = stage_dir(seq_dir, s01_ingest.DIR_NAME)
    setup_logging(seq_dir / "logs" / "pipeline.log")

    existing = read_manifest(out)
    if existing is not None and existing.status == "success" and not args.force:
        log.info("%s is already done (finished %s); use --force to re-run", out, existing.finished)
        return 0

    manifest = s01_ingest.run_ingest(files.root, seq_dir, _loader_options(args), _sync_options(args))
    print(f"ingest {manifest.status} in {manifest.duration_s:.1f} s -> {out}")
    _print_report({**manifest.metrics, "warnings": manifest.warnings})
    return 0


def cmd_keyframes(args: argparse.Namespace) -> int:
    files = discover_files(args.input)
    seq_dir = sequence_dir(args.output_root, args.scene or files.root.name, files.sequence_id)
    out = stage_dir(seq_dir, s02_keyframes.DIR_NAME)
    setup_logging(seq_dir / "logs" / "pipeline.log")

    ingest = read_manifest(stage_dir(seq_dir, s01_ingest.DIR_NAME))
    if ingest is None or ingest.status != "success":
        log.error("no successful ingest in %s; run `recon ingest` first", seq_dir)
        return 1
    existing = read_manifest(out)
    if existing is not None and existing.status == "success" and not args.force:
        log.info("%s is already done (finished %s); use --force to re-run", out, existing.finished)
        return 0

    manifest = s02_keyframes.run_keyframes(seq_dir, KeyframeOptions(uniform_fps=args.uniform_fps))
    print(f"keyframes {manifest.status} in {manifest.duration_s:.1f} s -> {out}")
    _print_report({**manifest.metrics, "warnings": manifest.warnings})
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    setup_logging()
    sequence = OpenCameraSensorsLoader(args.input, _loader_options(args)).load()
    sync = synchronize(sequence, _sync_options(args))
    print(f"{sequence.sequence_id} ({sequence.source_type}) at {sequence.root}")
    _print_report(s01_ingest.sync_report(sequence, sync))
    return 0


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="capture directory (root or stamp dir)")
    parser.add_argument("--time-offset-ns", type=int, default=0, help="added to frame times before sensor lookup")
    parser.add_argument("--gravity-method", choices=["complementary", "lowpass"], default="complementary")
    parser.add_argument("--no-verify-refs", action="store_true", help="skip the reference JPEG check")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recon", description="Phone video + IMU to a textured room mesh.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="load, validate and sync a capture; write 01_ingest/")
    _add_capture_args(ingest)
    ingest.add_argument("--scene", help="scene name (default: the capture folder name)")
    ingest.add_argument("--output-root", type=Path, default=Path("outputs"))
    ingest.add_argument("--force", action="store_true", help="re-run even if a successful result exists")
    ingest.set_defaults(func=cmd_ingest)

    keyframes = commands.add_parser("keyframes", help="select keyframes, extract them, write IMU priors; 02_keyframes/")
    keyframes.add_argument("--input", required=True, type=Path, help="capture directory (root or stamp dir)")
    keyframes.add_argument("--scene", help="scene name (default: the capture folder name)")
    keyframes.add_argument("--output-root", type=Path, default=Path("outputs"))
    keyframes.add_argument(
        "--uniform-fps", type=float, default=KeyframeOptions().uniform_fps, help="keyframes per second of video"
    )
    keyframes.add_argument("--force", action="store_true", help="re-run even if a successful result exists")
    keyframes.set_defaults(func=cmd_keyframes)

    inspect = commands.add_parser("inspect", help="load and check a capture, print a summary; writes nothing")
    _add_capture_args(inspect)
    inspect.set_defaults(func=cmd_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
