"""Step 1: load and validate a capture, sync sensors to frames, and write `01_ingest/`."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recon.capture.open_camera_sensors import LoaderOptions, OpenCameraSensorsLoader
from recon.core.types import CaptureSequence, FrameSensorData, SensorKind
from recon.io.tables import write_csv, write_json, write_npz
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.report import plots
from recon.sensors.gyro import chain_rotations
from recon.sensors.sync import COLUMNS, GRAV_CAM_COLUMNS, SyncOptions, synchronize

log = logging.getLogger(__name__)

STAGE = "ingest"
DIR_NAME = "01_ingest"
FRAMES_COLUMNS = ["idx", "t_ns", "pts_s", "dt_ms", "dropped_before", "ref_jpeg"]
SENSOR_COLUMNS = ["t_ns", "x", "y", "z"]

ARTIFACTS = {
    "sequence": "sequence.json",
    "frames": "frames.csv",
    **{f"sensors/{kind}": f"sensors/{kind}.csv" for kind in SensorKind},
    "frame_sensor_data": "sync/frame_sensor_data.csv",
    "frame_rotations": "sync/frame_rotations.npz",
    "sync_report": "reports/sync_report.json",
    "plot_timeline": "reports/plots/timeline.png",
    "plot_sensor_rates": "reports/plots/sensor_rates.png",
    "plot_gravity": "reports/plots/gravity.png",
    "plot_omega": "reports/plots/omega.png",
}


def sequence_info(sequence: CaptureSequence) -> dict[str, Any]:
    """The `sequence.json` record: where the capture is, its video and sensors, and the camera-IMU extrinsics."""
    cameras = {}
    for camera_id, cam in sequence.cameras.items():
        duration_s = float(cam.frame_t_ns[-1] - cam.frame_t_ns[0]) * 1e-9 if cam.num_frames else 0.0
        cameras[camera_id] = {
            "path": cam.path,
            "codec": cam.codec,
            "stored_size": cam.stored_size,
            "upright_size": cam.upright_size,
            "rotation_deg": cam.rotation_deg,
            "num_frames": cam.num_frames,
            "nominal_fps": cam.nominal_fps,
            "avg_fps": (cam.num_frames - 1) / duration_s if duration_s > 0 else None,
            "duration_s": duration_s,
            "t_first_ns": cam.frame_t_ns[0] if cam.num_frames else None,
            "t_last_ns": cam.frame_t_ns[-1] if cam.num_frames else None,
        }
    return {
        "sequence_id": sequence.sequence_id,
        "source_type": sequence.source_type,
        "root": sequence.root,
        "clock": {"units": "ns", "note": "capture clock of this sequence; not comparable across sequences"},
        "cameras": cameras,
        "sensors": {
            kind: {"n": len(s.t_ns), "rate_hz": s.rate_hz, "units": s.units, "frame": s.frame, "file": s.source_file}
            for kind, s in sequence.sensors.items()
        },
        "extrinsics": {
            camera_id: {
                "R_cam_imu": e.R_cam_imu,
                "t_cam_imu": e.t_cam_imu,
                "time_offset_ns": e.time_offset_ns,
                "source": e.source,
            }
            for camera_id, e in sequence.extrinsics.items()
        },
        "device_orientation_deg": sequence.checks.get("device_orientation", {}).get("deg"),
        "metadata": sequence.metadata,
    }


def sync_report(sequence: CaptureSequence, sync: FrameSensorData, camera_id: str = "cam0") -> dict[str, Any]:
    """Loader checks and sync results in one flat-ish summary (`reports/sync_report.json`)."""
    cam = sequence.cameras[camera_id]
    checks, timing = sequence.checks, sequence.checks["frame_timing"]
    duration_s = float(cam.frame_t_ns[-1] - cam.frame_t_ns[0]) * 1e-9

    sensors = {}
    for kind, stats in checks.get("sensors", {}).items():
        synced = sync.report["sensors"].get(kind, {})
        sensors[kind] = {
            "n": stats["n"],
            "rate_hz": stats["rate_hz"],
            "coverage_start_ms": synced.get("coverage_start_ms"),
            "coverage_end_ms": synced.get("coverage_end_ms"),
            "max_gap_ms": stats["max_gap_ms"],
            "long_gaps": stats["long_gaps"],
            "dup_removed": stats["duplicates"],
            "out_of_order": stats["out_of_order"],
            "valid_ratio": synced.get("valid_ratio"),
            "max_dt_ms": synced.get("max_dt_ms"),
        }

    refs = checks.get("reference_images")
    ref_psnr = None
    if refs and refs["checked"]:
        ref_psnr = {
            "checked": refs["checked"],
            "min": refs["min_psnr_db"],
            "median": float(np.median(list(refs["psnr_db"].values()))),
            "best_rotation": refs["best_rotation"],
            "rotation_mismatch": refs["rotation_mismatch"],
        }

    return {
        "sequence_id": sequence.sequence_id,
        "num_frames": cam.num_frames,
        "duration_s": duration_s,
        "fps_nominal": cam.nominal_fps,
        "fps_avg": (cam.num_frames - 1) / duration_s if duration_s > 0 else None,
        "dropped_frames": timing["drops_t"],
        "frame_timing_ok": timing["ok"],
        "pts_ts_corr": timing["delta_corr"],
        "pts_ts_resid_max_ms": timing["residual_max_ms"],
        "pts_ts_resid_std_ms": timing["residual_std_ms"],
        "camera_time_offset_ns": sync.report["camera_time_offset_ns"],
        "sensors": sensors,
        "frames_valid_ratio": sync.report["frames_valid_ratio"],
        "ref_jpeg_psnr_db": ref_psnr,
        "gravity_method": sync.report["gravity"]["method"],
        "gravity_mean_cam": sync.report["gravity"]["grav_cam_mean"],
        "grav_conf_median": sync.report["gravity"]["grav_conf_median"],
        "device_orientation_deg": checks.get("device_orientation", {}).get("deg"),
        "warnings": checks.get("warnings", []) + sync.report["warnings"],
    }


def _frames_table(sequence: CaptureSequence, camera_id: str = "cam0") -> pd.DataFrame:
    cam = sequence.cameras[camera_id]
    dropped = np.zeros(cam.num_frames, dtype=bool)
    dropped[sequence.checks["frame_timing"]["drops_t"]] = True
    dt_ms = np.full(cam.num_frames, np.nan)
    dt_ms[1:] = np.diff(cam.frame_t_ns) * 1e-6
    refs = sequence.reference_images
    return pd.DataFrame(
        {
            "idx": np.arange(cam.num_frames),
            "t_ns": cam.frame_t_ns,
            "pts_s": cam.pts_s,
            "dt_ms": dt_ms,
            "dropped_before": dropped,
            "ref_jpeg": [refs[i].name if i in refs else "" for i in range(cam.num_frames)],
        }
    )


def _write_plots(out: Path, sequence: CaptureSequence, sync: FrameSensorData, camera_id: str = "cam0") -> None:
    cam = sequence.cameras[camera_id]
    t0 = int(cam.frame_t_ns[0])
    df = sync.frames
    t_s = df["t_s"].to_numpy()
    sensors = {str(kind): s for kind, s in sequence.sensors.items()}
    drops = sequence.checks["frame_timing"]["drops_t"]

    plots.plot_sensor_timeline(sensors, t0, (0.0, float(t_s[-1])), out / ARTIFACTS["plot_timeline"])
    plots.plot_sample_rates(sensors, out / ARTIFACTS["plot_sensor_rates"])
    plots.plot_vector_components(
        t_s,
        df[GRAV_CAM_COLUMNS].to_numpy(),
        GRAV_CAM_COLUMNS,
        out / ARTIFACTS["plot_gravity"],
        ylabel="gravity (down), camera frame",
    )
    plots.plot_series_with_events(
        t_s,
        df["omega_norm"].to_numpy(),
        t_s[drops],
        out / ARTIFACTS["plot_omega"],
        ylabel="|ω| [rad/s]",
        event_label="dropped frame",
    )


def _write_artifacts(out: Path, sequence: CaptureSequence, sync: FrameSensorData) -> dict[str, str]:
    artifacts = dict(ARTIFACTS)
    write_json(sequence_info(sequence), out / ARTIFACTS["sequence"])
    write_csv(_frames_table(sequence), out / ARTIFACTS["frames"], FRAMES_COLUMNS)

    for kind in SensorKind:
        key = f"sensors/{kind}"
        if (stream := sequence.sensors.get(kind)) is None:
            del artifacts[key]
            continue
        df = pd.DataFrame(stream.values, columns=SENSOR_COLUMNS[1:])
        df.insert(0, "t_ns", stream.t_ns)
        write_csv(df, out / ARTIFACTS[key], SENSOR_COLUMNS)

    write_csv(sync.frames, out / ARTIFACTS["frame_sensor_data"], COLUMNS)
    if sync.dR_prev is None:
        del artifacts["frame_rotations"]
    else:
        write_npz(
            out / ARTIFACTS["frame_rotations"],
            idx=sync.frames["idx"].to_numpy(),
            t_ns=sync.frames["t_ns"].to_numpy(),
            dR_prev=sync.dR_prev,
            R_cum=chain_rotations(sync.dR_prev),
        )
    _write_plots(out, sequence, sync)
    return artifacts


def run_ingest(
    input_dir: Path | str,
    out: Path,
    loader_opts: LoaderOptions | None = None,
    sync_opts: SyncOptions | None = None,
) -> StageOutput:
    """Run step 1 on one capture, writing into the existing stage dir `out` (`01_ingest/`)."""
    t_start = time.perf_counter()
    log.info("loading %s", input_dir)
    sequence = OpenCameraSensorsLoader(input_dir, loader_opts).load()
    t_load = time.perf_counter() - t_start
    log.info("synchronizing sensors to %d frames", sequence.cameras["cam0"].num_frames)
    sync = synchronize(sequence, sync_opts)
    t_sync = time.perf_counter() - t_start - t_load

    artifacts = _write_artifacts(out, sequence, sync)
    report = sync_report(sequence, sync)
    write_json(report, out / ARTIFACTS["sync_report"])
    return StageOutput(
        artifacts=artifacts,
        metrics={
            "num_frames": report["num_frames"],
            "num_dropped": len(report["dropped_frames"]),
            "frames_valid_ratio": report["frames_valid_ratio"],
            "ref_jpeg_min_psnr_db": (report["ref_jpeg_psnr_db"] or {}).get("min"),
            "gravity_mean_cam": report["gravity_mean_cam"],
            "load_s": round(t_load, 3),
            "sync_s": round(t_sync, 3),
        },
        warnings=report["warnings"],
    )


@register_stage
class IngestStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "sequence"
    config_section = "ingest"

    @property
    def input_dir(self) -> Path:
        return self.ctx.inputs[self.sequence_id]

    def hash_inputs(self) -> dict[str, Any]:
        return {"loader": self.ctx.config.loader, "input": str(self.input_dir.resolve())}

    def run(self) -> StageOutput:
        return run_ingest(self.input_dir, self.out, self.ctx.config.loader, self.options)

    def expected_artifacts(self) -> list[Path]:
        optional = {key for key in ARTIFACTS if key.startswith("sensors/")} | {"frame_rotations"}
        return [self.out / rel for key, rel in ARTIFACTS.items() if key not in optional]
