"""Step 4: move the refined SfM model into the world frame W (Z up, gravity −Z, floor z = 0) in `04_alignment/`."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from recon.align import AlignOptions
from recon.align.apply import align_model, apply_transform, tum_lines
from recon.io.colmap import write_model
from recon.io.tables import read_json, write_json
from recon.pipeline.stage import Stage, StageOutput, register_stage
from recon.pipeline.stages import s03_sfm
from recon.pipeline.stages.common import gravity_vectors, read_keyframe_inputs
from recon.report import plots
from recon.sfm.quality import camera_centres

log = logging.getLogger(__name__)

STAGE = "align"
DIR_NAME = "04_alignment"
UNITS = "arbitrary"

ARTIFACTS = {
    "model": "sparse_aligned",
    "model_txt": "sparse_aligned_txt",
    "points": "sparse_points_aligned.ply",
    "transform": "transform.json",
    "trajectory": "trajectory_tum",
    "report": "reports/alignment_report.json",
    "plot_gravity": "reports/plots/gravity_residuals.png",
    "plot_height": "reports/plots/height_hist.png",
    "plot_trajectory": "reports/plots/trajectory_topdown.png",
}


def _deg(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _write_plots(out: Path, rec: pycolmap.Reconstruction, residuals: np.ndarray, floor_detected: bool) -> None:
    plots.plot_histogram(
        residuals, out / ARTIFACTS["plot_gravity"], "gravity residual (prior vs world down)", "degrees", bins=50
    )
    centres = np.array(list(camera_centres(rec).values()))
    points = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    plots.plot_height_histogram(
        points[:, 2], 0.0 if floor_detected else None, centres[:, 2], out / ARTIFACTS["plot_height"]
    )
    plots.plot_trajectory_topdown(centres, points, out / ARTIFACTS["plot_trajectory"], axes="xy")


def run_align(
    sfm_dir: Path, sequence_dirs: dict[str, Path], out: Path, options: AlignOptions | None = None, seed: int = 0
) -> StageOutput:
    """Run step 4 on the refined model of the sfm stage dir `sfm_dir`, with the keyframes of `sequence_dirs`
    (sequence_id -> sequence dir), writing into the stage dir `out`."""
    options = options or AlignOptions()
    timings = {}

    t0 = time.perf_counter()
    rec = pycolmap.Reconstruction(sfm_dir / s03_sfm.ARTIFACTS["model"])
    sfm_report = read_json(sfm_dir / s03_sfm.ARTIFACTS["report"])
    inputs = read_keyframe_inputs(sequence_dirs)
    gravity = gravity_vectors(inputs.gravity, 0.0)
    conf = dict(zip(inputs.gravity["image_name"], inputs.gravity["conf"].astype(float)))
    t_ns = dict(zip(inputs.images["image_name"], inputs.images["t_ns"].astype(int)))
    timings["load"] = time.perf_counter() - t0

    W_from_sfm, steps, diagnostics = align_model(rec, gravity, conf, options, seed)
    timings.update(diagnostics["timings"])
    warnings = diagnostics["warnings"]
    for warning in warnings:
        log.warning(warning)

    t0 = time.perf_counter()
    aligned = apply_transform(rec, W_from_sfm)
    write_model(aligned, out / ARTIFACTS["model"])
    write_model(aligned, out / ARTIFACTS["model_txt"], text=True)
    aligned.export_PLY(out / ARTIFACTS["points"])
    transform = {
        "W_from_sfm": W_from_sfm.to_dict(),
        "steps": steps,
        "units": UNITS,
        "up_axis": "+Z",
        "gravity": [0.0, 0.0, -1.0],
    }
    write_json(transform, out / ARTIFACTS["transform"])
    trajectory = out / ARTIFACTS["trajectory"]
    trajectory.mkdir(parents=True, exist_ok=True)
    for sequence_id, lines in tum_lines(aligned, t_ns).items():
        (trajectory / f"{sequence_id}.txt").write_text("".join(f"{line}\n" for line in lines))
    _write_plots(out, aligned, diagnostics["residuals_deg"], diagnostics["floor"]["detected"])
    timings["write"] = time.perf_counter() - t0

    points = np.array([p.xyz for p in aligned.points3D.values()]).reshape(-1, 3)
    lo, hi = np.percentile(points, [2, 98], axis=0)
    report: dict[str, Any] = {
        "input_model": f"{s03_sfm.DIR_NAME}/{s03_sfm.ARTIFACTS['model']}",
        "num_registered": aligned.num_reg_images(),
        "mean_reproj_px": aligned.compute_mean_reprojection_error(),
        "sfm": {k: sfm_report.get(k) for k in ("mapper_used", "gravity_prior_used", "gravity_spread_deg")},
        "gravity": diagnostics["gravity"],
        "floor": diagnostics["floor"],
        "W_from_sfm": transform["W_from_sfm"],
        "extent": {"min": lo.tolist(), "max": hi.tolist()},
        "units": UNITS,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
    }
    write_json(report, out / ARTIFACTS["report"])
    gravity_report, floor_report = report["gravity"], report["floor"]
    log.info(
        "aligned %d images: %s gravity (residual median %s deg), floor %s",
        report["num_registered"],
        gravity_report["method"],
        _deg(gravity_report["residual_median_deg"]),
        f"tilt {_deg(floor_report['tilt_to_gravity_deg'])} deg" if floor_report["detected"] else "not found",
    )
    return StageOutput(
        artifacts=dict(ARTIFACTS),
        metrics={
            "gravity_residual_median_deg": gravity_report["residual_median_deg"],
            "gravity_method": gravity_report["method"],
            "floor_detected": floor_report["detected"],
            "floor_tilt_deg": floor_report["tilt_to_gravity_deg"],
            "num_registered": report["num_registered"],
        },
        warnings=list(warnings),
    )


@register_stage
class AlignStage(Stage):
    name = STAGE
    dir_name = DIR_NAME
    scope = "scene"
    requires = (s03_sfm.STAGE,)
    config_section = "align"

    def hash_inputs(self) -> dict[str, Any]:
        return {"seed": self.ctx.config.run.seed}

    def run(self) -> StageOutput:
        sequence_dirs = {sid: self.ctx.sequence_dir(sid) for sid in self.ctx.sequence_ids}
        return run_align(
            self.ctx.stage_dir(s03_sfm.STAGE), sequence_dirs, self.out, self.options, self.ctx.config.run.seed
        )

    def expected_artifacts(self) -> list[Path]:
        return [self.out / rel for rel in ARTIFACTS.values()]
