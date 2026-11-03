from dataclasses import dataclass, field


def _default_thresholds() -> dict[str, dict[str, float | bool]]:
    return {
        "ingest": {"frames_valid_ratio": 0.99, "ref_jpeg_min_psnr_db": 25.0},
        "keyframes": {"max_gap_s": 1.0},
        "sfm": {"registered_ratio": 0.95, "mean_reproj_px": 1.0},
        "align": {"gravity_residual_median_deg": 3.0},
        "mesh_post": {"untextured_face_frac": 0.01, "wall_rms_max_frac": 0.002, "qa_p95_tau": 2.0},
        "export": {"all_valid": True},
    }


@dataclass(slots=True)
class ReportOptions:
    # stage -> metric -> bound: a minimum for LOWER_BOUNDS metrics, a maximum for other numbers, equal for booleans
    thresholds: dict[str, dict[str, float | bool]] = field(default_factory=_default_thresholds)
