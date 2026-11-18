from dataclasses import dataclass, field


def _default_thresholds() -> dict[str, dict[str, float | bool]]:
    return {
        "ingest": {"frames_valid_ratio": 0.99, "ref_jpeg_min_psnr_db": 25.0},
        "keyframes": {"max_gap_s": 1.0},
        "sfm": {"registered_ratio": 0.95, "mean_reproj_px": 1.0},
        "align": {"gravity_residual_median_deg": 3.0},
        "mesh_post": {"untextured_face_frac": 0.01, "wall_rms_max_frac": 0.0025, "qa_p95_tau": 2.0},
        "export": {"all_valid": True},
    }


@dataclass(slots=True)
class FigureOptions:
    enabled: bool = True  # render the figures of the run (needs an OpenGL window; a failure is a warning)
    width: int = 1400  # size of a rendered view in pixels
    height: int = 800
    azimuth: float = 35.0  # camera direction around the vertical axis and above the horizon, degrees
    elevation: float = 50.0
    zoom: float = 1.25
    clip_frac: float = 0.04  # views from above cut the model this fraction of the room height below the ceiling plane
    max_points: int = 1_500_000  # dense points drawn
    waypoints: int = 12  # numbered time steps on the camera path figure
    timeout_s: float = 1200.0  # the rendering process is stopped after this long


@dataclass(slots=True)
class HoleOptions:
    enabled: bool = True  # share of the surface the keyframes see (mono depth) that mesh_raw / the final mesh miss
    views: int = 24  # reference views, evenly spaced


@dataclass(slots=True)
class ReportOptions:
    figures: FigureOptions = field(default_factory=FigureOptions)
    holes: HoleOptions = field(default_factory=HoleOptions)
    # stage -> metric -> bound: a minimum for LOWER_BOUNDS metrics, a maximum for other numbers, equal for booleans
    thresholds: dict[str, dict[str, float | bool]] = field(default_factory=_default_thresholds)
