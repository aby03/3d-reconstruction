from dataclasses import dataclass, field
from typing import Literal

from recon.dense.filter import FilterOptions
from recon.dense.fusion import FusionOptions
from recon.dense.mono import MonoOptions
from recon.dense.openmvs import MeshOptions, OpenMvsOptions, PreviewOptions, RefineOptions
from recon.dense.simplerecon import SimpleReconOptions
from recon.dense.undistort import UndistortOptions


@dataclass(slots=True)
class GsOptions:
    long_side: int = 960
    iterations: int = 7000
    max_minutes: float | None = None
    max_gaussians: int = 500_000
    holdout_every: int = 8
    sh_degree: int = 3
    lambda_dssim: float = 0.2
    lambda_distortion: float = 0.01
    lambda_normal: float = 0.05
    mono_weight: float = 0.1
    init_mono_points: int = 100_000
    render_long_side: int = 640
    depth: Literal["median", "expected"] = "median"
    min_alpha: float = 0.5
    novel_views: int = 120
    seed: int | None = None


@dataclass(slots=True)
class DenseOptions:
    backend: Literal["openmvs", "2dgs"] = "openmvs"  # 2dgs: 2D Gaussian Splatting (gsplat), needs `pixi run -e gsplat`
    undistort: UndistortOptions = field(default_factory=UndistortOptions)
    openmvs: OpenMvsOptions = field(default_factory=OpenMvsOptions)
    filter: FilterOptions = field(default_factory=FilterOptions)
    depth_source: Literal["mono", "simplerecon"] = "mono"  # where the per-keyframe depth for fusion, fill and carving comes from
    mono: MonoOptions = field(default_factory=MonoOptions)
    simplerecon: SimpleReconOptions = field(default_factory=SimpleReconOptions)
    fusion: FusionOptions = field(default_factory=FusionOptions)
    mesh: MeshOptions = field(default_factory=MeshOptions)
    refine: RefineOptions = field(default_factory=RefineOptions)
    preview: PreviewOptions = field(default_factory=PreviewOptions)
    gs: GsOptions = field(default_factory=GsOptions)
