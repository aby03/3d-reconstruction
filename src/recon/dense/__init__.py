from dataclasses import dataclass, field
from typing import Literal

from recon.dense.filter import FilterOptions
from recon.dense.fusion import FusionOptions
from recon.dense.mono import MonoOptions
from recon.dense.openmvs import MeshOptions, OpenMvsOptions, PreviewOptions, RefineOptions
from recon.dense.simplerecon import SimpleReconOptions
from recon.dense.undistort import UndistortOptions


@dataclass(slots=True)
class DenseOptions:
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
