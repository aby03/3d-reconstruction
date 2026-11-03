from dataclasses import dataclass, field

from recon.dense.filter import FilterOptions
from recon.dense.fusion import FusionOptions
from recon.dense.mono import MonoOptions
from recon.dense.openmvs import MeshOptions, OpenMvsOptions, PreviewOptions, RefineOptions
from recon.dense.undistort import UndistortOptions


@dataclass(slots=True)
class DenseOptions:
    undistort: UndistortOptions = field(default_factory=UndistortOptions)
    openmvs: OpenMvsOptions = field(default_factory=OpenMvsOptions)
    filter: FilterOptions = field(default_factory=FilterOptions)
    mono: MonoOptions = field(default_factory=MonoOptions)
    fusion: FusionOptions = field(default_factory=FusionOptions)
    mesh: MeshOptions = field(default_factory=MeshOptions)
    refine: RefineOptions = field(default_factory=RefineOptions)
    preview: PreviewOptions = field(default_factory=PreviewOptions)
