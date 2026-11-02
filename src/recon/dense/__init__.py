from dataclasses import dataclass, field

from recon.dense.openmvs import MeshOptions, OpenMvsOptions
from recon.dense.undistort import UndistortOptions


@dataclass(slots=True)
class DenseOptions:
    undistort: UndistortOptions = field(default_factory=UndistortOptions)
    openmvs: OpenMvsOptions = field(default_factory=OpenMvsOptions)
    mesh: MeshOptions = field(default_factory=MeshOptions)
