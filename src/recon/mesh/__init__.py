from dataclasses import dataclass, field
from typing import Literal

from recon.mesh.cleanup import CleanupOptions
from recon.mesh.consistency import ConsistencyOptions
from recon.mesh.decimate import DecimateOptions
from recon.mesh.holes import HoleOptions
from recon.mesh.merge import MergeOptions
from recon.mesh.planes import PlaneOptions
from recon.mesh.roi import RoiOptions
from recon.mesh.texture_fix import TextureFixOptions
from recon.mesh.texturing import TextureOptions


@dataclass(slots=True)
class MeshPostOptions:
    input: Literal["refined", "raw"] = "refined"
    roi: RoiOptions = field(default_factory=RoiOptions)
    merge: MergeOptions = field(default_factory=MergeOptions)
    cleanup: CleanupOptions = field(default_factory=CleanupOptions)
    consistency: ConsistencyOptions = field(default_factory=ConsistencyOptions)
    holes: HoleOptions = field(default_factory=HoleOptions)
    planes: PlaneOptions = field(default_factory=PlaneOptions)
    decimate: DecimateOptions = field(default_factory=DecimateOptions)
    texture: TextureOptions = field(default_factory=TextureOptions)
    texture_fix: TextureFixOptions = field(default_factory=TextureFixOptions)
