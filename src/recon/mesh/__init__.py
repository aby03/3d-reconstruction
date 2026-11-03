from dataclasses import dataclass, field
from typing import Literal

from recon.mesh.cleanup import CleanupOptions
from recon.mesh.decimate import DecimateOptions
from recon.mesh.texturing import TextureOptions


@dataclass(slots=True)
class MeshPostOptions:
    input: Literal["refined", "raw"] = "refined"
    cleanup: CleanupOptions = field(default_factory=CleanupOptions)
    decimate: DecimateOptions = field(default_factory=DecimateOptions)
    texture: TextureOptions = field(default_factory=TextureOptions)
