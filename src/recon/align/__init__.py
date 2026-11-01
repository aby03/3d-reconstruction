from dataclasses import dataclass, field

from recon.align.floor import FloorOptions
from recon.align.gravity import GravityOptions


@dataclass(slots=True)
class AlignOptions:
    gravity: GravityOptions = field(default_factory=GravityOptions)
    floor: FloorOptions = field(default_factory=FloorOptions)
