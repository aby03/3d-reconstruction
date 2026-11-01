from dataclasses import dataclass, field

from recon.align.cam_imu import CamImuOptions
from recon.align.floor import FloorOptions
from recon.align.gravity import GravityOptions
from recon.align.manhattan import ManhattanOptions


@dataclass(slots=True)
class AlignOptions:
    gravity: GravityOptions = field(default_factory=GravityOptions)
    floor: FloorOptions = field(default_factory=FloorOptions)
    manhattan: ManhattanOptions = field(default_factory=ManhattanOptions)
    cam_imu: CamImuOptions = field(default_factory=CamImuOptions)
