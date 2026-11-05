from dataclasses import dataclass, field

from recon.sfm.continuity import ContinuityOptions
from recon.sfm.features import FeatureOptions
from recon.sfm.mapper import MapperOptions
from recon.sfm.matching import MatchingOptions
from recon.sfm.priors import PriorOptions
from recon.sfm.refine import RefineOptions
from recon.sfm.repair import RepairOptions


@dataclass(slots=True)
class SfmOptions:
    initial_camera_model: str = "SIMPLE_RADIAL"
    features: FeatureOptions = field(default_factory=FeatureOptions)
    matching: MatchingOptions = field(default_factory=MatchingOptions)
    priors: PriorOptions = field(default_factory=PriorOptions)
    mapper: MapperOptions = field(default_factory=MapperOptions)
    refine: RefineOptions = field(default_factory=RefineOptions)
    continuity: ContinuityOptions = field(default_factory=ContinuityOptions)
    repair: RepairOptions = field(default_factory=RepairOptions)
