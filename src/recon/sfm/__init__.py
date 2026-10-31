from dataclasses import dataclass, field

from recon.sfm.features import FeatureOptions
from recon.sfm.mapper import MapperOptions
from recon.sfm.matching import MatchingOptions
from recon.sfm.priors import PriorOptions


@dataclass(slots=True)
class SfmOptions:
    initial_camera_model: str = "SIMPLE_RADIAL"
    features: FeatureOptions = field(default_factory=FeatureOptions)
    matching: MatchingOptions = field(default_factory=MatchingOptions)
    priors: PriorOptions = field(default_factory=PriorOptions)
    mapper: MapperOptions = field(default_factory=MapperOptions)
