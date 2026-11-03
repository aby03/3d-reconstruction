from dataclasses import dataclass, field


@dataclass(slots=True)
class GlbOptions:
    double_sided: bool = True  # room interiors are seen from inside


@dataclass(slots=True)
class ExportOptions:
    glb: GlbOptions = field(default_factory=GlbOptions)
