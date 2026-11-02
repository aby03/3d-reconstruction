"""Stage API: what a pipeline step declares (name, directory, scope, upstream, config) and how it runs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from recon.pipeline.config import config_hash
from recon.pipeline.context import RunContext


@dataclass(slots=True)
class StageOutput:
    artifacts: dict[str, str]  # name -> path relative to the stage dir
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    backends: dict[str, str] = field(default_factory=dict)  # operation -> backend that ran it, e.g. decode: nvdec
    tools: dict[str, str | None] = field(default_factory=dict)  # external tool or library -> version


class Stage(ABC):
    name: ClassVar[str]
    dir_name: ClassVar[str]  # "NN_<name>"; the number gives the canonical order
    scope: ClassVar[Literal["sequence", "scene"]]  # sequence: once per input recording; scene: once per run
    requires: ClassVar[tuple[str, ...]] = ()
    config_section: ClassVar[str]

    def __init__(self, ctx: RunContext, sequence_id: str | None = None) -> None:
        self.ctx = ctx
        self.sequence_id = sequence_id

    @property
    def out(self) -> Path:
        return self.ctx.stage_dir(self.name, self.sequence_id)

    @property
    def options(self) -> Any:
        return getattr(self.ctx.config, self.config_section)

    def hash_inputs(self) -> dict[str, Any]:
        """Inputs besides the config section that change the result (other sections, input paths)."""
        return {}

    def config_hash(self) -> str:
        return config_hash(self.options, self.hash_inputs())

    @abstractmethod
    def run(self) -> StageOutput:
        """Write the stage's results into `self.out`, which exists and is empty."""

    @abstractmethod
    def expected_artifacts(self) -> list[Path]:
        """Files and directories that a finished run always leaves in `self.out`."""


_REGISTRY: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    if cls.name in _REGISTRY:
        raise ValueError(f"stage {cls.name!r} is registered twice")
    _REGISTRY[cls.name] = cls
    return cls


def registered_stages() -> list[type[Stage]]:
    """Every stage in canonical order."""
    import recon.pipeline.stages  # noqa: F401  (registers the stages)

    return sorted(_REGISTRY.values(), key=lambda cls: cls.dir_name)


def get_stage(name: str) -> type[Stage]:
    stages = {cls.name: cls for cls in registered_stages()}
    if name not in stages:
        raise KeyError(f"unknown stage {name!r}; stages: {', '.join(stages)}")
    return stages[name]
