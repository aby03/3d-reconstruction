"""What a stage gets to work with: where the scene and the run live, the inputs and the config."""

import logging
from dataclasses import dataclass
from pathlib import Path

from recon.pipeline.config import PipelineConfig
from recon.pipeline.device import Backend, DeviceInfo, Preference, probe_device, resolve_backend
from recon.pipeline.run_dir import stage_dir


@dataclass(slots=True)
class RunContext:
    scene_dir: Path  # <output_root>/<scene>/
    run_dir: Path  # <scene_dir>/runs/<run_id>/
    inputs: dict[str, Path]  # sequence_id -> capture directory
    config: PipelineConfig
    logger: logging.Logger
    device: DeviceInfo | None = None  # probed on the first backend() call that needs it
    hardware: str | None = None  # name of the hardware preset in `config`

    @property
    def sequence_ids(self) -> list[str]:
        return list(self.inputs)

    def backend(self, capability: str, preference: Preference = "auto") -> Backend:
        """CUDA or CPU for `capability`; an `auto` preference follows `device.preference`."""
        if preference == "auto":
            preference = self.config.device.preference
        if preference == "cpu":
            return "cpu"
        if self.device is None:
            self.device = probe_device(self.config.device.gpu_index)
        return resolve_backend(capability, preference, self.device)

    def sequence_dir(self, sequence_id: str) -> Path:
        return self.scene_dir / "sequences" / sequence_id

    def stage_dir(self, name: str, sequence_id: str | None = None) -> Path:
        """`sequences/<sequence_id>/<dir_name>/` for a sequence stage, `runs/<run_id>/<dir_name>/` for a scene stage."""
        from recon.pipeline.stage import get_stage

        stage = get_stage(name)
        if stage.scope == "scene":
            return stage_dir(self.run_dir, stage.dir_name)
        if sequence_id is None:
            raise ValueError(f"stage {name!r} is per sequence: pass a sequence_id")
        return stage_dir(self.sequence_dir(sequence_id), stage.dir_name)
