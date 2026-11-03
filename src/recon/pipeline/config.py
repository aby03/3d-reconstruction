"""Pipeline config: `configs/default.yaml`, a hardware preset, `--config` files, then `--set a.b=c`, validated into
option dataclasses."""

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from recon.align import AlignOptions
from recon.capture.open_camera_sensors import LoaderOptions
from recon.dense import DenseOptions
from recon.io.tables import to_jsonable
from recon.keyframes.select import KeyframeOptions
from recon.mesh import MeshPostOptions
from recon.pipeline.device import DeviceInfo, DeviceOptions, Preference, probe_device
from recon.sensors.sync import SyncOptions
from recon.sfm import SfmOptions

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
HARDWARE_DIR = DEFAULT_CONFIG.parent / "hardware"
LARGE_VRAM_MB = 10 * 1024
MIN_VRAM_MB = 4 * 1024


class ConfigError(ValueError):
    pass


@dataclass(slots=True)
class RunOptions:
    output_root: Path = Path("outputs")
    seed: int = 0


@dataclass(slots=True)
class PipelineConfig:
    run: RunOptions = field(default_factory=RunOptions)
    device: DeviceOptions = field(default_factory=DeviceOptions)
    loader: LoaderOptions = field(default_factory=LoaderOptions)
    ingest: SyncOptions = field(default_factory=SyncOptions)
    keyframes: KeyframeOptions = field(default_factory=KeyframeOptions)
    sfm: SfmOptions = field(default_factory=SfmOptions)
    align: AlignOptions = field(default_factory=AlignOptions)
    dense: DenseOptions = field(default_factory=DenseOptions)
    mesh_post: MeshPostOptions = field(default_factory=MeshPostOptions)


SECTIONS: dict[str, type] = {f.name: f.type for f in dataclasses.fields(PipelineConfig)}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def parse_override(text: str) -> dict:
    """`a.b=c` -> {"a": {"b": c}}, with `c` parsed as a YAML scalar or list."""
    key, sep, value = text.partition("=")
    if not sep or not key or any(not part for part in key.split(".")):
        raise ConfigError(f"bad override {text!r}, expected key.path=value")
    tree: Any = yaml.safe_load(value) if value else None
    for part in reversed(key.split(".")):
        tree = {part: tree}
    return tree


def _check_keys(name: str, cls: type, data: Any) -> None:
    """Unknown keys are errors, also inside nested option dataclasses (pydantic ignores them on dataclasses)."""
    if not isinstance(data, dict):
        raise ConfigError(f"{name}: expected a mapping, got {data!r}")
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    if unknown := sorted(set(data) - set(fields)):
        raise ConfigError(f"unknown config key(s): {', '.join(f'{name}.{k}' for k in unknown)}")
    for key, value in data.items():
        if dataclasses.is_dataclass(fields[key]):
            _check_keys(f"{name}.{key}", fields[key], value)


def _validate_section(name: str, cls: type, data: Any) -> Any:
    _check_keys(name, cls, data)
    try:
        return TypeAdapter(cls).validate_python(data)
    except ValidationError as e:
        lines = [f"{'.'.join(map(str, (name, *err['loc'])))}: {err['msg']}" for err in e.errors()]
        raise ConfigError("invalid config:\n  " + "\n  ".join(lines)) from None


def validate_config(data: dict) -> PipelineConfig:
    if unknown := sorted(set(data) - set(SECTIONS)):
        raise ConfigError(f"unknown config section(s): {', '.join(unknown)}")
    return PipelineConfig(**{name: _validate_section(name, cls, data.get(name, {})) for name, cls in SECTIONS.items()})


def hardware_presets() -> list[str]:
    return sorted(p.stem for p in HARDWARE_DIR.glob("*.yaml"))


def select_hardware(info: DeviceInfo, preference: Preference = "auto") -> str | None:
    """The preset for this machine: `cpu` without CUDA OpenMVS (or when the CPU is asked for), else by VRAM
    (`cuda_large` from 10 GB, `cuda_6gb` from 4 GB); None below 4 GB (the defaults)."""
    if preference == "cpu" or not info.has("dense"):
        return "cpu"
    if info.vram_mb is not None and info.vram_mb >= LARGE_VRAM_MB:
        return "cuda_large"
    if info.vram_mb is None or info.vram_mb >= MIN_VRAM_MB:
        return "cuda_6gb"
    return None


def resolve_hardware(
    hardware: str = "auto", config_files: Sequence[Path] = (), overrides: Sequence[str] = ()
) -> str | None:
    """The preset name `hardware` stands for: `auto` picks one with `select_hardware` (honouring `device` from the
    config files and overrides), `none` none, anything else must name a file in `configs/hardware/`."""
    if hardware == "none":
        return None
    if hardware == "auto":
        device = load_config(config_files, overrides).device
        if device.preference == "cpu":
            return select_hardware(DeviceInfo(), "cpu")
        return select_hardware(probe_device(device.gpu_index), device.preference)
    if hardware not in hardware_presets():
        raise ConfigError(f"unknown hardware preset {hardware!r}; known: auto, none, {', '.join(hardware_presets())}")
    return hardware


def load_config(
    config_files: Sequence[Path] = (), overrides: Sequence[str] = (), hardware: str = "none"
) -> PipelineConfig:
    """Merge the default config, the hardware preset (`resolve_hardware`), `config_files` and `overrides` (later
    wins), then validate. Unknown keys are errors."""
    data = _read_yaml(DEFAULT_CONFIG)
    if (preset := resolve_hardware(hardware, config_files, overrides)) is not None:
        data = _merge(data, _read_yaml(HARDWARE_DIR / f"{preset}.yaml"))
    for path in config_files:
        data = _merge(data, _read_yaml(path))
    for text in overrides:
        data = _merge(data, parse_override(text))
    return validate_config(data)


def config_dict(obj: Any) -> Any:
    """Plain JSON/YAML types of a config (or any dataclass): enums, paths and tuples become str and lists."""
    return to_jsonable(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj)


def dump_config(config: PipelineConfig) -> str:
    return yaml.safe_dump(config_dict(config), sort_keys=False)


def config_hash(section: Any, extra: dict[str, Any] | None = None) -> str:
    """sha1 of the canonical JSON (sorted keys, floats via repr) of a config section plus any extra inputs."""
    payload = {"section": config_dict(section), "extra": {k: config_dict(v) for k, v in (extra or {}).items()}}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha1(text.encode()).hexdigest()
