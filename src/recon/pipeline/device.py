"""Device probing and backend choice: the one place that decides whether an operation runs on CUDA or the CPU."""

import functools
import importlib
import logging
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

Backend = Literal["cuda", "cpu"]
Preference = Literal["auto", "cuda", "cpu"]
OPENMVS_DENSIFY = "DensifyPointCloud"


@dataclass(slots=True)
class DeviceOptions:
    preference: Preference = "auto"  # auto: CUDA where available, else CPU with a warning
    gpu_index: int = 0


@dataclass(slots=True)
class DeviceInfo:
    cuda_available: bool = False
    gpu_name: str | None = None
    vram_mb: int | None = None
    compute_capability: str | None = None
    torch_cuda: bool = False
    ffmpeg_nvdec: bool = False
    pycolmap_cuda: bool = False
    openmvs_cuda: bool = False
    onnx_cuda: bool = False

    def has(self, capability: str) -> bool:
        """Whether `capability` (a key of CAPABILITIES) can run on CUDA here."""
        return all(getattr(self, flag) for flag in CAPABILITIES[capability])


# capability -> DeviceInfo flags it needs for its CUDA path
CAPABILITIES = {
    "decode": ("cuda_available", "ffmpeg_nvdec"),
    "metrics": ("torch_cuda",),
    "sfm": ("pycolmap_cuda",),
    "dense": ("openmvs_cuda",),
    "onnx": ("onnx_cuda",),
}


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout


def _probe_torch(info: DeviceInfo, gpu_index: int) -> None:
    import torch

    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        props = torch.cuda.get_device_properties(gpu_index)
        info.torch_cuda = info.cuda_available = True
        info.gpu_name = props.name
        info.vram_mb = props.total_memory // 2**20
        info.compute_capability = f"{props.major}.{props.minor}"


def _probe_nvidia_smi(info: DeviceInfo, gpu_index: int) -> None:
    if info.cuda_available or not shutil.which("nvidia-smi"):
        return
    fields = "name,memory.total,compute_cap"
    rows = _run("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits").strip().splitlines()
    if gpu_index < len(rows):
        name, vram, cc = (v.strip() for v in rows[gpu_index].split(","))
        info.cuda_available, info.gpu_name, info.vram_mb, info.compute_capability = True, name, int(vram), cc


def _probe_ffmpeg(info: DeviceInfo, gpu_index: int) -> None:
    info.ffmpeg_nvdec = bool(shutil.which("ffmpeg")) and "h264_cuvid" in _run("ffmpeg", "-hide_banner", "-decoders")


def _probe_pycolmap(info: DeviceInfo, gpu_index: int) -> None:
    import pycolmap

    info.pycolmap_cuda = bool(getattr(pycolmap, "has_cuda", False))


def _probe_openmvs(info: DeviceInfo, gpu_index: int) -> None:
    info.openmvs_cuda = bool(shutil.which(OPENMVS_DENSIFY)) and "--cuda-device" in _run(OPENMVS_DENSIFY, "--help")


def _probe_onnx(info: DeviceInfo, gpu_index: int) -> None:
    import onnxruntime

    info.onnx_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()


@functools.cache
def probe_device(gpu_index: int = 0) -> DeviceInfo:
    """What this machine can run on CUDA. Every probe is cheap and a missing tool or library just reads as False."""
    info = DeviceInfo()
    for probe in (_probe_torch, _probe_nvidia_smi, _probe_ffmpeg, _probe_pycolmap, _probe_openmvs, _probe_onnx):
        try:
            probe(info, gpu_index)
        except Exception as e:  # noqa: BLE001  (a broken install must not stop the pipeline)
            log.debug("%s failed: %s", probe.__name__, e)
    return info


def resolve_backend(capability: str, preference: Preference, info: DeviceInfo) -> Backend:
    """CUDA or CPU for one capability. `auto` falls back to the CPU with a warning; `cuda` raises if unavailable."""
    if capability not in CAPABILITIES:
        raise KeyError(f"unknown capability {capability!r}; known: {', '.join(CAPABILITIES)}")
    if preference == "cpu":
        return "cpu"
    if info.has(capability):
        return "cuda"
    missing = [flag for flag in CAPABILITIES[capability] if not getattr(info, flag)]
    if preference == "cuda":
        raise RuntimeError(f"{capability} on CUDA was requested but is unavailable (missing: {', '.join(missing)})")
    log.warning("%s: no CUDA path available (missing: %s), using the CPU", capability, ", ".join(missing))
    return "cpu"


# import name -> label of the Python packages the pipeline uses
PYTHON_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "cv2": "opencv",
    "av": "PyAV",
    "torch": "torch",
    "pycolmap": "pycolmap",
    "onnxruntime": "onnxruntime",
    "pydantic": "pydantic",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
}
TOOLS = {"ffmpeg": ("ffmpeg", "-version"), "OpenMVS": (OPENMVS_DENSIFY, "--help")}


def tool_versions() -> dict[str, str | None]:
    """Version of Python, each Python package and each external tool; None when it is missing."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for module, label in PYTHON_PACKAGES.items():
        try:
            versions[label] = str(getattr(importlib.import_module(module), "__version__", "?"))
        except Exception:  # noqa: BLE001
            versions[label] = None
    for label, cmd in TOOLS.items():
        if not shutil.which(cmd[0]):
            versions[label] = None
            continue
        try:
            match = re.search(r"(?:version|v)\s?(\d+\.\d+(?:\.\d+)?)", _run(*cmd))
        except Exception:  # noqa: BLE001
            match = None
        versions[label] = match.group(1) if match else "?"
    return versions
