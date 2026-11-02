"""External binaries: discovery, runs logged to a log dir with timing and peak GPU memory, errors with the log tail."""

import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LOG_TAIL_LINES = 40
VRAM_INTERVAL_S = 0.25
BUILD_HINT = "OpenMVS: build it with `pixi run -e build openmvs`"


class ToolError(RuntimeError):
    def __init__(self, tool: str, returncode: int | None, log_tail: str, message: str | None = None) -> None:
        self.tool, self.returncode, self.log_tail = tool, returncode, log_tail
        super().__init__(message or f"{tool} exited with status {returncode}; log tail:\n{log_tail}")


class ToolTimeout(ToolError):
    pass


@dataclass(slots=True)
class ToolResult:
    tool: str
    argv: list[str]
    returncode: int
    duration_s: float
    peak_vram_mb: int | None  # GPU-wide memory used above the value just before the call; None without nvidia-smi
    log_path: Path


def _vram_used_mb(gpu_index: int) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class _VramSampler:
    """Samples the GPU's used memory every VRAM_INTERVAL_S in a thread; `peak_mb` is the rise over the baseline."""

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.baseline = _vram_used_mb(gpu_index)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        if self.baseline is not None:
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(VRAM_INTERVAL_S):
            if (used := _vram_used_mb(self.gpu_index)) is not None:
                self.peak = max(self.peak, used)

    def stop(self) -> int | None:
        if self.baseline is None:
            return None
        self._stop.set()
        self._thread.join()
        return max(self.peak - self.baseline, 0)


def _tail(path: Path, start: int = 0, lines: int = LOG_TAIL_LINES) -> str:
    """The last `lines` lines of `path` after byte `start`."""
    with path.open("rb") as f:
        f.seek(start)
        text = f.read().decode(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


class ToolRunner:
    """Runs external tools with stdout + stderr appended to `log_dir/<tool>.log`, and keeps a result per call."""

    def __init__(self, log_dir: Path, bin_dir: Path | str | None = None, gpu_index: int = 0, sample_vram: bool = True):
        self.log_dir = Path(log_dir)
        self.bin_dir = Path(bin_dir) if bin_dir is not None else None
        self.gpu_index = gpu_index
        self.sample_vram = sample_vram and shutil.which("nvidia-smi") is not None
        self.results: list[ToolResult] = []
        self._versions: dict[str, str | None] = {}

    def which(self, tool: str) -> Path:
        """`bin_dir/<tool>` when a bin dir is set, else the tool on PATH."""
        found = shutil.which(tool, path=str(self.bin_dir)) if self.bin_dir is not None else shutil.which(tool)
        if found is None:
            where = self.bin_dir if self.bin_dir is not None else "PATH"
            raise ToolError(tool, None, "", f"{tool} not found in {where} ({BUILD_HINT})")
        return Path(found)

    def version(self, tool: str) -> str | None:
        """The version printed by `<tool> --help` (e.g. "2.4.0"), None when it prints none."""
        if tool not in self._versions:
            with tempfile.TemporaryDirectory() as tmp:  # OpenMVS drops a log file into the current directory
                out = subprocess.run(
                    [self.which(tool), "--help"], capture_output=True, text=True, timeout=30, cwd=tmp
                ).stdout
            match = re.search(r"(?:version|v)\s?(\d+\.\d+(?:\.\d+)?)", out)
            self._versions[tool] = match.group(1) if match else None
        return self._versions[tool]

    def _collect_tool_logs(self, tool: str, cwd: Path, before: set[Path]) -> None:
        """Move the `<tool>-<stamp>.log` files the tool wrote into `cwd` to the log dir."""
        for path in sorted(set(cwd.glob(f"{tool}-*.log")) - before):
            shutil.move(path, self.log_dir / path.name)

    def run(self, tool: str, args: list[str], cwd: Path, timeout_s: float | None = None) -> ToolResult:
        """Run `tool args` in `cwd`. A non-zero exit raises ToolError (with the tail of this call's output); after
        `timeout_s` the tool is killed and ToolTimeout raised."""
        argv = [str(self.which(tool)), *map(str, args)]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{tool}.log"
        before = set(cwd.glob(f"{tool}-*.log"))
        with log_path.open("a") as log:
            log.write(f"\n$ cd {cwd} && {shlex.join(argv)}\n")
            log.flush()
            start = log.tell()
            sampler = _VramSampler(self.gpu_index) if self.sample_vram else None
            t0 = time.perf_counter()
            process = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            timed_out = False
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                duration = time.perf_counter() - t0
                peak = sampler.stop() if sampler is not None else None
                self._collect_tool_logs(tool, cwd, before)
        result = ToolResult(tool, argv, process.returncode, round(duration, 3), peak, log_path)
        self.results.append(result)
        if timed_out:
            message = f"{tool} killed after {timeout_s:.0f} s"
            raise ToolTimeout(tool, process.returncode, _tail(log_path, start), message)
        if process.returncode != 0:
            raise ToolError(tool, process.returncode, _tail(log_path, start))
        return result
