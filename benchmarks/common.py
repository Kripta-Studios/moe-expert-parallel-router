"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))


def cuda_time_ms(fn: Callable[[], object], warmup: int = 3, iters: int = 10) -> float:
    """Median of `iters` CUDA-event timings (after warm-up, which also triggers
    Triton autotuning and cuBLAS heuristics)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def env_info() -> dict[str, str]:
    info = {"python": platform.python_version(), "torch": torch.__version__, "os": platform.platform()}
    info["cpu"] = platform.processor()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = str(torch.version.cuda)
        info["gpu_state_at_start"] = gpu_snapshot()
        try:
            import triton

            info["triton"] = triton.__version__
        except ImportError:
            pass
    return info


def gpu_snapshot() -> str:
    """The GPU is shared with other jobs; record how busy it was so readers can
    judge the timings (time-slicing between CUDA contexts inflates them)."""
    try:
        q = "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw"
        return subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def save_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps({"env": env_info(), **payload}, indent=2))
    return path
