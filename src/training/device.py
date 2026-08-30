"""CUDA device helpers. GPU benchmarks must not silently fall back to CPU."""

from __future__ import annotations

import torch


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this script. torch.cuda.is_available() is False. "
            "No CPU fallback is used for GPU training/benchmarks. "
            "Run python scripts/check_environment.py and fix the PyTorch CUDA stack."
        )
    return torch.device("cuda")


def gpu_memory_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to read GPU memory stats.")
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "gpu_name": torch.cuda.get_device_name(0),
    }
