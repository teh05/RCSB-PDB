"""JAX GPU helpers. Do not import torch in this module."""

from __future__ import annotations

from typing import Any

import jax


def require_jax_gpu():
    devices = jax.devices()
    gpu = [
        d
        for d in devices
        if d.platform.lower() in {"gpu", "cuda"}
        or "gpu" in type(d).__name__.lower()
        or "cuda" in type(d).__name__.lower()
    ]
    if not gpu:
        raise RuntimeError(
            "JAX GPU is required. No CPU fallback. "
            f"Devices: {[str(d) for d in devices]}. "
            "On native Windows install winjax. "
            "Run python scripts/check_environment.py."
        )
    backend = str(jax.default_backend()).lower()
    if backend not in {"gpu", "cuda"}:
        raise RuntimeError(
            "JAX GPU is required. No CPU fallback. "
            f"default_backend={backend!r}, devices={[str(d) for d in devices]}."
        )
    return gpu[0]


def jax_memory_stats() -> dict[str, Any]:
    dev = require_jax_gpu()
    out: dict[str, Any] = {
        "gpu_name": str(dev),
        "backend": str(jax.default_backend()),
        "allocated_mb": None,
        "peak_allocated_mb": None,
    }
    try:
        stats = dev.memory_stats()
    except Exception:
        stats = None
    if isinstance(stats, dict):
        if "bytes_in_use" in stats:
            out["allocated_mb"] = float(stats["bytes_in_use"]) / (1024**2)
        if "peak_bytes_in_use" in stats:
            out["peak_allocated_mb"] = float(stats["peak_bytes_in_use"]) / (1024**2)
    return out
