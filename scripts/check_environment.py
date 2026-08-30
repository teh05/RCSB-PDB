"""Fail-fast environment validation for the RCSB-PDB GNN pipeline.

On Windows, PyTorch (CUDA 12.8) and winjax (CUDA 13 / cuDNN 9) must not
share one process. This script probes each stack in a child process and
refuses to continue if either framework lacks a CUDA GPU.

No CPU fallback is performed for GPU-required checks.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


PROBE_TORCH = r"""
import json, sys
out = {"ok": False, "error": None}
try:
    import torch
    out["version"] = torch.__version__
    out["cuda_version"] = str(torch.version.cuda)
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    out["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if not torch.cuda.is_available():
        out["error"] = (
            "torch.cuda.is_available() is False. "
            "PyTorch cannot see an NVIDIA GPU. Check NVIDIA driver, "
            "that this is a CUDA wheel (not cpu), and CUDA capability."
        )
        print(json.dumps(out))
        sys.exit(2)
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    out["matmul_ok"] = bool(torch.isfinite(y).all().item())
    out["capability"] = list(torch.cuda.get_device_capability(0))
    out["ok"] = True
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    sys.exit(1)
print(json.dumps(out))
"""


PROBE_JAX = r"""
import json, sys
out = {"ok": False, "error": None, "winjax": None, "flax_version": None}
try:
    import jax
    import jax.numpy as jnp
    out["version"] = jax.__version__
    devices = jax.devices()
    out["devices"] = [str(d) for d in devices]
    out["backend"] = str(jax.default_backend())
    try:
        from importlib.metadata import version as _pkg_ver
        out["winjax"] = _pkg_ver("winjax")
    except Exception:
        try:
            import winjax as _wj  # noqa: F401
            out["winjax"] = getattr(_wj, "__version__", "installed")
        except Exception:
            out["winjax"] = None
    gpu_like = [
        d for d in devices
        if d.platform.lower() in {"gpu", "cuda"}
        or "gpu" in type(d).__name__.lower()
        or "cuda" in type(d).__name__.lower()
    ]
    if not gpu_like:
        out["error"] = (
            "JAX has no GPU device (got: {devs}). "
            "Official jax[cuda13] wheels are Linux-only. On native Windows "
            "install winjax (PJRT CUDA 13 plugin). Do not use CPU jax for "
            "GPU benchmarks. If winjax is installed but devices are CPU, "
            "the NVIDIA driver likely cannot load the CUDA 13 plugin."
        ).format(devs=out["devices"])
        print(json.dumps(out))
        sys.exit(2)
    x = jnp.ones((256, 256), dtype=jnp.float32)
    y = (x @ x).block_until_ready()
    out["matmul_ok"] = bool(jnp.isfinite(y).all())
    import flax
    from flax import linen, nnx  # noqa: F401
    out["flax_version"] = flax.__version__
    out["ok"] = True
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    sys.exit(1)
print(json.dumps(out))
"""


def _run_probe(code: str, label: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    payload: dict[str, Any]
    try:
        payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except json.JSONDecodeError:
        payload = {}
    if proc.returncode != 0 or not payload.get("ok"):
        err = payload.get("error") or (proc.stderr or stdout or f"{label} probe failed")
        return {"ok": False, "error": err, "raw_stderr": (proc.stderr or "")[-2000:], **payload}
    return payload


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    proc = subprocess.run(
        [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def main() -> int:
    print("=" * 72)
    print("RCSB-PDB environment validation")
    print("=" * 72)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print()

    smi = _nvidia_smi()
    if smi:
        print(f"nvidia-smi GPU: {smi}")
    else:
        print("nvidia-smi: NOT FOUND")
    print()

    numpy_ver = _pkg_version("numpy")
    gemmi_ver = _pkg_version("gemmi")
    flax_ver = _pkg_version("flax")
    print(f"NumPy version: {numpy_ver or 'MISSING'}")
    print(f"Gemmi version: {gemmi_ver or 'MISSING'}")
    print(f"Flax version: {flax_ver or 'MISSING'}")
    print()

    errors: list[str] = []

    print("--- PyTorch (isolated process) ---")
    torch_info = _run_probe(PROBE_TORCH, "PyTorch")
    if torch_info.get("ok"):
        print(f"PyTorch version: {torch_info['version']}")
        print(f"PyTorch CUDA version: {torch_info['cuda_version']}")
        print(f"CUDA available: {torch_info['cuda_available']}")
        print(f"GPU: {torch_info['gpu_name']}")
        print(f"GPU count: {torch_info['gpu_count']}")
        print(f"Compute capability: {tuple(torch_info['capability'])}")
        print(f"CUDA matmul: {'ok' if torch_info.get('matmul_ok') else 'FAILED'}")
    else:
        print("CUDA available: False")
        print(f"PyTorch GPU probe FAILED: {torch_info.get('error')}")
        if torch_info.get("raw_stderr"):
            print(torch_info["raw_stderr"][-800:])
        errors.append("PyTorch CUDA is required and was not detected.")
    print()

    print("--- JAX (isolated process) ---")
    jax_info = _run_probe(PROBE_JAX, "JAX")
    if jax_info.get("ok"):
        print(f"JAX version: {jax_info['version']}")
        print(f"winjax: {jax_info.get('winjax')}")
        print(f"Flax version (importable): {jax_info.get('flax_version')}")
        print("JAX devices:")
        for d in jax_info["devices"]:
            print(f"  {d}")
        print(f"JAX backend: {jax_info['backend']}")
        print(f"JAX GPU matmul: {'ok' if jax_info.get('matmul_ok') else 'FAILED'}")
    else:
        print("JAX devices: CPU-only or unavailable")
        print(f"JAX GPU probe FAILED: {jax_info.get('error')}")
        if jax_info.get("raw_stderr"):
            print(jax_info["raw_stderr"][-800:])
        errors.append("JAX GPU is required and was not detected.")
    print()

    if flax_ver is None:
        errors.append("Flax is not installed.")
    if gemmi_ver is None:
        errors.append("Gemmi is not installed.")
    if numpy_ver is None:
        errors.append("NumPy is not installed.")

    if errors:
        print("=" * 72)
        print("ENVIRONMENT CHECK FAILED — GPU training/benchmark must not continue.")
        print("No CPU fallback will be used.")
        for e in errors:
            print(f"  - {e}")
        print()
        print("Expected:")
        print("  CUDA available: True")
        print("  GPU: NVIDIA GeForce RTX 5060 Laptop GPU")
        print("  JAX devices: GpuDevice / CudaDevice")
        print()
        print("Windows notes:")
        print("  - Official jax[cuda13] is Linux-only.")
        print("  - Native Windows JAX GPU uses winjax (CUDA 13 PJRT).")
        print("  - Do not import torch and jax in the same process (cuDNN clash).")
        print("  - Use scripts/train_pytorch.py and scripts/train_jax.py separately.")
        print("=" * 72)
        return 1

    print("=" * 72)
    print("ENVIRONMENT CHECK PASSED")
    print("PyTorch CUDA and JAX GPU are both visible.")
    print("Safe to proceed to STEP 2 (mmCIF parse of 1UBQ).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
