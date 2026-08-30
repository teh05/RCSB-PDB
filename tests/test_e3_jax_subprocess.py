from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_e3_jax.py"


def test_jax_egnn_equivariance_subprocess():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 2:
        pytest.skip(proc.stdout.strip() or proc.stderr.strip() or "JAX GPU unavailable")
    if proc.returncode != 0:
        pytest.fail(
            f"check_e3_jax.py failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    metrics = ROOT / "results" / "metrics" / "phase8_e3_jax.json"
    assert metrics.exists(), proc.stdout
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload.get("ok") is True
