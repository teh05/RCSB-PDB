"""SE(3) equivariance check for the Flax EGNN.

Must run in a process that does not import torch.
Exit 0 on pass, 2 if JAX GPU is missing, 1 on failed residuals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.se3 import random_rotation
from src.models.jax.egnn import EGNN
from src.paths import METRICS_DIR, ensure_dirs
from src.training.jax_device import require_jax_gpu

ATOL = 1e-3


def _chain_edges(n: int) -> np.ndarray:
    src = np.arange(n - 1, dtype=np.int32)
    dst = np.arange(1, n, dtype=np.int32)
    return np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)


def main() -> int:
    ensure_dirs()
    try:
        device = require_jax_gpu()
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2

    rng_np = np.random.default_rng(0)
    n, in_dim = 12, 21
    x = np.eye(in_dim, dtype=np.float32)[rng_np.integers(0, in_dim, size=n)]
    coords = rng_np.normal(size=(n, 3)).astype(np.float32)
    edge_index = _chain_edges(n)
    R = random_rotation(rng_np).astype(np.float32)
    t = rng_np.normal(size=(3,)).astype(np.float32)

    x_j = jax.device_put(jnp.asarray(x), device)
    coords_j = jax.device_put(jnp.asarray(coords), device)
    ei = jax.device_put(jnp.asarray(edge_index), device)
    R_j = jax.device_put(jnp.asarray(R), device)
    t_j = jax.device_put(jnp.asarray(t), device)

    model = EGNN(hidden_dim=32, num_layers=2)
    params = model.init(jax.random.PRNGKey(0), x_j, coords_j, ei)

    pred0, c0 = model.apply(params, x_j, coords_j, ei, return_coords=True)
    pred_r, c_r = model.apply(params, x_j, coords_j @ R_j.T, ei, return_coords=True)
    pred_t, c_t = model.apply(params, x_j, coords_j + t_j, ei, return_coords=True)

    inv_R = float(jnp.max(jnp.abs(pred0 - pred_r)))
    inv_t = float(jnp.max(jnp.abs(pred0 - pred_t)))
    eq_R = float(jnp.max(jnp.abs(c0 @ R_j.T - c_r)))
    eq_t = float(jnp.max(jnp.abs((c0 + t_j) - c_t)))

    report = {
        "ok": inv_R < ATOL and inv_t < ATOL and eq_R < ATOL and eq_t < ATOL,
        "framework": "jax",
        "model": "EGNN",
        "device": str(device),
        "atol": ATOL,
        "max_abs_pred_rotation": inv_R,
        "max_abs_pred_translation": inv_t,
        "max_abs_coord_rotation": eq_R,
        "max_abs_coord_translation": eq_t,
        "note": "predicted distances should be SE(3)-invariant; coords SE(3)-equivariant",
    }
    out = METRICS_DIR / "phase8_e3_jax.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
