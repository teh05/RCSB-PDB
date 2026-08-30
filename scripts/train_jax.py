"""Train the Flax EGNN on the debug protein graphs (JAX GPU process).

Do not import torch in this process (Windows cuDNN clash with winjax).

Task: edge-wise C-alpha distance regression. Three proteins; no split;
not a generalization result.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import serialization

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.build import load_ca_graphs
from src.graph.collate import collate_graphs
from src.models.jax.egnn import EGNN
from src.paths import CHECKPOINTS_DIR, CONFIGS_DIR, GRAPHS_DIR, METRICS_DIR, ensure_dirs
from src.training.jax_device import jax_memory_stats, require_jax_gpu


def _to_jax(data: dict):
    return {
        "x": jnp.asarray(data["x"], dtype=jnp.float32),
        "coords": jnp.asarray(data["coords"], dtype=jnp.float32),
        "edge_index": jnp.asarray(data["edge_index"], dtype=jnp.int32),
        "edge_dist": jnp.asarray(data["edge_dist"], dtype=jnp.float32),
    }


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "egnn.yaml"
    cfg = load_config(cfg_path)

    if str(cfg.get("device", "cuda")).lower() != "cuda":
        raise RuntimeError("Training requires device: cuda. No CPU fallback.")

    name = str(cfg.get("model", "egnn")).lower()
    if name != "egnn":
        raise ValueError("scripts/train_jax.py implements EGNN only. Use configs/egnn.yaml.")

    device = require_jax_gpu()
    seed = int(cfg["seed"])
    cutoff = float(cfg["radius_cutoff"])
    graphs = load_ca_graphs(GRAPHS_DIR, cutoff)
    raw = _to_jax(collate_graphs(graphs))
    batch = {k: jax.device_put(v, device) for k, v in raw.items()}

    model = EGNN(hidden_dim=int(cfg["hidden_dim"]), num_layers=int(cfg["num_layers"]))
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, batch["x"], batch["coords"], batch["edge_index"])
    tx = optax.adam(float(cfg["learning_rate"]))
    opt_state = tx.init(params)

    def loss_fn(p):
        pred = model.apply(p, batch["x"], batch["coords"], batch["edge_index"])
        return jnp.mean((pred - batch["edge_dist"]) ** 2)

    @jax.jit
    def train_step(p, ostate):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, ostate = tx.update(grads, ostate, p)
        p = optax.apply_updates(p, updates)
        return p, ostate, loss

    print(f"JAX device: {device}")
    print(f"backend: {jax.default_backend()}")
    print("model: EGNN")
    print(f"graphs: {[g.pdb_id for g in graphs]}")
    print(f"nodes: {int(batch['x'].shape[0])} edges: {int(batch['edge_index'].shape[1])}")
    print("task: edge_ca_distance_regression (diagnostic, not biological labels)")
    print("split: none - debug set only, no generalization claim")
    print()

    init_loss = float(loss_fn(params).block_until_ready())
    print(f"forward ok  initial_loss={init_loss:.4f}")

    print("compiling train_step...")
    t_compile = time.perf_counter()
    _ = train_step(params, opt_state)[2].block_until_ready()
    compile_s = time.perf_counter() - t_compile
    print(f"compile_s={compile_s:.3f}")
    params = model.init(rng, batch["x"], batch["coords"], batch["edge_index"])
    opt_state = tx.init(params)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        step_t0 = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state)
        loss_f = float(loss.block_until_ready())
        step_ms = (time.perf_counter() - step_t0) * 1000
        mem = jax_memory_stats()
        row = {
            "epoch": epoch,
            "loss": loss_f,
            "step_ms": step_ms,
            "peak_allocated_mb": mem.get("peak_allocated_mb"),
        }
        history.append(row)
        peak = mem.get("peak_allocated_mb")
        peak_s = f"{peak:.1f}MB" if isinstance(peak, (int, float)) else "n/a"
        print(f"epoch {epoch:03d}  loss={loss_f:.4f}  step={step_ms:.1f}ms  peak_alloc={peak_s}")

    elapsed = time.perf_counter() - t0
    mem = jax_memory_stats()
    mean_ex_first = None
    if len(history) > 1:
        mean_ex_first = sum(r["step_ms"] for r in history[1:]) / (len(history) - 1)

    ckpt = CHECKPOINTS_DIR / "egnn_jax_debug.msgpack"
    ckpt.write_bytes(serialization.to_bytes(params))

    metrics = {
        "experiment_id": "phase8_egnn_jax",
        "model": "EGNN",
        "framework": "jax",
        "representation": cfg["representation"],
        "dataset": "raw_debug_mmcif",
        "n_proteins": len(graphs),
        "pdb_ids": [g.pdb_id for g in graphs],
        "GPU": mem["gpu_name"],
        "task": "edge_ca_distance_regression",
        "note": "engineering debug on 3 proteins; not scientific generalization",
        "final_loss": history[-1]["loss"],
        "training_time_s": elapsed,
        "compile_s": compile_s,
        "first_step_ms": history[0]["step_ms"],
        "mean_step_ms_ex_first": mean_ex_first,
        "memory": mem,
        "epochs": cfg["epochs"],
        "radius_cutoff": cutoff,
    }
    json_path = METRICS_DIR / "phase8_egnn_jax.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_path = METRICS_DIR / "phase8_egnn_jax_epochs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "loss", "step_ms", "peak_allocated_mb"])
        writer.writeheader()
        writer.writerows(history)

    print()
    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
