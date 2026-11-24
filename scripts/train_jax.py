"""Train the Flax EGNN (JAX GPU process). Do not import torch.

Mini-batches are padded to a fixed shape so train_step can be JIT-compiled once.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.collate import collate_graphs, pad_collated
from src.models.jax.egnn import EGNN
from src.paths import CHECKPOINTS_DIR, CONFIGS_DIR, METRICS_DIR, ensure_dirs
from src.training.dataset import load_filtered_graphs, split_graph_lists
from src.training.jax_device import jax_memory_stats, require_jax_gpu
from src.training.split import batched_slices


def _finite_or_none(value: float) -> float | None:
    if value != value:
        return None
    return float(value)


def _to_jax(data: dict, device):
    keys = ("x", "coords", "edge_index", "edge_dist", "edge_mask", "node_mask")
    out = {}
    for k in keys:
        dtype = jnp.int32 if k == "edge_index" else jnp.float32
        out[k] = jax.device_put(jnp.asarray(data[k], dtype=dtype), device)
    return out


def _pad_sizes(graphs, batch_size: int) -> tuple[int, int]:
    max_n = max(g.n_nodes for g in graphs)
    max_e = max(g.n_edges for g in graphs)
    return int(batch_size * max_n + 1), int(batch_size * max_e + 8)


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
    graphs = load_filtered_graphs(cfg)
    train_graphs, val_graphs, test_graphs = split_graph_lists(graphs, cfg)
    batch_size = int(cfg.get("batch_size", 8))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    test_fraction = float(cfg.get("test_fraction", 0.0))
    pad_nodes, pad_edges = _pad_sizes(graphs, batch_size)

    def make_batch(chunk):
        return _to_jax(pad_collated(collate_graphs(chunk), pad_nodes, pad_edges), device)

    probe = make_batch(train_graphs[: min(batch_size, len(train_graphs))])
    model = EGNN(hidden_dim=int(cfg["hidden_dim"]), num_layers=int(cfg["num_layers"]))
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, probe["x"], probe["coords"], probe["edge_index"])
    tx = optax.adam(float(cfg["learning_rate"]))
    opt_state = tx.init(params)

    def masked_mse(p, batch):
        pred = model.apply(p, batch["x"], batch["coords"], batch["edge_index"])
        se = (pred - batch["edge_dist"]) ** 2
        denom = jnp.maximum(jnp.sum(batch["edge_mask"]), 1.0)
        return jnp.sum(se * batch["edge_mask"]) / denom

    @jax.jit
    def train_step(p, ostate, batch):
        loss, grads = jax.value_and_grad(masked_mse)(p, batch)
        updates, ostate = tx.update(grads, ostate, p)
        p = optax.apply_updates(p, updates)
        return p, ostate, loss

    @jax.jit
    def eval_loss(p, batch):
        return masked_mse(p, batch)

    print(f"JAX device: {device}")
    print(f"backend: {jax.default_backend()}")
    print("model: EGNN")
    print(
        f"n_proteins: {len(graphs)}  train: {len(train_graphs)}  "
        f"val: {len(val_graphs)}  test: {len(test_graphs)}"
    )
    print(f"batch_size: {batch_size}  pad_nodes: {pad_nodes}  pad_edges: {pad_edges}")
    if val_fraction <= 0 and test_fraction <= 0:
        print("split: train-all")
    print("task: edge_ca_distance_regression (diagnostic, not biological labels)")
    print()

    init_loss = float(eval_loss(params, probe).block_until_ready())
    print(f"forward ok  probe_loss={init_loss:.4f}")

    print("compiling train_step...")
    t_compile = time.perf_counter()
    _ = train_step(params, opt_state, probe)[2].block_until_ready()
    compile_s = time.perf_counter() - t_compile
    print(f"compile_s={compile_s:.3f}")
    params = model.init(rng, probe["x"], probe["coords"], probe["edge_index"])
    opt_state = tx.init(params)

    rng_np = np.random.default_rng(seed)
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        step_t0 = time.perf_counter()
        order = rng_np.permutation(len(train_graphs))
        last_loss = 0.0
        n_steps = 0
        for a, b in batched_slices(len(train_graphs), batch_size):
            chunk = [train_graphs[int(i)] for i in order[a:b]]
            batch = make_batch(chunk)
            params, opt_state, loss = train_step(params, opt_state, batch)
            last_loss = float(loss.block_until_ready())
            n_steps += 1
        val_loss = float("nan")
        if val_graphs:
            totals = []
            for a, b in batched_slices(len(val_graphs), batch_size):
                totals.append(float(eval_loss(params, make_batch(val_graphs[a:b])).block_until_ready()))
            val_loss = float(np.mean(totals))
        epoch_ms = (time.perf_counter() - step_t0) * 1000
        mem = jax_memory_stats()
        row = {
            "epoch": epoch,
            "loss": last_loss,
            "val_loss": val_loss,
            "epoch_ms": epoch_ms,
            "steps": n_steps,
            "peak_allocated_mb": mem.get("peak_allocated_mb"),
        }
        history.append(row)
        peak = mem.get("peak_allocated_mb")
        peak_s = f"{peak:.1f}MB" if isinstance(peak, (int, float)) else "n/a"
        val_s = f"{val_loss:.4f}" if val_loss == val_loss else "n/a"
        print(
            f"epoch {epoch:03d}  train_last={last_loss:.4f}  val={val_s}  "
            f"epoch={epoch_ms:.0f}ms  peak_alloc={peak_s}"
        )

    def mean_eval(p, graphs_):
        if not graphs_:
            return float("nan")
        totals = []
        for a, b in batched_slices(len(graphs_), batch_size):
            totals.append(float(eval_loss(p, make_batch(graphs_[a:b])).block_until_ready()))
        return float(np.mean(totals))

    elapsed = time.perf_counter() - t0
    train_mse = mean_eval(params, train_graphs)
    test_mse = mean_eval(params, test_graphs)
    mem = jax_memory_stats()
    large = len(graphs) >= 50
    ckpt = CHECKPOINTS_DIR / ("egnn_jax.msgpack" if large else "egnn_jax_debug.msgpack")
    ckpt.write_bytes(serialization.to_bytes(params))
    json_path = METRICS_DIR / ("phase10_egnn_jax.json" if large else "phase8_egnn_jax.json")
    csv_path = METRICS_DIR / ("phase10_egnn_jax_epochs.csv" if large else "phase8_egnn_jax_epochs.csv")

    metrics = {
        "experiment_id": "phase10_egnn_jax" if large else "phase8_egnn_jax",
        "model": "EGNN",
        "framework": "jax",
        "representation": cfg["representation"],
        "dataset": "rcsb_xray_protein" if large else "raw_debug_mmcif",
        "n_proteins": len(graphs),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_test": len(test_graphs),
        "pdb_ids": [g.pdb_id for g in graphs],
        "batch_size": batch_size,
        "pad_nodes": pad_nodes,
        "pad_edges": pad_edges,
        "GPU": mem["gpu_name"],
        "task": "edge_ca_distance_regression",
        "note": "diagnostic edge-distance task; not biochemical function prediction",
        "final_loss": history[-1]["loss"],
        "final_train_mse": _finite_or_none(train_mse),
        "final_val_loss": _finite_or_none(float(history[-1]["val_loss"])),
        "final_test_mse": _finite_or_none(test_mse),
        "training_time_s": elapsed,
        "compile_s": compile_s,
        "memory": mem,
        "epochs": cfg["epochs"],
        "radius_cutoff": cutoff,
    }
    json_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["epoch", "loss", "val_loss", "epoch_ms", "steps", "peak_allocated_mb"]
        )
        writer.writeheader()
        for row in history:
            csv_row = dict(row)
            if csv_row["val_loss"] != csv_row["val_loss"]:
                csv_row["val_loss"] = ""
            writer.writerow(csv_row)

    printable = {k: v for k, v in metrics.items() if k != "pdb_ids"}
    print()
    print(json.dumps(printable, indent=2, allow_nan=False))
    print(f"pdb_ids: {len(metrics['pdb_ids'])}")
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
