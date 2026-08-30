"""Train a PyTorch GNN on protein graphs.

Task: edge-wise C-alpha distance regression (diagnostic).
Mini-batch + train/val split when the set is large enough.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.build import load_ca_graphs
from src.models.pytorch.baseline import BaselineGCN
from src.models.pytorch.egnn import EGNN
from src.paths import CHECKPOINTS_DIR, CONFIGS_DIR, GRAPHS_DIR, METRICS_DIR, ensure_dirs
from src.training.batching import BatchedGraph, graphs_to_batch
from src.training.device import gpu_memory_stats, require_cuda
from src.training.split import batched_slices, split_indices


def _model_name(cfg: dict) -> str:
    return str(cfg.get("model", "gcn")).lower()


def build_model(cfg: dict, in_dim: int, n_proteins: int) -> tuple[nn.Module, str, str]:
    hidden = int(cfg["hidden_dim"])
    layers = int(cfg["num_layers"])
    name = _model_name(cfg)
    large = n_proteins >= 50
    if name in {"gcn", "baselinegcn", "baseline"}:
        exp = "phase9_gcn_pytorch" if large else "phase7_baseline_debug"
        return BaselineGCN(in_dim, hidden, layers), "BaselineGCN", exp
    if name in {"egnn"}:
        exp = "phase9_egnn_pytorch" if large else "phase8_egnn_pytorch"
        return EGNN(in_dim, hidden, layers), "EGNN", exp
    raise ValueError(f"Unknown model {name!r}. Use gcn or egnn.")


def predict(model: nn.Module, batch: BatchedGraph) -> torch.Tensor:
    if getattr(model, "uses_coords", False):
        return model(batch.x, batch.coords, batch.edge_index)
    return model(batch.x, batch.edge_index)


@torch.no_grad()
def _eval_loss(model: nn.Module, graphs: list, device: torch.device, batch_size: int) -> float:
    if not graphs:
        return float("nan")
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    n_edges = 0
    for a, b in batched_slices(len(graphs), batch_size):
        batch = graphs_to_batch(graphs[a:b], device)
        pred = predict(model, batch)
        total += float(loss_fn(pred, batch.edge_dist).detach())
        n_edges += int(batch.edge_dist.numel())
    model.train()
    return total / max(n_edges, 1)


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "baseline.yaml"
    cfg = load_config(cfg_path)

    if str(cfg.get("device", "cuda")).lower() != "cuda":
        raise RuntimeError("Training requires device: cuda. No CPU fallback.")

    device = require_cuda()
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    cutoff = float(cfg["radius_cutoff"])
    graphs = load_ca_graphs(GRAPHS_DIR, cutoff)
    min_ca = int(cfg.get("min_polymer_residues", 50))
    max_ca = int(cfg.get("max_polymer_residues", 350))
    graphs = [g for g in graphs if min_ca <= g.n_nodes <= max_ca]
    if not graphs:
        raise RuntimeError("No graphs in residue-count filter. Run preprocess/build_graph.")
    batch_size = int(cfg.get("batch_size", 8))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    train_idx, val_idx = split_indices(len(graphs), val_fraction=val_fraction, seed=seed)
    train_graphs = [graphs[int(i)] for i in train_idx]
    val_graphs = [graphs[int(i)] for i in val_idx]

    in_dim = int(train_graphs[0].node_features.shape[-1])
    model, model_label, experiment_id = build_model(cfg, in_dim, len(graphs))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.MSELoss()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device: {device}")
    print(f"model: {model_label}")
    print(f"n_proteins: {len(graphs)}  train: {len(train_graphs)}  val: {len(val_graphs)}")
    print(f"batch_size: {batch_size}")
    print("task: edge_ca_distance_regression (diagnostic, not biological labels)")
    print()

    model.train()
    probe = graphs_to_batch(train_graphs[: min(batch_size, len(train_graphs))], device)
    pred = predict(model, probe)
    loss = loss_fn(pred, probe.edge_dist)
    loss.backward()
    opt.zero_grad(set_to_none=True)
    print(f"forward/backward ok  probe_loss={float(loss.detach()):.4f}")

    history = []
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        order = rng.permutation(len(train_graphs))
        last_loss = 0.0
        n_steps = 0
        for a, b in batched_slices(len(train_graphs), batch_size):
            chunk = [train_graphs[int(i)] for i in order[a:b]]
            batch = graphs_to_batch(chunk, device)
            opt.zero_grad(set_to_none=True)
            pred = predict(model, batch)
            loss = loss_fn(pred, batch.edge_dist)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            n_steps += 1
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - step_t0) * 1000
        val_loss = _eval_loss(model, val_graphs, device, batch_size)
        mem = gpu_memory_stats()
        row = {
            "epoch": epoch,
            "loss": last_loss,
            "val_loss": val_loss,
            "epoch_ms": step_ms,
            "steps": n_steps,
            "peak_allocated_mb": mem["peak_allocated_mb"],
        }
        history.append(row)
        val_s = f"{val_loss:.4f}" if val_loss == val_loss else "n/a"
        print(
            f"epoch {epoch:03d}  train_last={row['loss']:.4f}  val={val_s}  "
            f"epoch={step_ms:.0f}ms  peak_alloc={mem['peak_allocated_mb']:.1f}MB"
        )

    elapsed = time.perf_counter() - t0
    mem = gpu_memory_stats()
    large = len(graphs) >= 50
    if model_label == "BaselineGCN":
        ckpt = CHECKPOINTS_DIR / ("gcn_pytorch.pt" if large else "baseline_gcn_debug.pt")
        json_path = METRICS_DIR / ("phase9_gcn_pytorch.json" if large else "phase7_baseline.json")
        csv_path = METRICS_DIR / ("phase9_gcn_pytorch_epochs.csv" if large else "phase7_baseline_epochs.csv")
    else:
        ckpt = CHECKPOINTS_DIR / ("egnn_pytorch.pt" if large else "egnn_pytorch_debug.pt")
        json_path = METRICS_DIR / ("phase9_egnn_pytorch.json" if large else "phase8_egnn_pytorch.json")
        csv_path = METRICS_DIR / ("phase9_egnn_pytorch_epochs.csv" if large else "phase8_egnn_pytorch_epochs.csv")

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt)

    metrics = {
        "experiment_id": experiment_id,
        "model": model_label,
        "framework": "pytorch",
        "representation": cfg["representation"],
        "dataset": "rcsb_xray_protein" if large else "raw_debug_mmcif",
        "n_proteins": len(graphs),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "batch_size": batch_size,
        "GPU": mem["gpu_name"],
        "task": "edge_ca_distance_regression",
        "note": "diagnostic edge-distance task; not biochemical function prediction",
        "final_loss": history[-1]["loss"],
        "final_val_loss": history[-1]["val_loss"],
        "training_time_s": elapsed,
        "memory": mem,
        "epochs": cfg["epochs"],
        "radius_cutoff": cutoff,
    }
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["epoch", "loss", "val_loss", "epoch_ms", "steps", "peak_allocated_mb"]
        )
        writer.writeheader()
        writer.writerows(history)

    print()
    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
