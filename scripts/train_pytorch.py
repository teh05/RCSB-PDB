"""Train a PyTorch GNN on the debug protein graphs.

Task: edge-wise C-alpha distance regression (self-supervised / diagnostic).
Labels are geometric distances derived from coordinates, not biological
function annotations. Three proteins are insufficient for generalization.

Models:
  gcn  — BaselineGCN (Phase 7), topology + residue type
  egnn — EGNN (Phase 8), SE(3) messages from coordinates
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

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


def _model_name(cfg: dict) -> str:
    return str(cfg.get("model", "gcn")).lower()


def build_model(cfg: dict, in_dim: int) -> tuple[nn.Module, str, str]:
    hidden = int(cfg["hidden_dim"])
    layers = int(cfg["num_layers"])
    name = _model_name(cfg)
    if name in {"gcn", "baselinegcn", "baseline"}:
        return BaselineGCN(in_dim, hidden, layers), "BaselineGCN", "phase7_baseline_debug"
    if name in {"egnn"}:
        return EGNN(in_dim, hidden, layers), "EGNN", "phase8_egnn_pytorch"
    raise ValueError(f"Unknown model {name!r}. Use gcn or egnn.")


def predict(model: nn.Module, batch: BatchedGraph) -> torch.Tensor:
    if getattr(model, "uses_coords", False):
        return model(batch.x, batch.coords, batch.edge_index)
    return model(batch.x, batch.edge_index)


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "baseline.yaml"
    cfg = load_config(cfg_path)

    if str(cfg.get("device", "cuda")).lower() != "cuda":
        raise RuntimeError("Training requires device: cuda. No CPU fallback.")

    device = require_cuda()
    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.manual_seed_all(int(cfg["seed"]))
    torch.cuda.reset_peak_memory_stats()

    cutoff = float(cfg["radius_cutoff"])
    graphs = load_ca_graphs(GRAPHS_DIR, cutoff)
    batch = graphs_to_batch(graphs, device)

    in_dim = int(batch.x.size(-1))
    model, model_label, experiment_id = build_model(cfg, in_dim)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.MSELoss()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device: {device}")
    print(f"model: {model_label}")
    print(f"graphs: {[g.pdb_id for g in graphs]}")
    print(f"nodes: {batch.x.size(0)} edges: {batch.edge_index.size(1)}")
    print("task: edge_ca_distance_regression (diagnostic, not biological labels)")
    print("split: none - debug set only, no generalization claim")
    print()

    model.train()
    pred = predict(model, batch)
    loss = loss_fn(pred, batch.edge_dist)
    loss.backward()
    opt.zero_grad(set_to_none=True)
    print(f"forward/backward ok  initial_loss={float(loss.detach()):.4f}")

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        pred = predict(model, batch)
        loss = loss_fn(pred, batch.edge_dist)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - step_t0) * 1000
        mem = gpu_memory_stats()
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "step_ms": step_ms,
            "peak_allocated_mb": mem["peak_allocated_mb"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  loss={row['loss']:.4f}  "
            f"step={step_ms:.1f}ms  peak_alloc={mem['peak_allocated_mb']:.1f}MB"
        )

    elapsed = time.perf_counter() - t0
    mem = gpu_memory_stats()
    mean_ex_first = None
    if len(history) > 1:
        mean_ex_first = sum(r["step_ms"] for r in history[1:]) / (len(history) - 1)

    if model_label == "BaselineGCN":
        ckpt = CHECKPOINTS_DIR / "baseline_gcn_debug.pt"
        json_path = METRICS_DIR / "phase7_baseline.json"
        csv_path = METRICS_DIR / "phase7_baseline_epochs.csv"
    else:
        ckpt = CHECKPOINTS_DIR / "egnn_pytorch_debug.pt"
        json_path = METRICS_DIR / "phase8_egnn_pytorch.json"
        csv_path = METRICS_DIR / "phase8_egnn_pytorch_epochs.csv"

    torch.save({"model": model.state_dict(), "config": cfg}, ckpt)

    metrics = {
        "experiment_id": experiment_id,
        "model": model_label,
        "framework": "pytorch",
        "representation": cfg["representation"],
        "dataset": "raw_debug_mmcif",
        "n_proteins": len(graphs),
        "pdb_ids": [g.pdb_id for g in graphs],
        "GPU": mem["gpu_name"],
        "task": "edge_ca_distance_regression",
        "note": "engineering debug on 3 proteins; not scientific generalization",
        "final_loss": history[-1]["loss"],
        "training_time_s": elapsed,
        "first_step_ms": history[0]["step_ms"],
        "mean_step_ms_ex_first": mean_ex_first,
        "memory": mem,
        "epochs": cfg["epochs"],
        "radius_cutoff": cutoff,
    }
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
