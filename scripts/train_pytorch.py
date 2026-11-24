"""Train a PyTorch model on protein graphs.

Task: edge-wise C-alpha distance regression (diagnostic).
Supports PairMLP, BaselineGCN, GAT, SchNet, EGNN.
"""

from __future__ import annotations

import argparse
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
from src.models.pytorch.factory import build_pytorch_model, slug_for
from src.paths import CHECKPOINTS_DIR, CONFIGS_DIR, METRICS_DIR, ensure_dirs
from src.training.batching import BatchedGraph, graphs_to_batch
from src.training.dataset import load_filtered_graphs, split_graph_lists
from src.training.device import gpu_memory_stats, require_cuda
from src.training.split import batched_slices


def _finite_or_none(value: float) -> float | None:
    if value != value:
        return None
    return float(value)


def predict(model: nn.Module, batch: BatchedGraph) -> torch.Tensor:
    if getattr(model, "uses_coords", False):
        return model(batch.x, batch.coords, batch.edge_index)
    return model(batch.x, batch.edge_index)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def eval_loss(model: nn.Module, graphs: list, device: torch.device, batch_size: int) -> float:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default=str(CONFIGS_DIR / "egnn.yaml"))
    p.add_argument("--model", default=None)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--metrics-csv", default=None)
    return p.parse_args()


def main() -> int:
    ensure_dirs()
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        cfg["num_layers"] = args.num_layers
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    if str(cfg.get("device", "cuda")).lower() != "cuda":
        raise RuntimeError("Training requires device: cuda. No CPU fallback.")

    device = require_cuda()
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    cutoff = float(cfg["radius_cutoff"])
    graphs = load_filtered_graphs(cfg)
    train_graphs, val_graphs, test_graphs = split_graph_lists(graphs, cfg)
    batch_size = int(cfg.get("batch_size", 8))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    test_fraction = float(cfg.get("test_fraction", 0.0))

    in_dim = int(train_graphs[0].node_features.shape[-1])
    model, model_label = build_pytorch_model(
        str(cfg.get("model", "egnn")),
        in_dim,
        int(cfg["hidden_dim"]),
        int(cfg["num_layers"]),
        cutoff=cutoff,
    )
    slug = slug_for(model_label)
    large = len(graphs) >= 50
    experiment_id = args.experiment_id or (
        f"phase10_{slug}_pytorch" if large else f"debug_{slug}_pytorch"
    )
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.MSELoss()
    n_params = count_params(model)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device: {device}")
    print(f"model: {model_label}  params: {n_params}")
    print(
        f"n_proteins: {len(graphs)}  train: {len(train_graphs)}  "
        f"val: {len(val_graphs)}  test: {len(test_graphs)}"
    )
    print(f"batch_size: {batch_size}")
    if val_fraction <= 0 and test_fraction <= 0:
        print("split: train-all")
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
        val_loss = eval_loss(model, val_graphs, device, batch_size)
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
    train_mse = eval_loss(model, train_graphs, device, batch_size)
    test_mse = eval_loss(model, test_graphs, device, batch_size)
    mem = gpu_memory_stats()

    ckpt = Path(args.ckpt) if args.ckpt else (CHECKPOINTS_DIR / f"{slug}_pytorch.pt")
    json_path = Path(args.metrics_json) if args.metrics_json else (METRICS_DIR / f"{experiment_id}.json")
    csv_path = Path(args.metrics_csv) if args.metrics_csv else (METRICS_DIR / f"{experiment_id}_epochs.csv")

    torch.save({"model": model.state_dict(), "config": cfg, "experiment_id": experiment_id}, ckpt)

    metrics = {
        "experiment_id": experiment_id,
        "model": model_label,
        "framework": "pytorch",
        "representation": cfg["representation"],
        "dataset": "rcsb_xray_protein" if large else "raw_debug_mmcif",
        "n_proteins": len(graphs),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_test": len(test_graphs),
        "n_params": n_params,
        "hidden_dim": int(cfg["hidden_dim"]),
        "num_layers": int(cfg["num_layers"]),
        "learning_rate": float(cfg["learning_rate"]),
        "pdb_ids": [g.pdb_id for g in graphs],
        "batch_size": batch_size,
        "GPU": mem["gpu_name"],
        "task": "edge_ca_distance_regression",
        "note": "diagnostic edge-distance task; not biochemical function prediction",
        "final_loss": history[-1]["loss"],
        "final_train_mse": _finite_or_none(train_mse),
        "final_val_loss": _finite_or_none(float(history[-1]["val_loss"])),
        "final_test_mse": _finite_or_none(test_mse),
        "training_time_s": elapsed,
        "memory": mem,
        "epochs": cfg["epochs"],
        "radius_cutoff": cutoff,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
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
