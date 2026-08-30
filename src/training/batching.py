"""PyTorch training utilities for the debug GNN baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph.build import ProteinGraph
from src.graph.collate import collate_graphs


@dataclass
class BatchedGraph:
    x: torch.Tensor
    coords: torch.Tensor
    edge_index: torch.Tensor
    edge_dist: torch.Tensor
    batch: torch.Tensor
    pdb_ids: list[str]


def graphs_to_batch(graphs: list[ProteinGraph], device: torch.device) -> BatchedGraph:
    data = collate_graphs(graphs)
    return BatchedGraph(
        x=torch.from_numpy(data["x"]).to(device),
        coords=torch.from_numpy(data["coords"]).to(device),
        edge_index=torch.from_numpy(data["edge_index"]).to(device),
        edge_dist=torch.from_numpy(data["edge_dist"]).to(device),
        batch=torch.from_numpy(data["batch"]).to(device),
        pdb_ids=data["pdb_ids"],
    )
