"""Minimal GCN baseline (PyTorch).

Diagnostic task (not a biological label):
  Edge-wise C-alpha distance regression.
  For each graph edge (i, j), predict ||r_i - r_j|| from node embeddings
  that are computed from residue-type features and graph connectivity.

This verifies: graph tensors, batching, forward, backward, CUDA execution.
It is not a claim of biochemical function prediction.
"""

from __future__ import annotations

import torch
from torch import nn


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros(n, src.size(-1))
    out.index_add_(0, index, src)
    denom = src.new_zeros(n, 1)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    denom.index_add_(0, index, ones)
    return out / denom.clamp(min=1.0)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nbr = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        nbr = scatter_mean(h[src], dst, h.size(0))
        return torch.relu(self.lin_self(h) + self.lin_nbr(nbr))


class BaselineGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = [GraphConv(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(GraphConv(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.embed(x, edge_index)
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([h[src], h[dst]], dim=-1)
        return self.edge_mlp(pair).squeeze(-1)
