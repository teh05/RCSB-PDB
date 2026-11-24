"""Pairwise MLP baseline: no message passing.

Each residue is encoded independently. Edge distances are predicted from
concatenated endpoint embeddings. Ablation against graph models.
"""

from __future__ import annotations

import torch
from torch import nn


class PairMLP(nn.Module):
    uses_coords = False

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.node_mlp = nn.Sequential(*layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_mlp(x)
        src, dst = edge_index[0], edge_index[1]
        return self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
