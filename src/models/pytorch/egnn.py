"""E(3)-equivariant GNN (EGNN, Satorras et al. 2021).

Node embeddings h are SE(3)-invariant. Coordinates are SE(3)-equivariant.
The diagnostic edge head predicts C-alpha distances from invariant embeddings
(same head shape as BaselineGCN). Geometry enters messages as ||x_i - x_j||^2.

This is an engineering baseline, not a claim of biochemical function prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.pytorch.baseline import scatter_mean


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, h: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        # i = dst (updated node), j = src (neighbor)
        diff = coords[dst] - coords[src]
        sqdist = (diff * diff).sum(dim=-1, keepdim=True)
        m_ij = self.phi_e(torch.cat([h[dst], h[src], sqdist], dim=-1))
        coords = coords + scatter_mean(diff * self.phi_x(m_ij), dst, coords.size(0))
        m_i = scatter_mean(m_ij, dst, h.size(0))
        h = h + self.phi_h(torch.cat([h, m_i], dim=-1))
        return h, coords


class EGNN(nn.Module):
    uses_coords = True

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def propagate(
        self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)
        return h, coords

    def forward_full(
        self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, coords_out = self.propagate(x, coords, edge_index)
        src, dst = edge_index[0], edge_index[1]
        pred = self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
        return pred, coords_out

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        pred, _ = self.forward_full(x, coords, edge_index)
        return pred
