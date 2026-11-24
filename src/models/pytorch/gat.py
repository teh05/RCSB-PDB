"""Single-head Graph Attention Network (Velickovic et al. 2018), simplified.

Topology + residue type only. Coordinates are not used (not E(3)-equivariant).
"""

from __future__ import annotations

import torch
from torch import nn


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    src = src.reshape(-1)
    mx = src.new_full((n,), float("-inf"))
    mx.scatter_reduce_(0, index, src, reduce="amax")
    exp = torch.exp(src - mx[index])
    denom = src.new_zeros(n)
    denom.index_add_(0, index, exp)
    return (exp / denom[index].clamp(min=1e-8)).unsqueeze(-1)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(out_dim * 2, 1, bias=False)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        wh = self.lin(h)
        e = torch.nn.functional.leaky_relu(
            self.att(torch.cat([wh[dst], wh[src]], dim=-1)), negative_slope=0.2
        )
        alpha = scatter_softmax(e, dst, h.size(0))
        out = wh.new_zeros(wh.shape)
        out.index_add_(0, dst, wh[src] * alpha)
        return torch.nn.functional.elu(out)


class GAT(nn.Module):
    uses_coords = False

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = [GATLayer(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(GATLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        src, dst = edge_index[0], edge_index[1]
        return self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
