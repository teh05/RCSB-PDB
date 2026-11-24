"""SchNet-style continuous-filter conv (Schutt et al. 2017), simplified.

Uses Gaussian RBF of ||x_i - x_j|| in messages. Coordinates enter only as
distances (SE(3)-invariant messages). Diagnostic edge head matches other models.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.pytorch.baseline import scatter_mean


class RBFExpansion(nn.Module):
    def __init__(self, n_rbf: int = 16, cutoff: float = 8.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = (n_rbf / cutoff) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        d = dist.unsqueeze(-1)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


class SchNetLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, rbf: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        w = self.filter(rbf)
        msg = self.phi(h[src]) * w
        return h + scatter_mean(msg, dst, h.size(0))


class SchNet(nn.Module):
    uses_coords = True

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, cutoff: float = 8.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.rbf = RBFExpansion(n_rbf=16, cutoff=cutoff)
        self.layers = nn.ModuleList([SchNetLayer(hidden_dim, 16) for _ in range(num_layers)])
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        dist = torch.linalg.norm(coords[dst] - coords[src], dim=-1)
        rbf = self.rbf(dist)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, rbf)
        return self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
