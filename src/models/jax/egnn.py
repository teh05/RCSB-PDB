"""Flax EGNN matching src.models.pytorch.egnn (architecture, not init).

Do not import this module in a process that already imported torch.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


def scatter_mean(src: jnp.ndarray, index: jnp.ndarray, n: int) -> jnp.ndarray:
    out = jnp.zeros((n,) + src.shape[1:], dtype=src.dtype).at[index].add(src)
    ones = jnp.ones((src.shape[0],) + (1,) * (src.ndim - 1), dtype=src.dtype)
    denom = jnp.zeros((n,) + (1,) * (src.ndim - 1), dtype=src.dtype).at[index].add(ones)
    return out / jnp.maximum(denom, 1.0)


class EGNNLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, h, coords, edge_index):
        src, dst = edge_index[0], edge_index[1]
        diff = coords[dst] - coords[src]
        sqdist = jnp.sum(diff * diff, axis=-1, keepdims=True)
        m_ij = nn.silu(
            nn.Dense(self.hidden_dim, name="phi_e1")(
                jnp.concatenate([h[dst], h[src], sqdist], axis=-1)
            )
        )
        m_ij = nn.silu(nn.Dense(self.hidden_dim, name="phi_e2")(m_ij))
        coord_w = jnp.tanh(
            nn.Dense(1, name="phi_x2")(nn.silu(nn.Dense(self.hidden_dim, name="phi_x1")(m_ij)))
        )
        coords = coords + scatter_mean(diff * coord_w, dst, coords.shape[0])
        m_i = scatter_mean(m_ij, dst, h.shape[0])
        dh = nn.Dense(self.hidden_dim, name="phi_h2")(
            nn.silu(
                nn.Dense(self.hidden_dim, name="phi_h1")(jnp.concatenate([h, m_i], axis=-1))
            )
        )
        return h + dh, coords


class EGNN(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x, coords, edge_index, return_coords: bool = False):
        h = nn.silu(nn.Dense(self.hidden_dim, name="input_proj")(x))
        for i in range(self.num_layers):
            h, coords = EGNNLayer(self.hidden_dim, name=f"layer{i}")(h, coords, edge_index)
        src, dst = edge_index[0], edge_index[1]
        hidden = nn.relu(
            nn.Dense(self.hidden_dim, name="edge1")(jnp.concatenate([h[src], h[dst]], axis=-1))
        )
        pred = nn.Dense(1, name="edge2")(hidden).squeeze(-1)
        if return_coords:
            return pred, coords
        return pred
