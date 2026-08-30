from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.se3 import random_rotation
from src.models.pytorch.egnn import EGNN
from src.training.device import require_cuda

ATOL = 1e-3


def _chain_edges(n: int) -> torch.Tensor:
    src = torch.arange(n - 1)
    dst = torch.arange(1, n)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_egnn_se3_equivariance_cuda():
    device = require_cuda()
    torch.manual_seed(0)
    n, in_dim = 12, 21
    x = torch.eye(in_dim)[torch.randint(0, in_dim, (n,))].to(device)
    coords = torch.randn(n, 3, device=device)
    edge_index = _chain_edges(n).to(device)

    rng = np.random.default_rng(0)
    r = torch.tensor(random_rotation(rng), dtype=torch.float32, device=device)
    t = torch.tensor(rng.normal(size=3), dtype=torch.float32, device=device)

    model = EGNN(in_dim=in_dim, hidden_dim=32, num_layers=2).to(device)
    model.eval()
    with torch.no_grad():
        pred0, c0 = model.forward_full(x, coords, edge_index)
        pred_r, c_r = model.forward_full(x, coords @ r.T, edge_index)
        pred_t, c_t = model.forward_full(x, coords + t, edge_index)

    assert torch.allclose(pred0, pred_r, atol=ATOL, rtol=0)
    assert torch.allclose(pred0, pred_t, atol=ATOL, rtol=0)
    assert torch.allclose(c0 @ r.T, c_r, atol=ATOL, rtol=0)
    assert torch.allclose(c0 + t, c_t, atol=ATOL, rtol=0)
