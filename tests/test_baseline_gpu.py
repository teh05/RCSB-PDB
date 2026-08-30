from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.build import build_residue_graph
from src.io.mmcif import load_structure
from src.models.pytorch.baseline import BaselineGCN
from src.preprocessing.coords import extract_ca
from src.training.batching import graphs_to_batch
from src.training.device import require_cuda

RAW = ROOT / "raw"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not (RAW / "1UBQ.cif.gz").exists(), reason="missing mmCIF")
def test_baseline_forward_backward_cuda():
    device = require_cuda()
    s = load_structure(RAW / "1UBQ.cif.gz")
    g = build_residue_graph(extract_ca(s), radius_cutoff=8.0)
    batch = graphs_to_batch([g], device)
    model = BaselineGCN(in_dim=batch.x.size(-1), hidden_dim=32, num_layers=2).to(device)
    pred = model(batch.x, batch.edge_index)
    assert pred.shape == batch.edge_dist.shape
    loss = torch.nn.functional.mse_loss(pred, batch.edge_dist)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert next(model.parameters()).device.type == "cuda"
