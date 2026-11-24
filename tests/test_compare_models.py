from __future__ import annotations

import torch

from src.models.pytorch.factory import COMPARE_MODELS, build_pytorch_model


def test_all_models_forward_shape():
    n, e, din = 12, 20, 21
    x = torch.randn(n, din)
    coords = torch.randn(n, 3)
    ei = torch.randint(0, n, (2, e))
    for name in COMPARE_MODELS:
        model, label = build_pytorch_model(name, din, 16, 2, cutoff=8.0)
        if getattr(model, "uses_coords", False):
            pred = model(x, coords, ei)
        else:
            pred = model(x, ei)
        assert pred.shape == (e,), f"{label} {pred.shape}"
        loss = pred.pow(2).mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
