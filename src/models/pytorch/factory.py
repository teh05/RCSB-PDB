"""Build PyTorch diagnostic models by name."""

from __future__ import annotations

from torch import nn

from src.models.pytorch.baseline import BaselineGCN
from src.models.pytorch.egnn import EGNN
from src.models.pytorch.gat import GAT
from src.models.pytorch.pair_mlp import PairMLP
from src.models.pytorch.schnet import SchNet

MODEL_ALIASES = {
    "mlp": "PairMLP",
    "pairmlp": "PairMLP",
    "pair_mlp": "PairMLP",
    "gcn": "BaselineGCN",
    "baselinegcn": "BaselineGCN",
    "baseline": "BaselineGCN",
    "gat": "GAT",
    "schnet": "SchNet",
    "egnn": "EGNN",
}

COMPARE_MODELS = ("mlp", "gcn", "gat", "schnet", "egnn")


def normalize_model_name(name: str) -> str:
    key = str(name).lower().strip()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model {name!r}. Use one of: {', '.join(COMPARE_MODELS)}."
        )
    return MODEL_ALIASES[key]


def build_pytorch_model(
    name: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    *,
    cutoff: float = 8.0,
) -> tuple[nn.Module, str]:
    label = normalize_model_name(name)
    if label == "PairMLP":
        return PairMLP(in_dim, hidden_dim, num_layers), label
    if label == "BaselineGCN":
        return BaselineGCN(in_dim, hidden_dim, num_layers), label
    if label == "GAT":
        return GAT(in_dim, hidden_dim, num_layers), label
    if label == "SchNet":
        return SchNet(in_dim, hidden_dim, num_layers, cutoff=cutoff), label
    return EGNN(in_dim, hidden_dim, num_layers), label


def slug_for(label: str) -> str:
    return {
        "PairMLP": "mlp",
        "BaselineGCN": "gcn",
        "GAT": "gat",
        "SchNet": "schnet",
        "EGNN": "egnn",
    }[label]
