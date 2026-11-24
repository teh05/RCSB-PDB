"""Load C-alpha graphs and apply the config residue filter + split.

NumPy only — no torch/jax.
"""

from __future__ import annotations

from src.graph.build import ProteinGraph, load_ca_graphs
from src.paths import GRAPHS_DIR
from src.training.split import split_indices


def load_filtered_graphs(cfg: dict) -> list[ProteinGraph]:
    cutoff = float(cfg["radius_cutoff"])
    graphs = load_ca_graphs(GRAPHS_DIR, cutoff)
    min_ca = int(cfg.get("min_polymer_residues", 50))
    max_ca = int(cfg.get("max_polymer_residues", 350))
    graphs = [g for g in graphs if min_ca <= g.n_nodes <= max_ca]
    if not graphs:
        raise RuntimeError("No graphs in residue-count filter. Run preprocess/build_graph.")
    return graphs


def split_graph_lists(
    graphs: list[ProteinGraph], cfg: dict
) -> tuple[list[ProteinGraph], list[ProteinGraph], list[ProteinGraph]]:
    seed = int(cfg["seed"])
    val_fraction = float(cfg.get("val_fraction", 0.2))
    test_fraction = float(cfg.get("test_fraction", 0.0))
    train_idx, val_idx, test_idx = split_indices(
        len(graphs),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train = [graphs[int(i)] for i in train_idx]
    val = [graphs[int(i)] for i in val_idx]
    test = [graphs[int(i)] for i in test_idx]
    return train, val, test
