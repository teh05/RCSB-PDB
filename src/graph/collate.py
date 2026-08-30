"""Framework-free graph collation (NumPy only)."""

from __future__ import annotations

import numpy as np

from src.graph.build import ProteinGraph


def collate_graphs(graphs: list[ProteinGraph]) -> dict:
    xs: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    pdb_ids: list[str] = []
    offset = 0
    for i, g in enumerate(graphs):
        n = g.n_nodes
        xs.append(np.asarray(g.node_features, dtype=np.float32))
        coords.append(np.asarray(g.coords, dtype=np.float32))
        edges.append(np.asarray(g.edge_index, dtype=np.int64) + offset)
        dists.append(np.asarray(g.edge_dist, dtype=np.float32))
        batch.append(np.full((n,), i, dtype=np.int64))
        pdb_ids.append(g.pdb_id)
        offset += n
    return {
        "x": np.concatenate(xs, axis=0),
        "coords": np.concatenate(coords, axis=0),
        "edge_index": np.concatenate(edges, axis=1),
        "edge_dist": np.concatenate(dists, axis=0),
        "batch": np.concatenate(batch, axis=0),
        "pdb_ids": pdb_ids,
    }
