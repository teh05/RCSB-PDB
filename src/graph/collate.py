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


def pad_collated(data: dict, max_nodes: int, max_edges: int) -> dict:
    """Pad a collated batch so JAX can JIT a fixed shape.

    One dummy node is reserved at the first padded index. Dummy edges are
    dummy->dummy and must be ignored via edge_mask in the loss.
    """
    x = np.asarray(data["x"], dtype=np.float32)
    coords = np.asarray(data["coords"], dtype=np.float32)
    ei = np.asarray(data["edge_index"], dtype=np.int64)
    dist = np.asarray(data["edge_dist"], dtype=np.float32)
    n = int(x.shape[0])
    e = int(ei.shape[1])
    if max_nodes < 2:
        raise ValueError("max_nodes must be >= 2")
    if n > max_nodes - 1:
        raise ValueError(f"batch has {n} nodes; need max_nodes >= {n + 1}")
    if e > max_edges:
        raise ValueError(f"batch has {e} edges; need max_edges >= {e}")
    dummy = n
    x_p = np.zeros((max_nodes, x.shape[1]), dtype=np.float32)
    x_p[:n] = x
    c_p = np.zeros((max_nodes, 3), dtype=np.float32)
    c_p[:n] = coords
    ei_p = np.full((2, max_edges), dummy, dtype=np.int64)
    if e:
        ei_p[:, :e] = ei
    d_p = np.zeros((max_edges,), dtype=np.float32)
    d_p[:e] = dist
    node_mask = np.zeros((max_nodes,), dtype=np.float32)
    node_mask[:n] = 1.0
    edge_mask = np.zeros((max_edges,), dtype=np.float32)
    edge_mask[:e] = 1.0
    out = dict(data)
    out.update(
        {
            "x": x_p,
            "coords": c_p,
            "edge_index": ei_p,
            "edge_dist": d_p,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
        }
    )
    return out
