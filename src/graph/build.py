"""Protein graphs G = (V, E, X, R) at residue (C-alpha) level.

Edge strategies:
  1. sequential: i <-> i+1 within the same chain
  2. radius: connect if ||CA_i - CA_j|| < cutoff (Angstrom)

Cutoff is never hard-coded at call sites — pass it from config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing.coords import AA20, CoordinateSet, UNK_RESIDUE_INDEX

EDGE_SEQUENTIAL = 0
EDGE_RADIUS = 1


@dataclass
class ProteinGraph:
    pdb_id: str
    representation: str
    coords: np.ndarray  # R, (N, 3)
    node_features: np.ndarray  # X, (N, F)
    residue_names: np.ndarray
    chain_ids: np.ndarray
    seq_ids: np.ndarray
    edge_index: np.ndarray  # (2, E) directed
    edge_type: np.ndarray  # (E,)
    edge_dist: np.ndarray  # (E,)
    radius_cutoff: float

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def residue_one_hot(residue_indices: np.ndarray) -> np.ndarray:
    n_classes = UNK_RESIDUE_INDEX + 1
    eye = np.eye(n_classes, dtype=np.float32)
    idx = np.clip(residue_indices.astype(np.int64), 0, n_classes - 1)
    return eye[idx]


def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def sequential_edges(chain_ids: np.ndarray) -> np.ndarray:
    """Directed sequential edges between consecutive nodes of the same chain."""
    src: list[int] = []
    dst: list[int] = []
    n = len(chain_ids)
    for i in range(n - 1):
        if chain_ids[i] == chain_ids[i + 1]:
            src.extend([i, i + 1])
            dst.extend([i + 1, i])
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray([src, dst], dtype=np.int64)


def radius_edges(coords: np.ndarray, cutoff: float) -> np.ndarray:
    """Directed radius edges (both directions). No self-loops."""
    if cutoff <= 0:
        raise ValueError("radius_cutoff must be > 0")
    dist = _pairwise_dist(coords)
    n = coords.shape[0]
    ii, jj = np.where((dist < cutoff) & (dist > 0) & (np.arange(n)[:, None] != np.arange(n)[None, :]))
    if ii.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([ii, jj], axis=0).astype(np.int64)


def build_residue_graph(
    coords: CoordinateSet,
    *,
    radius_cutoff: float,
    include_sequential: bool = True,
    include_radius: bool = True,
) -> ProteinGraph:
    if coords.representation != "ca":
        # Residue graph uses CA representatives even if all-atom is passed
        raise ValueError("build_residue_graph expects representation='ca'")

    x = residue_one_hot(coords.residue_indices)
    parts: list[np.ndarray] = []
    types: list[np.ndarray] = []

    if include_sequential:
        seq = sequential_edges(coords.chain_ids)
        parts.append(seq)
        types.append(np.full(seq.shape[1], EDGE_SEQUENTIAL, dtype=np.int64))
    if include_radius:
        rad = radius_edges(coords.coords, radius_cutoff)
        parts.append(rad)
        types.append(np.full(rad.shape[1], EDGE_RADIUS, dtype=np.int64))

    if not parts:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
    else:
        edge_index = np.concatenate(parts, axis=1)
        edge_type = np.concatenate(types, axis=0)

    # unique directed edges; sequential preferred if duplicated
    if edge_index.shape[1] > 0:
        keys = edge_index[0] * coords.n + edge_index[1]
        _, keep = np.unique(keys, return_index=True)
        keep = np.sort(keep)
        edge_index = edge_index[:, keep]
        edge_type = edge_type[keep]

    if edge_index.shape[1] > 0:
        src, dst = edge_index
        diff = coords.coords[src] - coords.coords[dst]
        edge_dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
    else:
        edge_dist = np.zeros((0,), dtype=np.float32)

    return ProteinGraph(
        pdb_id=coords.pdb_id,
        representation=coords.representation,
        coords=coords.coords.astype(np.float32),
        node_features=x,
        residue_names=coords.residue_names,
        chain_ids=coords.chain_ids,
        seq_ids=coords.seq_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_dist=edge_dist,
        radius_cutoff=float(radius_cutoff),
    )


def save_graph(graph: ProteinGraph, path) -> None:
    np.savez_compressed(
        path,
        pdb_id=graph.pdb_id,
        representation=graph.representation,
        coords=graph.coords,
        node_features=graph.node_features,
        residue_names=graph.residue_names,
        chain_ids=graph.chain_ids,
        seq_ids=graph.seq_ids,
        edge_index=graph.edge_index,
        edge_type=graph.edge_type,
        edge_dist=graph.edge_dist,
        radius_cutoff=graph.radius_cutoff,
        amino_acids=AA20,
    )


def load_ca_graphs(graphs_dir: str | Path, cutoff: float) -> list[ProteinGraph]:
    graphs_dir = Path(graphs_dir)
    pattern = f"*_ca_r{cutoff:g}.npz"
    paths = sorted(graphs_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No graphs matching {pattern} in {graphs_dir}. "
            "Run python scripts/build_graph.py first."
        )
    return [load_graph(p) for p in paths]


def load_graph(path) -> ProteinGraph:
    data = np.load(path, allow_pickle=True)
    return ProteinGraph(
        pdb_id=str(data["pdb_id"]),
        representation=str(data["representation"]),
        coords=data["coords"],
        node_features=data["node_features"],
        residue_names=data["residue_names"],
        chain_ids=data["chain_ids"],
        seq_ids=data["seq_ids"],
        edge_index=data["edge_index"],
        edge_type=data["edge_type"],
        edge_dist=data["edge_dist"],
        radius_cutoff=float(data["radius_cutoff"]),
    )
