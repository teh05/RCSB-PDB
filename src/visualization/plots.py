"""Diagnostic matplotlib visualizations (not publication figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.graph.build import EDGE_RADIUS, EDGE_SEQUENTIAL, ProteinGraph
from src.paths import PLOTS_DIR, ensure_dirs
from src.preprocessing.coords import CoordinateSet


def _chain_colors(chain_ids: np.ndarray) -> np.ndarray:
    uniq = list(dict.fromkeys(chain_ids.tolist()))
    lut = {c: i for i, c in enumerate(uniq)}
    return np.asarray([lut[c] for c in chain_ids], dtype=np.float32)


def plot_allatom_scatter(coords: CoordinateSet, out_path: Path | None = None) -> Path:
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / f"{coords.pdb_id}_allatom_scatter.png")
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    c = _chain_colors(coords.chain_ids)
    ax.scatter(coords.coords[:, 0], coords.coords[:, 1], coords.coords[:, 2], c=c, s=4, alpha=0.7)
    ax.set_title(f"{coords.pdb_id} all-atom scatter (N={coords.n})")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_ca_backbone(coords: CoordinateSet, out_path: Path | None = None) -> Path:
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / f"{coords.pdb_id}_ca_backbone.png")
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    chains = coords.chain_ids
    xyz = coords.coords
    for chain in dict.fromkeys(chains.tolist()):
        mask = chains == chain
        pts = xyz[mask]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.4, label=f"chain {chain}")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8)
    ax.set_title(f"{coords.pdb_id} C-alpha backbone (N={coords.n})")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_protein_graph(graph: ProteinGraph, out_path: Path | None = None) -> Path:
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / f"{graph.pdb_id}_graph.png")
    xyz = graph.coords
    fig, ax = plt.subplots(figsize=(7, 6))
    # 2D projection x-y for readability of edges
    ei = graph.edge_index
    et = graph.edge_type
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i >= j:
            continue
        color = "#9aa0a6" if et[k] == EDGE_RADIUS else "#1f4e79"
        lw = 0.4 if et[k] == EDGE_RADIUS else 1.2
        ax.plot([xyz[i, 0], xyz[j, 0]], [xyz[i, 1], xyz[j, 1]], color=color, lw=lw, alpha=0.5)
    c = _chain_colors(graph.chain_ids)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=c, s=18, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"{graph.pdb_id} graph XY (seq=dark, radius={graph.radius_cutoff:g} Å, "
        f"V={graph.n_nodes}, E={graph.n_edges})"
    )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
