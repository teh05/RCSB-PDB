"""Diagnostic matplotlib visualizations (not publication figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from matplotlib.collections import LineCollection

from src.graph.build import EDGE_RADIUS, ProteinGraph
from src.paths import PLOTS_DIR, ensure_dirs
from src.preprocessing.coords import CoordinateSet


def _chain_colors(chain_ids: np.ndarray) -> np.ndarray:
    uniq = list(dict.fromkeys(chain_ids.tolist()))
    lut = {c: i for i, c in enumerate(uniq)}
    return np.asarray([lut[c] for c in chain_ids], dtype=np.float32)


def plot_allatom_scatter(coords: CoordinateSet, out_path: Path | None = None) -> Path:
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / f"{coords.pdb_id}_allatom_scatter.png")
    if out_path.is_file() and out_path.stat().st_size > 100:
        return out_path
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    c = _chain_colors(coords.chain_ids)
    ax.scatter(coords.coords[:, 0], coords.coords[:, 1], coords.coords[:, 2], c=c, s=4, alpha=0.7)
    ax.set_title(f"{coords.pdb_id} all-atom scatter (N={coords.n})")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_ca_backbone(coords: CoordinateSet, out_path: Path | None = None) -> Path:
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / f"{coords.pdb_id}_ca_backbone.png")
    if out_path.is_file() and out_path.stat().st_size > 100:
        return out_path
    fig = plt.figure(figsize=(6, 5))
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
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_protein_graph(graph: ProteinGraph, out_path: Path | None = None) -> Path:
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / f"{graph.pdb_id}_graph.png")
    if out_path.is_file() and out_path.stat().st_size > 100:
        return out_path
    xyz = graph.coords
    fig, ax = plt.subplots(figsize=(6, 5))
    ei = graph.edge_index
    et = graph.edge_type
    seq_segs: list[list[tuple[float, float]]] = []
    rad_segs: list[list[tuple[float, float]]] = []
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i >= j:
            continue
        seg = [(float(xyz[i, 0]), float(xyz[i, 1])), (float(xyz[j, 0]), float(xyz[j, 1]))]
        if et[k] == EDGE_RADIUS:
            rad_segs.append(seg)
        else:
            seq_segs.append(seg)
    if rad_segs:
        ax.add_collection(LineCollection(rad_segs, colors="#9aa0a6", linewidths=0.35, alpha=0.45))
    if seq_segs:
        ax.add_collection(LineCollection(seq_segs, colors="#1f4e79", linewidths=1.1, alpha=0.7))
    c = _chain_colors(graph.chain_ids)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=c, s=14, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"{graph.pdb_id} graph XY (seq=dark, radius={graph.radius_cutoff:g} Å, "
        f"V={graph.n_nodes}, E={graph.n_edges})"
    )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def write_protein_gallery(
    pdb_ids: list[str],
    out_path: Path | None = None,
    *,
    src_prefix: str = "",
) -> Path:
    """HTML gallery: cartoon, all-atom, CA backbone, graph for each PDB id."""
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / "gallery.html")
    prefix = src_prefix
    cards: list[str] = []
    for pid in pdb_ids:
        imgs = [
            (f"{pid}_cartoon.png", "cartoon"),
            (f"{pid}_allatom_scatter.png", "all-atom"),
            (f"{pid}_ca_backbone.png", "C-alpha"),
            (f"{pid}_graph.png", "graf"),
        ]
        cells = []
        for fname, label in imgs:
            p = PLOTS_DIR / fname
            if p.is_file():
                cells.append(
                    f'<figure><img src="{prefix}{fname}" alt="{pid} {label}" loading="lazy">'
                    f"<figcaption>{label}</figcaption></figure>"
                )
            else:
                cells.append(f"<figure><figcaption>{label}: missing</figcaption></figure>")
        cards.append(f'<section id="{pid}"><h2>{pid}</h2><div class="row">{"".join(cells)}</div></section>')
    nav = "".join(f'<a href="#{p}">{p}</a> ' for p in pdb_ids)
    html = f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>Gallery {len(pdb_ids)} protein</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 16px; background: #f7f7f5; color: #1a1a1a; }}
h1 {{ font-size: 1.3rem; }}
h2 {{ font-size: 1.05rem; margin: 1.2rem 0 0.4rem; }}
.row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
figure {{ margin: 0; background: #fff; padding: 6px; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 0.75rem; color: #555; margin-top: 4px; }}
nav {{ max-height: 9rem; overflow: auto; line-height: 1.7; }}
nav a {{ margin-right: 0.55rem; font-size: 0.78rem; }}
@media (max-width: 900px) {{ .row {{ grid-template-columns: repeat(2, 1fr); }} }}
</style></head><body>
<h1>{len(pdb_ids)} protein — cartoon, all-atom, C-alpha, graf</h1>
<nav>{nav}</nav>
{"".join(cards)}
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
