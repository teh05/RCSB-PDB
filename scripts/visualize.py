"""Plots for every processed structure (matplotlib + optional PyMOL cartoons)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.build import load_graph
from src.paths import CONFIGS_DIR, GRAPHS_DIR, PLOTS_DIR, STRUCTURES_DIR, ensure_dirs
from src.preprocessing.coords import load_coordinate_set
from src.visualization.plots import (
    plot_allatom_scatter,
    plot_ca_backbone,
    plot_protein_graph,
    write_protein_gallery,
)
from src.visualization.pymol_cartoon import render_raw_cartoons


def _select_structure_files(files: list[Path], max_n: int) -> list[Path]:
    files = sorted(files)
    if max_n > 0:
        return files[:max_n]
    return files


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "egnn.yaml"
    cfg = load_config(cfg_path)
    cutoff = float(cfg["radius_cutoff"])
    plot_max = int(cfg.get("plot_max", 0))
    cartoon_max = int(cfg.get("cartoon_max", 0))

    aa_files = _select_structure_files(list(STRUCTURES_DIR.glob("*_allatom.npz")), plot_max)
    pdb_ids: list[str] = []
    for i, aa in enumerate(aa_files, 1):
        pid = aa.name.replace("_allatom.npz", "").upper()
        pdb_ids.append(pid)
        try:
            coords = load_coordinate_set(aa)
            plot_allatom_scatter(coords)
            ca_path = STRUCTURES_DIR / f"{coords.pdb_id}_ca.npz"
            if ca_path.exists():
                plot_ca_backbone(load_coordinate_set(ca_path))
            gpath = GRAPHS_DIR / f"{coords.pdb_id}_ca_r{cutoff:g}.npz"
            if gpath.exists():
                plot_protein_graph(load_graph(gpath))
        except Exception as exc:
            print(f"skip plot {aa.name}: {exc}", flush=True)
        if i == 1 or i % 50 == 0 or i == len(aa_files):
            print(f"mpl {i}/{len(aa_files)} {pid}", flush=True)

    cartoons, skip = render_raw_cartoons(max_n=cartoon_max)
    if skip:
        print(skip)
    else:
        print(f"cartoons: {len(cartoons)}")

    extra = [p.name.replace("_cartoon.png", "") for p in PLOTS_DIR.glob("*_cartoon.png")]
    pdb_ids = sorted(set(pdb_ids) | set(extra))
    gallery = write_protein_gallery(pdb_ids)
    print(f"gallery: {gallery}")
    print(f"proteins plotted: {len(pdb_ids)}")
    print(f"output dir: {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
