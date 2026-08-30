"""Diagnostic 3D / graph plots from processed structures.

Optional: PyMOL cartoons from raw mmCIF when an executable is on PATH.
Large datasets only plot a small sample (plot_max / cartoon_max).
"""

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
from src.visualization.plots import plot_allatom_scatter, plot_ca_backbone, plot_protein_graph
from src.visualization.pymol_cartoon import render_raw_cartoons


def _select_structure_files(files: list[Path], max_n: int, prefer: tuple[str, ...]) -> list[Path]:
    by_id = {}
    for p in files:
        stem = p.name.replace("_allatom.npz", "")
        by_id[stem.upper()] = p
    chosen: list[Path] = []
    for pid in prefer:
        if pid in by_id and by_id[pid] not in chosen:
            chosen.append(by_id[pid])
        if max_n and len(chosen) >= max_n:
            return chosen
    for p in files:
        if max_n and len(chosen) >= max_n:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "baseline.yaml"
    cfg = load_config(cfg_path)
    cutoff = float(cfg["radius_cutoff"])
    plot_max = int(cfg.get("plot_max", 8))
    cartoon_max = int(cfg.get("cartoon_max", 8))
    prefer = ("1UBQ", "1LYZ", "4HHB")

    written: list[str] = []
    aa_files = _select_structure_files(sorted(STRUCTURES_DIR.glob("*_allatom.npz")), plot_max, prefer)
    for aa in aa_files:
        coords = load_coordinate_set(aa)
        written.append(str(plot_allatom_scatter(coords)))
        ca_path = STRUCTURES_DIR / f"{coords.pdb_id}_ca.npz"
        if ca_path.exists():
            written.append(str(plot_ca_backbone(load_coordinate_set(ca_path))))
        gpath = GRAPHS_DIR / f"{coords.pdb_id}_ca_r{cutoff:g}.npz"
        if gpath.exists():
            written.append(str(plot_protein_graph(load_graph(gpath))))

    cartoons, skip = render_raw_cartoons(max_n=cartoon_max, prefer_ids=prefer)
    if skip:
        print(skip)
        print("Install PyMOL and retry to write results/plots/<PDB>_cartoon.png from raw mmCIF.")
        print("  pymol raw/<PDB>.cif.gz")
    else:
        written.extend(str(p) for p in cartoons)

    print("plots:")
    for p in written:
        print(" ", p)
    print(f"output dir: {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
