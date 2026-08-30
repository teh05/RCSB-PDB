"""Build residue graphs for every processed C-alpha structure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.build import build_residue_graph, save_graph
from src.paths import CONFIGS_DIR, GRAPHS_DIR, STRUCTURES_DIR, ensure_dirs
from src.preprocessing.coords import load_coordinate_set


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "baseline.yaml"
    cfg = load_config(cfg_path)
    cutoff = float(cfg["radius_cutoff"])
    files = sorted(STRUCTURES_DIR.glob("*_ca.npz"))
    if not files:
        raise FileNotFoundError(
            f"No C-alpha npz in {STRUCTURES_DIR}. Run python scripts/preprocess.py first."
        )
    for path in files:
        try:
            coords = load_coordinate_set(path)
            graph = build_residue_graph(coords, radius_cutoff=cutoff)
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
            continue
        out = GRAPHS_DIR / f"{graph.pdb_id}_ca_r{cutoff:g}.npz"
        save_graph(graph, out)
        n_seq = int((graph.edge_type == 0).sum())
        n_rad = int((graph.edge_type == 1).sum())
        print(
            f"{graph.pdb_id}  V={graph.n_nodes}  E={graph.n_edges}  "
            f"seq={n_seq}  radius={n_rad}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
