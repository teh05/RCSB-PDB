"""Parse every mmCIF in raw/, write metadata + coordinate representations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import STRUCTURES_DIR, ensure_dirs
from src.preprocessing.coords import (
    extract_all_atom,
    extract_ca,
    representation_cost_note,
    save_coordinate_set,
)
from src.preprocessing.validate import summarize_raw_dir


def main() -> int:
    ensure_dirs()
    structures, csv_path = summarize_raw_dir()
    notes = []
    n_skip = 0
    for i, structure in enumerate(structures, 1):
        ca_out = STRUCTURES_DIR / f"{structure.pdb_id}_ca.npz"
        aa_out = STRUCTURES_DIR / f"{structure.pdb_id}_allatom.npz"
        if ca_out.is_file() and aa_out.is_file():
            n_skip += 1
            continue
        try:
            all_atom = extract_all_atom(structure)
            ca = extract_ca(structure)
        except Exception as exc:
            print(f"skip {structure.pdb_id}: {exc}")
            continue
        save_coordinate_set(all_atom, aa_out)
        save_coordinate_set(ca, ca_out)
        note = representation_cost_note(all_atom, ca)
        notes.append(note)
        if len(notes) == 1 or len(notes) % 100 == 0:
            print(note, flush=True)
    print(f"wrote {csv_path}")
    print(f"coordinate files in {STRUCTURES_DIR}")
    print(f"processed {len(notes)} new  reused {n_skip}  total structures {len(structures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
