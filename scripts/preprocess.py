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
    for structure in structures:
        try:
            all_atom = extract_all_atom(structure)
            ca = extract_ca(structure)
        except Exception as exc:
            print(f"skip {structure.pdb_id}: {exc}")
            continue
        save_coordinate_set(all_atom, STRUCTURES_DIR / f"{structure.pdb_id}_allatom.npz")
        save_coordinate_set(ca, STRUCTURES_DIR / f"{structure.pdb_id}_ca.npz")
        note = representation_cost_note(all_atom, ca)
        notes.append(note)
        print(note)
    print(f"wrote {csv_path}")
    print(f"coordinate files in {STRUCTURES_DIR}")
    print(f"processed {len(notes)} structures (skipped files logged above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
