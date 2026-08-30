"""Smoke-parse a single mmCIF (STEP 2 default: 1UBQ)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.mmcif import load_structure


def main() -> int:
    path = ROOT / "raw" / "1UBQ.cif.gz"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    structure = load_structure(path)
    summary = structure.summary()
    print(json.dumps(summary, indent=2))

    # Diagnostic residue / atom breakdown for model 0
    model = structure.models[0]
    polymer_res = 0
    water_res = 0
    other_het = 0
    ca_count = 0
    for chain in model.chains:
        for res in chain.residues:
            if res.name in {"HOH", "WAT", "DOD"}:
                water_res += 1
            elif any(a.is_hetatm for a in res.atoms):
                other_het += 1
            else:
                polymer_res += 1
            if any(a.atom_name == "CA" for a in res.atoms):
                ca_count += 1
    print()
    print(f"polymer_residues: {polymer_res}")
    print(f"water_residues: {water_res}")
    print(f"other_hetero_residues: {other_het}")
    print(f"CA_atoms: {ca_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
