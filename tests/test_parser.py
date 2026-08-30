"""Parser unit tests — structures must not be empty."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.discover import discover_mmcif_files
from src.io.mmcif import load_structure

RAW = ROOT / "raw"
UBQ = RAW / "1UBQ.cif.gz"
LYZ = RAW / "1LYZ.cif.gz"
HHB = RAW / "4HHB.cif.gz"


@pytest.mark.skipif(not UBQ.exists(), reason="1UBQ.cif.gz missing")
def test_load_1ubq_not_empty():
    s = load_structure(UBQ)
    assert s.pdb_id == "1UBQ"
    assert s.n_models == 1
    assert s.n_chains >= 1
    assert "A" in s.chain_ids
    assert s.n_residues > 0
    assert s.n_atoms > 0
    ca = sum(
        1
        for c in s.models[0].chains
        for r in c.residues
        for a in r.atoms
        if a.atom_name == "CA"
    )
    assert ca == 76
    assert s.experimental_method is not None
    assert s.resolution == pytest.approx(1.8, abs=1e-3)


@pytest.mark.skipif(not UBQ.exists(), reason="1UBQ.cif.gz missing")
def test_1ubq_coordinates_finite():
    s = load_structure(UBQ)
    atom = s.models[0].chains[0].residues[0].atoms[0]
    assert all(isinstance(v, float) for v in (atom.x, atom.y, atom.z))


@pytest.mark.skipif(not LYZ.exists(), reason="1LYZ.cif.gz missing")
def test_load_1lyz_not_empty():
    s = load_structure(LYZ)
    assert s.pdb_id == "1LYZ"
    assert s.n_atoms > 0
    ca = sum(
        1
        for c in s.models[0].chains
        for r in c.residues
        for a in r.atoms
        if a.atom_name == "CA"
    )
    assert ca == 129
    assert s.resolution == pytest.approx(2.0, abs=1e-3)
    icodes = [
        r.insertion_code
        for c in s.models[0].chains
        for r in c.residues
        if r.insertion_code
    ]
    assert icodes, "1LYZ should expose insertion codes on some residues/atoms"


@pytest.mark.skipif(not HHB.exists(), reason="4HHB.cif.gz missing")
def test_load_4hhb_not_empty():
    s = load_structure(HHB)
    assert s.pdb_id == "4HHB"
    assert s.n_atoms > 0
    for cid in "ABCD":
        assert cid in s.chain_ids
    ca = sum(
        1
        for c in s.models[0].chains
        for r in c.residues
        for a in r.atoms
        if a.atom_name == "CA"
    )
    assert ca == 574
    het = {r.name for c in s.models[0].chains for r in c.residues if any(a.is_hetatm for a in r.atoms)}
    assert "HEM" in het
    assert s.resolution == pytest.approx(1.74, abs=1e-3)


def test_discover_is_generic():
    files = discover_mmcif_files(RAW)
    assert files
    stems = {p.name.split(".")[0].upper() for p in files}
    assert stems == {p.name.split(".")[0].upper() for p in files}
