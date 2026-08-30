from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.pymol_cartoon import (
    _pml_script,
    find_pymol,
    pdb_id_from_mmcif,
    render_cartoon,
)

RAW = ROOT / "raw"
UBQ = RAW / "1UBQ.cif.gz"


def test_pdb_id_from_mmcif():
    assert pdb_id_from_mmcif(Path("1ubq.cif.gz")) == "1UBQ"
    assert pdb_id_from_mmcif(Path("4HHB.cif")) == "4HHB"


def test_pml_script_uses_cartoon_and_quoted_paths(tmp_path: Path):
    cif = tmp_path / "1UBQ.cif.gz"
    png = tmp_path / "1UBQ_cartoon.png"
    text = _pml_script(cif, png, "1UBQ")
    assert "show cartoon" in text
    assert "show sticks, organic" in text
    assert "quit" in text
    assert cif.resolve().as_posix() in text
    assert png.resolve().as_posix() in text


@pytest.mark.skipif(find_pymol() is None, reason="PyMOL not installed")
@pytest.mark.skipif(not UBQ.exists(), reason="missing mmCIF")
def test_render_1ubq_cartoon(tmp_path: Path):
    out = tmp_path / "1UBQ_cartoon.png"
    written = render_cartoon(UBQ, out)
    assert written.is_file()
    assert written.stat().st_size > 1000
