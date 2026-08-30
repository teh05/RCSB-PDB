"""Generic mmCIF discovery — do not hard-code PDB IDs."""

from __future__ import annotations

from pathlib import Path


def discover_mmcif_files(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.cif.gz")) + sorted(raw_dir.glob("*.cif"))
    # de-duplicate if both exist; prefer listing unique stems
    seen: set[str] = set()
    out: list[Path] = []
    for p in files:
        key = p.name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
