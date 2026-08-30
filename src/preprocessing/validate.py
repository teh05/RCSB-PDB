"""Structure validation reports for the development mmCIF set."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.io.mmcif import StructureData, load_structure
from src.io.discover import discover_mmcif_files
from src.paths import METADATA_DIR, RAW_DIR, ensure_dirs


def _n_ca(structure: StructureData) -> int:
    return sum(
        1
        for chain in structure.models[0].chains
        for res in chain.residues
        for atom in res.atoms
        if atom.atom_name == "CA"
    )


def _n_polymer_residues(structure: StructureData) -> int:
    return sum(
        1
        for chain in structure.models[0].chains
        for res in chain.residues
        if any(a.atom_name == "CA" for a in res.atoms)
    )


def _n_water(structure: StructureData) -> int:
    return sum(
        1
        for chain in structure.models[0].chains
        for res in chain.residues
        if res.name in {"HOH", "WAT", "DOD"}
    )


def structure_report(structure: StructureData) -> dict[str, Any]:
    return {
        "pdb_id": structure.pdb_id,
        "n_models": structure.n_models,
        "n_chains": structure.n_chains,
        "chain_ids": ";".join(structure.chain_ids),
        "n_residues": structure.n_residues,
        "n_polymer_residues": _n_polymer_residues(structure),
        "n_water_residues": _n_water(structure),
        "n_atoms": structure.n_atoms,
        "n_ca": _n_ca(structure),
        "experimental_method": structure.experimental_method or "",
        "resolution": structure.resolution if structure.resolution is not None else "",
        "title": structure.title or "",
        "space_group": structure.space_group or "",
        "source_path": structure.source_path,
    }


def validate_not_empty(structure: StructureData) -> None:
    if structure.n_models < 1:
        raise ValueError(f"{structure.pdb_id}: no models")
    if structure.n_atoms < 1:
        raise ValueError(f"{structure.pdb_id}: no atoms")
    if _n_ca(structure) < 1:
        raise ValueError(f"{structure.pdb_id}: no C-alpha atoms")


def write_structure_summary_csv(
    reports: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    ensure_dirs()
    path = path or (METADATA_DIR / "structure_summary.csv")
    fieldnames = [
        "pdb_id",
        "n_models",
        "n_chains",
        "chain_ids",
        "n_residues",
        "n_polymer_residues",
        "n_water_residues",
        "n_atoms",
        "n_ca",
        "experimental_method",
        "resolution",
        "title",
        "space_group",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in reports:
            writer.writerow(row)
    return path


def summarize_raw_dir(
    raw_dir: Path | None = None,
    *,
    skip_errors: bool = True,
) -> tuple[list[StructureData], Path]:
    raw_dir = raw_dir or RAW_DIR
    structures: list[StructureData] = []
    reports: list[dict[str, Any]] = []
    files = discover_mmcif_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No mmCIF files in {raw_dir}")
    for path in files:
        try:
            print(f"parse {path.name}", flush=True)
            structure = load_structure(path)
            validate_not_empty(structure)
        except Exception as exc:
            if not skip_errors:
                raise
            print(f"skip {path.name}: {exc}")
            continue
        structures.append(structure)
        reports.append(structure_report(structure))
    if not structures:
        raise ValueError(f"No valid mmCIF structures in {raw_dir}")
    csv_path = write_structure_summary_csv(reports)
    return structures, csv_path
