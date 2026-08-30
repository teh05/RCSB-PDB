"""All-atom and C-alpha coordinate extraction.

Raw coordinates are always retained. C-alpha is a derived representation,
not a replacement for the all-atom set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.io.mmcif import StructureData

AA20 = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_TO_INDEX = {aa: i for i, aa in enumerate(AA20)}
UNK_RESIDUE_INDEX = len(AA20)


def residue_index(name: str) -> int:
    return RESIDUE_TO_INDEX.get(name, UNK_RESIDUE_INDEX)


def _pick_ca(atoms) -> object | None:
    cas = [a for a in atoms if a.atom_name == "CA"]
    if not cas:
        return None
    cas = sorted(cas, key=lambda a: (a.altloc not in ("", "A"), a.altloc, a.serial))
    return cas[0]


@dataclass
class CoordinateSet:
    pdb_id: str
    representation: str  # "all_atom" | "ca"
    coords: np.ndarray  # (N, 3)
    elements: np.ndarray
    atom_names: np.ndarray
    residue_names: np.ndarray
    residue_indices: np.ndarray
    chain_ids: np.ndarray
    seq_ids: np.ndarray
    insertion_codes: np.ndarray
    is_hetatm: np.ndarray

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])


def extract_all_atom(structure: StructureData, model_idx: int = 0) -> CoordinateSet:
    coords: list[list[float]] = []
    elements: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    residue_indices: list[int] = []
    chain_ids: list[str] = []
    seq_ids: list[int] = []
    insertion_codes: list[str] = []
    is_hetatm: list[bool] = []

    model = structure.models[model_idx]
    for chain in model.chains:
        for res in chain.residues:
            seen: set[str] = set()
            for a in sorted(res.atoms, key=lambda x: (x.atom_name, x.altloc not in ("", "A"), x.serial)):
                if a.atom_name in seen:
                    continue
                seen.add(a.atom_name)
                coords.append([a.x, a.y, a.z])
                elements.append(a.element)
                atom_names.append(a.atom_name)
                residue_names.append(res.name)
                residue_indices.append(residue_index(res.name))
                chain_ids.append(chain.chain_id)
                seq_ids.append(res.seq_id)
                insertion_codes.append(res.insertion_code)
                is_hetatm.append(bool(a.is_hetatm))

    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{structure.pdb_id}: all-atom extraction is empty")
    return CoordinateSet(
        pdb_id=structure.pdb_id,
        representation="all_atom",
        coords=arr,
        elements=np.asarray(elements),
        atom_names=np.asarray(atom_names),
        residue_names=np.asarray(residue_names),
        residue_indices=np.asarray(residue_indices, dtype=np.int64),
        chain_ids=np.asarray(chain_ids),
        seq_ids=np.asarray(seq_ids, dtype=np.int64),
        insertion_codes=np.asarray(insertion_codes),
        is_hetatm=np.asarray(is_hetatm, dtype=bool),
    )


def extract_ca(structure: StructureData, model_idx: int = 0) -> CoordinateSet:
    coords: list[list[float]] = []
    elements: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    residue_indices: list[int] = []
    chain_ids: list[str] = []
    seq_ids: list[int] = []
    insertion_codes: list[str] = []
    is_hetatm: list[bool] = []

    model = structure.models[model_idx]
    for chain in model.chains:
        for res in chain.residues:
            atom = _pick_ca(res.atoms)
            if atom is None:
                continue
            coords.append([atom.x, atom.y, atom.z])
            elements.append(atom.element)
            atom_names.append(atom.atom_name)
            residue_names.append(res.name)
            residue_indices.append(residue_index(res.name))
            chain_ids.append(chain.chain_id)
            seq_ids.append(res.seq_id)
            insertion_codes.append(res.insertion_code)
            is_hetatm.append(bool(atom.is_hetatm))

    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{structure.pdb_id}: C-alpha extraction is empty")
    return CoordinateSet(
        pdb_id=structure.pdb_id,
        representation="ca",
        coords=arr,
        elements=np.asarray(elements),
        atom_names=np.asarray(atom_names),
        residue_names=np.asarray(residue_names),
        residue_indices=np.asarray(residue_indices, dtype=np.int64),
        chain_ids=np.asarray(chain_ids),
        seq_ids=np.asarray(seq_ids, dtype=np.int64),
        insertion_codes=np.asarray(insertion_codes),
        is_hetatm=np.asarray(is_hetatm, dtype=bool),
    )


def save_coordinate_set(coords: CoordinateSet, path) -> None:
    np.savez_compressed(
        path,
        pdb_id=coords.pdb_id,
        representation=coords.representation,
        coords=coords.coords,
        elements=coords.elements,
        atom_names=coords.atom_names,
        residue_names=coords.residue_names,
        residue_indices=coords.residue_indices,
        chain_ids=coords.chain_ids,
        seq_ids=coords.seq_ids,
        insertion_codes=coords.insertion_codes,
        is_hetatm=coords.is_hetatm,
    )


def load_coordinate_set(path) -> CoordinateSet:
    data = np.load(path, allow_pickle=True)
    return CoordinateSet(
        pdb_id=str(data["pdb_id"]),
        representation=str(data["representation"]),
        coords=data["coords"],
        elements=data["elements"],
        atom_names=data["atom_names"],
        residue_names=data["residue_names"],
        residue_indices=data["residue_indices"],
        chain_ids=data["chain_ids"],
        seq_ids=data["seq_ids"],
        insertion_codes=data["insertion_codes"],
        is_hetatm=data["is_hetatm"],
    )


def representation_cost_note(all_atom: CoordinateSet, ca: CoordinateSet) -> str:
    ratio = all_atom.n / max(ca.n, 1)
    return (
        f"{all_atom.pdb_id}: all-atom N={all_atom.n} vs C-alpha N={ca.n} "
        f"(~{ratio:.1f}x nodes). Radius-graph edges scale ~N^2 locally, so "
        "all-atom graphs are substantially more expensive than C-alpha."
    )
