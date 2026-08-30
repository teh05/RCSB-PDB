"""mmCIF / PDBx structure loading via gemmi.

Do not parse mmCIF with regex. All coordinate and hierarchy extraction
goes through gemmi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gemmi


@dataclass(frozen=True)
class AtomRecord:
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    serial: int
    altloc: str
    occupancy: float
    b_factor: float
    is_hetatm: bool


@dataclass(frozen=True)
class ResidueRecord:
    name: str
    seq_id: int
    insertion_code: str
    residue_uid: str
    atoms: tuple[AtomRecord, ...]


@dataclass(frozen=True)
class ChainRecord:
    chain_id: str
    residues: tuple[ResidueRecord, ...]


@dataclass(frozen=True)
class ModelRecord:
    model_num: int
    chains: tuple[ChainRecord, ...]


@dataclass
class StructureData:
    """Organized protein structure extracted from mmCIF."""

    pdb_id: str
    source_path: str
    models: tuple[ModelRecord, ...]
    experimental_method: str | None = None
    resolution: float | None = None
    title: str | None = None
    space_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def chain_ids(self) -> list[str]:
        if not self.models:
            return []
        return [c.chain_id for c in self.models[0].chains]

    @property
    def n_chains(self) -> int:
        return len(self.chain_ids)

    @property
    def n_residues(self) -> int:
        if not self.models:
            return 0
        return sum(len(c.residues) for c in self.models[0].chains)

    @property
    def n_atoms(self) -> int:
        if not self.models:
            return 0
        return sum(
            len(r.atoms)
            for c in self.models[0].chains
            for r in c.residues
        )

    def summary(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "n_models": self.n_models,
            "n_chains": self.n_chains,
            "chain_ids": self.chain_ids,
            "n_residues": self.n_residues,
            "n_atoms": self.n_atoms,
            "experimental_method": self.experimental_method,
            "resolution": self.resolution,
            "title": self.title,
            "space_group": self.space_group,
            "source_path": self.source_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_cif_block(path: Path) -> gemmi.cif.Block:
    if path.suffixes[-2:] == [".cif", ".gz"] or str(path).endswith(".cif.gz"):
        doc = gemmi.cif.read(str(path))
    else:
        doc = gemmi.cif.read(str(path))
    if not doc:
        raise ValueError(f"No CIF blocks in {path}")
    return doc.sole_block()


def _meta_from_cif(path: Path) -> dict[str, Any]:
    block = _read_cif_block(path)
    meta: dict[str, Any] = {}

    def get(tag: str) -> str | None:
        val = block.find_value(tag)
        if val is None:
            return None
        return gemmi.cif.as_string(val)

    meta["entry_id"] = get("_entry.id")
    meta["title"] = get("_struct.title")
    meta["space_group"] = get("_symmetry.space_group_name_H-M")
    meta["experimental_method"] = get("_exptl.method")
    if meta["experimental_method"] is None:
        # multi-row loop fallback
        methods = []
        for row in block.find(["_exptl.method"]):
            methods.append(gemmi.cif.as_string(row[0]))
        if methods:
            meta["experimental_method"] = "; ".join(methods)

    res_raw = get("_refine.ls_d_res_high")
    if res_raw not in (None, "?", "."):
        try:
            meta["resolution"] = float(res_raw)
        except ValueError:
            meta["resolution"] = None
    else:
        meta["resolution"] = None
    return meta


def _residue_uid(chain_id: str, seq_id: int, icode: str, name: str) -> str:
    ic = icode.strip() if icode and icode not in {".", "?", " "} else ""
    return f"{chain_id}:{seq_id}{ic}:{name}"


def load_structure(
    path: str | Path,
    *,
    include_water: bool = True,
    include_hetero: bool = True,
) -> StructureData:
    """Load a compressed or plain mmCIF into an organized hierarchy.

    Parameters
    ----------
    path:
        Path to `.cif` or `.cif.gz`.
    include_water:
        Keep water residues (HOH/WAT/DOD) when True.
    include_hetero:
        Keep non-polymer HETATM residues when True. Polymer ATOM residues
        are always kept.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()

    meta = _meta_from_cif(path)
    pdb_id = (meta.get("entry_id") or structure.name or path.name.split(".")[0]).upper()

    models: list[ModelRecord] = []
    for model in structure:
        chains: list[ChainRecord] = []
        for chain in model:
            residues: list[ResidueRecord] = []
            for residue in chain:
                is_water = residue.name in {"HOH", "WAT", "DOD"}
                # gemmi Residue has het_flag: 'A' atom, 'H' het, ' ' unknown
                is_het = residue.het_flag == "H" or residue.is_water()
                if is_water and not include_water:
                    continue
                if is_het and not is_water and not include_hetero:
                    continue

                icode = residue.seqid.icode.strip() if residue.seqid.icode else ""
                seq_id = int(residue.seqid.num)
                atoms: list[AtomRecord] = []
                for atom in residue:
                    element = atom.element.name if atom.element else ""
                    atoms.append(
                        AtomRecord(
                            atom_name=atom.name,
                            element=element,
                            x=float(atom.pos.x),
                            y=float(atom.pos.y),
                            z=float(atom.pos.z),
                            serial=int(atom.serial),
                            altloc=atom.altloc if atom.altloc and atom.altloc != "\0" else "",
                            occupancy=float(atom.occ),
                            b_factor=float(atom.b_iso),
                            is_hetatm=bool(is_het),
                        )
                    )
                if not atoms:
                    continue
                residues.append(
                    ResidueRecord(
                        name=residue.name,
                        seq_id=seq_id,
                        insertion_code=icode,
                        residue_uid=_residue_uid(chain.name, seq_id, icode, residue.name),
                        atoms=tuple(atoms),
                    )
                )
            if residues:
                chains.append(ChainRecord(chain_id=chain.name, residues=tuple(residues)))
        models.append(ModelRecord(model_num=int(model.num), chains=tuple(chains)))

    data = StructureData(
        pdb_id=pdb_id,
        source_path=str(path.resolve()),
        models=tuple(models),
        experimental_method=meta.get("experimental_method"),
        resolution=meta.get("resolution"),
        title=meta.get("title"),
        space_group=meta.get("space_group"),
        metadata=meta,
    )
    if data.n_atoms == 0:
        raise ValueError(f"Parsed structure is empty: {path}")
    return data
