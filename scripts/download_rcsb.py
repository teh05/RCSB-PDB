"""Download mmCIF.gz files from RCSB into raw/.

Default: 1000 X-ray protein-only entries, resolution <= 2.5 A,
polymer residue count 50-350. Existing files are kept and counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io.discover import discover_mmcif_files
from src.io.rcsb import download_cif_gz, protein_xray_query, search_pdb_ids
from src.paths import CONFIGS_DIR, METADATA_DIR, RAW_DIR, ensure_dirs


def _existing_ids(raw_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in discover_mmcif_files(raw_dir):
        name = p.name
        if name.lower().endswith(".cif.gz"):
            ids.add(name[: -len(".cif.gz")].upper())
        elif name.lower().endswith(".cif"):
            ids.add(name[: -len(".cif")].upper())
    return ids


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "egnn.yaml"
    cfg = load_config(cfg_path)
    n_target = int(cfg.get("n_structures", 1000))
    res_max = float(cfg.get("resolution_max", 2.5))
    min_res = int(cfg.get("min_polymer_residues", 50))
    max_res = int(cfg.get("max_polymer_residues", 350))

    existing = _existing_ids(RAW_DIR)
    have = set(existing)
    print(f"raw dir: {RAW_DIR}", flush=True)
    print(f"existing mmCIF: {len(have)}", flush=True)
    print(f"target: {n_target}", flush=True)

    search_n = max(n_target + 400, n_target)
    query = protein_xray_query(
        n=search_n,
        resolution_max=res_max,
        min_polymer_residues=min_res,
        max_polymer_residues=max_res,
    )
    print(f"searching RCSB for {search_n} candidate IDs...", flush=True)
    ids = search_pdb_ids(query)
    print(f"search returned {len(ids)} IDs", flush=True)
    if not ids:
        raise RuntimeError("RCSB search returned no IDs")

    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for pdb_id in ids:
        if len(have) >= n_target:
            break
        dest = RAW_DIR / f"{pdb_id}.cif.gz"
        if pdb_id in have or dest.exists():
            skipped.append(pdb_id)
            have.add(pdb_id)
            continue
        try:
            download_cif_gz(pdb_id, dest)
            have.add(pdb_id)
            downloaded.append(pdb_id)
            print(f"  got {pdb_id}  ({len(have)}/{n_target})", flush=True)
        except OSError as exc:
            failed.append(pdb_id)
            print(f"  FAIL {pdb_id}: {exc}")
            dest.unlink(missing_ok=True)

    final_ids = sorted(_existing_ids(RAW_DIR))
    manifest = {
        "n_target": n_target,
        "n_on_disk": len(final_ids),
        "resolution_max": res_max,
        "min_polymer_residues": min_res,
        "max_polymer_residues": max_res,
        "method": "X-RAY DIFFRACTION",
        "polymer_entity_types": "Protein (only)",
        "n_downloaded_this_run": len(downloaded),
        "n_skipped_existing": len(skipped),
        "n_failed": len(failed),
        "failed_ids": failed,
        "pdb_ids": final_ids,
        "note": "engineering dataset from RCSB; not a curated benchmark split",
    }
    out = METADATA_DIR / "rcsb_download.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "pdb_ids"}, indent=2))
    print(f"wrote {out}")
    if len(final_ids) < n_target:
        print(f"WARNING: only {len(final_ids)} files in raw/ (target {n_target})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
