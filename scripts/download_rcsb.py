"""Download mmCIF.gz files from RCSB into raw/.

Existing files are kept and counted toward n_structures.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io.discover import discover_mmcif_files
from src.io.rcsb import download_cif_gz, search_pdb_ids_paginated
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
    min_res = int(cfg.get("search_min_polymer_residues", cfg.get("min_polymer_residues", 50)))
    max_res = int(cfg.get("search_max_polymer_residues", cfg.get("max_polymer_residues", 350)))
    workers = int(cfg.get("download_workers", 8))

    existing = _existing_ids(RAW_DIR)
    have = set(existing)
    print(f"raw dir: {RAW_DIR}", flush=True)
    print(f"existing mmCIF: {len(have)}", flush=True)
    print(f"target: {n_target}  workers: {workers}", flush=True)

    need = max(n_target - len(have), 0)
    search_n = max(n_target + 2000, need * 2 + len(have), 1000)
    print(f"searching RCSB for up to {search_n} candidate IDs...", flush=True)
    ids = search_pdb_ids_paginated(
        n_wanted=search_n,
        resolution_max=res_max,
        min_polymer_residues=min_res,
        max_polymer_residues=max_res,
    )
    print(f"search returned {len(ids)} IDs", flush=True)
    if not ids and len(have) < n_target:
        raise RuntimeError("RCSB search returned no IDs")

    todo = [pdb_id for pdb_id in ids if pdb_id not in have]
    extra = max(n_target - len(have), 0) + 400
    todo = todo[:extra]
    lock = threading.Lock()
    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    def fetch(pdb_id: str) -> str:
        dest = RAW_DIR / f"{pdb_id}.cif.gz"
        with lock:
            if len(have) >= n_target:
                return "full"
            if pdb_id in have or dest.exists():
                have.add(pdb_id)
                skipped.append(pdb_id)
                return "exists"
        try:
            download_cif_gz(pdb_id, dest, pause_s=0.02)
        except OSError as exc:
            dest.unlink(missing_ok=True)
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
            with lock:
                failed.append(pdb_id)
            print(f"  FAIL {pdb_id}: {exc}", flush=True)
            return "fail"
        with lock:
            have.add(pdb_id)
            downloaded.append(pdb_id)
            n = len(have)
            if len(downloaded) == 1 or n % 50 == 0 or n >= n_target:
                print(f"  got {pdb_id}  ({n}/{n_target})", flush=True)
        return "ok"

    if need and todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = []
            for pdb_id in todo:
                with lock:
                    if len(have) >= n_target:
                        break
                futs.append(pool.submit(fetch, pdb_id))
            for fut in as_completed(futs):
                fut.result()
                with lock:
                    if len(have) >= n_target:
                        break

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
