"""RCSB Search + file download (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
USER_AGENT = "RCSB-PDB-GNN-pipeline/0.1 (research; local dataset build)"


def protein_xray_query(
    *,
    n: int,
    resolution_max: float,
    min_polymer_residues: int,
    max_polymer_residues: int,
) -> dict[str, Any]:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution_max,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "greater_or_equal",
                        "value": min_polymer_residues,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "less_or_equal",
                        "value": max_polymer_residues,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": int(n)},
            "sort": [
                {
                    "sort_by": "rcsb_accession_info.initial_release_date",
                    "direction": "desc",
                }
            ],
            "results_content_type": ["experimental"],
        },
    }


def _request(url: str, data: bytes | None = None, timeout_s: float = 60.0) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json,*/*"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def search_pdb_ids(query: dict[str, Any], *, timeout_s: float = 120.0) -> list[str]:
    raw = _request(SEARCH_URL, data=json.dumps(query).encode("utf-8"), timeout_s=timeout_s)
    payload = json.loads(raw.decode("utf-8"))
    rows = payload.get("result_set") or []
    ids: list[str] = []
    for row in rows:
        ident = row.get("identifier") if isinstance(row, dict) else row
        if ident:
            ids.append(str(ident).upper())
    return ids


def download_cif_gz(
    pdb_id: str,
    dest: Path,
    *,
    timeout_s: float = 60.0,
    retries: int = 3,
    pause_s: float = 0.12,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read()
            if len(body) < 64:
                raise OSError(f"{pdb_id}: download too small ({len(body)} bytes)")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(body)
            tmp.replace(dest)
            time.sleep(pause_s)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(pause_s * attempt)
    raise OSError(f"Failed to download {pdb_id}: {last_err}")
