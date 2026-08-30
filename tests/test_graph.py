from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.build import build_residue_graph, sequential_edges
from src.io.mmcif import load_structure
from src.preprocessing.coords import extract_all_atom, extract_ca

RAW = ROOT / "raw"


@pytest.mark.skipif(not (RAW / "1UBQ.cif.gz").exists(), reason="missing mmCIF")
def test_ca_and_allatom_shapes():
    s = load_structure(RAW / "1UBQ.cif.gz")
    aa = extract_all_atom(s)
    ca = extract_ca(s)
    assert aa.coords.shape == (aa.n, 3)
    assert ca.coords.shape == (76, 3)
    assert aa.n > ca.n
    assert np.isfinite(aa.coords).all()
    assert np.isfinite(ca.coords).all()


@pytest.mark.skipif(not (RAW / "4HHB.cif.gz").exists(), reason="missing mmCIF")
def test_4hhb_ca_per_chain():
    s = load_structure(RAW / "4HHB.cif.gz")
    ca = extract_ca(s)
    assert ca.n == 574
    counts = {cid: int((ca.chain_ids == cid).sum()) for cid in ("A", "B", "C", "D")}
    assert counts["A"] == 141
    assert counts["B"] == 146
    assert counts["C"] == 141
    assert counts["D"] == 146


@pytest.mark.skipif(not (RAW / "1UBQ.cif.gz").exists(), reason="missing mmCIF")
def test_graph_sequential_and_radius():
    s = load_structure(RAW / "1UBQ.cif.gz")
    ca = extract_ca(s)
    g = build_residue_graph(ca, radius_cutoff=8.0)
    assert g.n_nodes == 76
    seq = sequential_edges(ca.chain_ids)
    # 75 undirected backbone steps -> 150 directed
    assert seq.shape[1] == 150
    assert g.n_edges > seq.shape[1]
    assert g.node_features.shape[0] == 76
    assert np.isfinite(g.edge_dist).all()
    # sequential edges stay inside one chain
    src, dst = g.edge_index
    same = ca.chain_ids[src] == ca.chain_ids[dst]
    seq_mask = g.edge_type == 0
    assert bool(np.all(same[seq_mask]))


@pytest.mark.skipif(not (RAW / "1UBQ.cif.gz").exists(), reason="missing mmCIF")
def test_collate_two_graphs():
    from src.graph.collate import collate_graphs

    s = load_structure(RAW / "1UBQ.cif.gz")
    g = build_residue_graph(extract_ca(s), radius_cutoff=8.0)
    batch = collate_graphs([g, g])
    assert batch["x"].shape[0] == 152
    assert int(batch["edge_index"].max()) < 152
    assert batch["batch"][-1] == 1
