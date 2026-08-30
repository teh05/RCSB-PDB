from src.graph.build import ProteinGraph
from src.graph.collate import collate_graphs, pad_collated
from src.io.rcsb import protein_xray_query
import numpy as np


def _toy(n: int, e: int, pdb_id: str = "TOY") -> ProteinGraph:
    coords = np.zeros((n, 3), dtype=np.float32)
    x = np.zeros((n, 21), dtype=np.float32)
    ei = np.zeros((2, e), dtype=np.int64)
    if e:
        ei[0] = np.arange(e) % n
        ei[1] = (np.arange(e) + 1) % n
    dist = np.ones((e,), dtype=np.float32)
    return ProteinGraph(
        pdb_id=pdb_id,
        representation="ca",
        coords=coords,
        node_features=x,
        residue_names=np.array(["ALA"] * n),
        chain_ids=np.array(["A"] * n),
        seq_ids=np.arange(n),
        edge_index=ei,
        edge_type=np.zeros((e,), dtype=np.int64),
        edge_dist=dist,
        radius_cutoff=8.0,
    )


def test_protein_xray_query_shape():
    q = protein_xray_query(n=1000, resolution_max=2.5, min_polymer_residues=50, max_polymer_residues=350)
    assert q["return_type"] == "entry"
    assert q["request_options"]["paginate"]["rows"] == 1000
    attrs = [n["parameters"]["attribute"] for n in q["query"]["nodes"]]
    assert "exptl.method" in attrs
    assert "rcsb_entry_info.deposited_polymer_monomer_count" in attrs


def test_pad_collated_masks():
    g = _toy(4, 6)
    data = collate_graphs([g])
    padded = pad_collated(data, max_nodes=8, max_edges=10)
    assert padded["x"].shape == (8, 21)
    assert padded["edge_index"].shape == (2, 10)
    assert float(padded["node_mask"].sum()) == 4
    assert float(padded["edge_mask"].sum()) == 6
    dummy = 4
    assert int(padded["edge_index"][0, 6]) == dummy
