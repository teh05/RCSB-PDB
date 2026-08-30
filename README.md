# RCSB PDB → Graph → E(3)-GNN (PyTorch vs JAX)

Development pipeline for converting RCSB mmCIF structures into protein graphs,
then comparing PyTorch and JAX/Flax geometric models on CUDA.

Default dataset: **1000** X-ray protein-only entries (resolution ≤ 2.5 Å,
50–350 polymer residues) downloaded into `raw/`. This is still an engineering
set, not a published generalization benchmark.

## Hardware (this machine)

- Windows, NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, sm_120)
- Python 3.14
- PyTorch CUDA 12.8 nightly
- JAX GPU via `winjax` (native Windows CUDA 13 PJRT). Official `jax[cuda13]` is Linux-only.
- Flax `0.12.9` installed with `--no-deps` to keep winjax's `jax==0.11.0` pin.

PyTorch and JAX **must run in separate processes** on Windows (different cuDNN families).

## Run

```bash
python scripts/check_environment.py
python scripts/download_rcsb.py configs/egnn.yaml
python scripts/preprocess.py
python scripts/build_graph.py configs/egnn.yaml
python scripts/visualize.py configs/egnn.yaml
python scripts/train_pytorch.py configs/egnn.yaml
python scripts/train_jax.py configs/egnn.yaml
python scripts/benchmark_cuda.py
```

Existing mmCIF files in `raw/` are kept and counted toward `n_structures`.
Training uses mini-batches and an 80/20 train/val split when N ≥ 5.

`benchmark_cuda.py` does not import either framework. It subprocesses the E(3)
checks and both EGNN trainers. Pass `--from-metrics` to rebuild the JSON
without retraining.

## Diagnostic task

Predict **C-alpha pairwise distances on graph edges**. Labels come from 3D
coordinates. Engineering check (forward/backward/GPU), not a biological function
model.

- Phase 7: `BaselineGCN` — residue type + graph topology (no coordinates in the net)
- Phase 8: `EGNN` — SE(3) messages from `||x_i - x_j||^2`; coordinates update
  equivariantly; predicted distances are invariant
- Phase 9: same EGNN task on the 1000-entry RCSB set (mini-batch + val split)

## Representations

- All-atom: every unique atom name per residue, `coords.shape = (N_atom, 3)`
- C-alpha: one `CA` per residue, `coords.shape = (N_res, 3)` (~8× fewer nodes)

Graphs are residue/C-alpha. Sequential edges stay inside a chain; radius cutoff is set in config.

`scripts/visualize.py` writes a **sample** of matplotlib plots (`plot_max`) and,
if PyMOL is installed, cartoons (`cartoon_max`) from `raw/*.cif.gz`.
It does not cartoon all 1000 structures.

```text
pymol raw/1UBQ.cif.gz
ChimeraX raw/4HHB.cif.gz
```
