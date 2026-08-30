# RCSB PDB → Graph → E(3)-GNN (PyTorch vs JAX)

Development pipeline for converting RCSB mmCIF structures into protein graphs,
then comparing PyTorch and JAX/Flax geometric models on CUDA.

The three local PDBs (`1UBQ`, `1LYZ`, `4HHB`) are a **debug dataset**, not a
generalization benchmark.

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
python scripts/preprocess.py
python scripts/build_graph.py
python scripts/visualize.py
python scripts/train_pytorch.py
python scripts/train_pytorch.py configs/egnn.yaml
python scripts/train_jax.py configs/egnn.yaml
python scripts/benchmark_cuda.py
```

`benchmark_cuda.py` does not import either framework. It subprocesses the E(3)
checks and both EGNN trainers, then writes `results/metrics/phase8_benchmark.json`.
Pass `--from-metrics` to rebuild that file from existing JSON without retraining.

## Diagnostic task

Predict **C-alpha pairwise distances on graph edges**. Labels come from 3D
coordinates. Engineering check (forward/backward/GPU), not a biological function
model and not a generalization result.

- Phase 7: `BaselineGCN` — residue type + graph topology (no coordinates in the net)
- Phase 8: `EGNN` — SE(3) messages from `||x_i - x_j||^2`; coordinates update
  equivariantly; predicted distances are invariant

## Representations

- All-atom: every unique atom name per residue, `coords.shape = (N_atom, 3)`
- C-alpha: one `CA` per residue, `coords.shape = (N_res, 3)` (~8× fewer nodes)

Graphs are residue/C-alpha. Sequential edges stay inside a chain; radius cutoff is set in `configs/baseline.yaml`.

Raw mmCIF files stay in `raw/` and must not be re-downloaded.

`scripts/visualize.py` writes matplotlib diagnostics and, if PyMOL is on PATH,
headless cartoons (`results/plots/<PDB>_cartoon.png`) from `raw/*.cif.gz`.
Cartoons are skipped when PyMOL is missing — it is not a required dependency.

```text
pymol raw/1UBQ.cif.gz
ChimeraX raw/4HHB.cif.gz
```
