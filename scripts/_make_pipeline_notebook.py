"""Generate notebooks/pipeline.ipynb. Run from repo root."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "pipeline.ipynb"


def cell(kind: str, source: str) -> dict:
    lines = source.strip("\n")
    if lines:
        lines = lines + "\n"
    src = [ln + "\n" for ln in lines.split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n") + "\n"
    base = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def main() -> None:
    cells = [
        cell(
            "markdown",
            """# RCSB PDB → graf protein → perbandingan GNN

Dataset target: **5000** mmCIF X-ray. Tugas diagnostik: regresi jarak C-alpha per edge (MSE). Bukan prediksi fungsi biologi.

| Fase | Isi |
|------|-----|
| 7 | BaselineGCN |
| 8 | EGNN debug |
| 9 | EGNN 1000 (arsip) |
| **10** | Banding **PairMLP, GCN, GAT, SchNet, EGNN** (PyTorch) + EGNN JAX |
| **11** | Tuning EGNN (hidden × layers) |

Split 80/10/10 (train/val/test). Di Windows jangan import torch dan JAX di kernel yang sama.

Tombol di sel 1 menjalankan subprocess. Atau set flag `RETRAIN_* = True` lalu jalankan sel terkait.
""",
        ),
        cell(
            "markdown",
            """## 1. Setup + tombol train ulang
""",
        ),
        cell(
            "code",
            """from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image, Markdown, display

%matplotlib inline
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.build import EDGE_RADIUS, load_graph
from src.io.discover import discover_mmcif_files
from src.io.mmcif import load_structure
from src.paths import (
    CONFIGS_DIR,
    GRAPHS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    RAW_DIR,
    STRUCTURES_DIR,
    ensure_dirs,
)
from src.preprocessing.coords import load_coordinate_set

ensure_dirs()
CFG = load_config(CONFIGS_DIR / "egnn.yaml")
CUTOFF = float(CFG["radius_cutoff"])
PY = sys.executable
EXAMPLE_ID = "1UBQ"
CFG_PATH = str(CONFIGS_DIR / "egnn.yaml")

RETRAIN_COMPARE = False
RETRAIN_JAX = False
RETRAIN_TUNE = False
FORCE_DOWNLOAD = False
FORCE_PREPROCESS = False
FORCE_VIZ = False


def run_script(script: str, *args: str) -> None:
    cmd = [PY, "-u", str(ROOT / "scripts" / script), *args]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} exited {proc.returncode}")


def metrics_ready(path: Path, n_expected: int) -> bool:
    if not path.is_file():
        return False
    m = json.loads(path.read_text(encoding="utf-8"))
    return int(m.get("n_proteins", 0)) >= n_expected and m.get("final_test_mse") is not None


print("ROOT:", ROOT)
print("n_structures:", CFG.get("n_structures"), "val/test:", CFG.get("val_fraction"), CFG.get("test_fraction"))
print("compare_models:", CFG.get("compare_models"))

try:
    import ipywidgets as w
except ImportError:
    print("ipywidgets tidak ada — pakai flag RETRAIN_* di atas, lalu jalankan sel fase 10/11.")
else:
    log = w.Output()

    def _btn(label, script, extra=()):
        b = w.Button(description=label, layout=w.Layout(width="220px"))

        def on_click(_):
            with log:
                log.clear_output()
                run_script(script, *extra)

        b.on_click(on_click)
        return b

    display(Markdown("**Train ulang (subprocess, GPU)** — klik salah satu:"))
    display(w.HBox([
        _btn("Bandingkan 5 model", "compare_models.py", (CFG_PATH,)),
        _btn("Train EGNN JAX", "train_jax.py", (CFG_PATH,)),
        _btn("Tune EGNN", "tune_egnn.py", (CFG_PATH,)),
        _btn("Plot eksperimen", "plot_experiment.py", ("--force",)),
    ]))
    display(w.HBox([
        _btn("Download 5000", "download_rcsb.py", (CFG_PATH,)),
        _btn("Preprocess", "preprocess.py"),
        _btn("Build graf", "build_graph.py", (CFG_PATH,)),
        _btn("Visualisasi protein", "visualize.py", (CFG_PATH,)),
    ]))
    display(log)
""",
        ),
        cell(
            "markdown",
            """## 2. Dataset `raw/` → 5000

File yang sudah ada tidak dihapus. Download hanya menambah sampai target.
""",
        ),
        cell(
            "code",
            """raw_files = discover_mmcif_files(RAW_DIR)
print(f"mmCIF di raw/: {len(raw_files)}")
n_target = int(CFG.get("n_structures", 5000))
if FORCE_DOWNLOAD or len(raw_files) < n_target:
    run_script("download_rcsb.py", CFG_PATH)
    raw_files = discover_mmcif_files(RAW_DIR)
print(f"setelah cek: {len(raw_files)} (target {n_target})")
""",
        ),
        cell(
            "markdown",
            """## 3. Parse contoh + preprocess
""",
        ),
        cell(
            "code",
            """cif_path = RAW_DIR / f"{EXAMPLE_ID}.cif.gz"
if not cif_path.exists():
    cif_path = next(p for p in raw_files if EXAMPLE_ID.lower() in p.name.lower())
structure = load_structure(cif_path)
for k, v in structure.summary().items():
    print(f"  {k:22s} {v}")

n_ca = len(list(STRUCTURES_DIR.glob("*_ca.npz")))
print(f"file C-alpha: {n_ca}   mmCIF: {len(raw_files)}")
if FORCE_PREPROCESS or n_ca < len(raw_files):
    run_script("preprocess.py")
    n_ca = len(list(STRUCTURES_DIR.glob("*_ca.npz")))
print("C-alpha:", n_ca)

ca = load_coordinate_set(STRUCTURES_DIR / f"{EXAMPLE_ID}_ca.npz")
aa = load_coordinate_set(STRUCTURES_DIR / f"{EXAMPLE_ID}_allatom.npz")
print(f"{EXAMPLE_ID} all-atom N={aa.n}  C-alpha N={ca.n}")
""",
        ),
        cell(
            "markdown",
            """## 4. Graf residu
""",
        ),
        cell(
            "code",
            """n_graphs = len(list(GRAPHS_DIR.glob(f"*_ca_r{CUTOFF:g}.npz")))
print(f"graf: {n_graphs}")
if n_graphs < n_ca:
    run_script("build_graph.py", CFG_PATH)
    n_graphs = len(list(GRAPHS_DIR.glob(f"*_ca_r{CUTOFF:g}.npz")))
print("graf:", n_graphs)

g = load_graph(GRAPHS_DIR / f"{EXAMPLE_ID}_ca_r{CUTOFF:g}.npz")
print(f"{g.pdb_id}: V={g.n_nodes}  E={g.n_edges}  cutoff={g.radius_cutoff} Å")
""",
        ),
        cell(
            "markdown",
            """## 5. Ringkasan dataset
""",
        ),
        cell(
            "code",
            """csv_path = ROOT / "metadata" / "structure_summary.csv"
rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
print(f"baris metadata: {len(rows)}")
n_ca_arr = np.array([float(r["n_ca"]) for r in rows if r.get("n_ca")])
resol = np.array([float(r["resolution"]) for r in rows if r.get("resolution") not in ("", "None")])
print(f"N_CA mean={n_ca_arr.mean():.0f} min={n_ca_arr.min():.0f} max={n_ca_arr.max():.0f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].hist(n_ca_arr, bins=40, color="#1f4e79", alpha=0.85)
axes[0].set_title("Jumlah C-alpha")
axes[1].hist(resol, bins=30, color="#3d6b4f", alpha=0.85)
axes[1].set_title("Resolusi (Å)")
fig.tight_layout()
plt.show()
""",
        ),
        cell(
            "markdown",
            r"""## 6. Model pembanding (fase 10)

Semua prediksi jarak edge dari \([h_i\|h_j]\). Perbedaan hanya cara menghitung \(h\):

| Model | Koordinat | Inti |
|--------|-----------|------|
| PairMLP | tidak | MLP per residu, tanpa pesan graf |
| BaselineGCN | tidak | mean-aggregate tetangga |
| GAT | tidak | perhatian \(\alpha_{ij}\) |
| SchNet | jarak RBF | filter kontinu \(\|x_i-x_j\|\) |
| EGNN | E(3) | pesan \(\|x_i-x_j\|^2\), update koordinat |

EGNN (Satorras 2021): \(m_{ij}=\phi_e(h_i,h_j,\|x_i-x_j\|^2)\).
""",
        ),
        cell(
            "markdown",
            """## 7. Fase 10 — train 5 model PyTorch

Sel ini **tidak** meng-import torch. Klik tombol di atas, atau set `RETRAIN_COMPARE = True`.
""",
        ),
        cell(
            "code",
            """cmp_path = METRICS_DIR / "phase10_compare.json"
need = RETRAIN_COMPARE or not cmp_path.is_file()
if not need:
    m0 = json.loads(cmp_path.read_text(encoding="utf-8"))
    n_run = (m0.get("models") or [{}])[0].get("n_proteins", 0)
    need = int(n_run) < n_graphs
if need:
    run_script("compare_models.py", CFG_PATH)
cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
print("best_test:", cmp.get("best_test"))

labels, test_s, val_s, times, npar = [], [], [], [], []
print(f"{'model':12s} {'params':>8s} {'train':>8s} {'val':>8s} {'test':>8s} {'sec':>8s}")
for row in cmp["models"]:
    labels.append(str(row["model"]))
    tr, va, te = row.get("final_train_mse"), row.get("final_val_loss"), row.get("final_test_mse")
    test_s.append(te)
    val_s.append(va)
    times.append(row.get("training_time_s") or 0)
    npar.append(row.get("n_params") or 0)
    print(f"{row['model']:12s} {npar[-1]:8d} {tr:8.4f} {va:8.4f} {te:8.4f} {times[-1]:8.1f}")

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].bar(x - 0.2, val_s, 0.4, label="val MSE", color="#c45c26")
axes[0].bar(x + 0.2, test_s, 0.4, label="test MSE", color="#1f4e79")
axes[0].set_xticks(x, labels, rotation=20)
axes[0].set_ylabel("MSE (Å²)")
axes[0].set_title("Fase 10 — error")
axes[0].legend()
axes[1].bar(labels, times, color="#3d6b4f")
axes[1].set_ylabel("detik")
axes[1].set_title("Waktu train")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
plt.show()
""",
        ),
        cell(
            "markdown",
            """## 8. EGNN JAX (subprocess)
""",
        ),
        cell(
            "code",
            """jx_metrics = METRICS_DIR / "phase10_egnn_jax.json"
if RETRAIN_JAX or not metrics_ready(jx_metrics, n_graphs):
    run_script("train_jax.py", CFG_PATH)
jx = json.loads(jx_metrics.read_text(encoding="utf-8"))
for k in ("n_proteins", "n_train", "n_val", "n_test", "final_train_mse",
          "final_val_loss", "final_test_mse", "training_time_s", "compile_s"):
    print(f"  {k:20s} {jx.get(k)}")

jx_csv = METRICS_DIR / "phase10_egnn_jax_epochs.csv"
hist = list(csv.DictReader(jx_csv.open(encoding="utf-8", newline="")))
epochs = [int(r["epoch"]) for r in hist]
tr = [float(r["loss"]) for r in hist]
va = [float(r["val_loss"]) for r in hist if (r.get("val_loss") or "").strip()]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(epochs, tr, label="train last-batch", color="#1f4e79")
if va:
    ax.plot(epochs[: len(va)], va, label="val MSE", color="#c45c26")
ax.set_title("EGNN JAX")
ax.legend()
fig.tight_layout()
plt.show()
""",
        ),
        cell(
            "markdown",
            """## 9. PyTorch vs JAX (EGNN)
""",
        ),
        cell(
            "code",
            """egnn_pt = METRICS_DIR / "phase10_egnn_pytorch.json"
if not egnn_pt.is_file():
    egnn_pt = METRICS_DIR / "phase10_egnn_pytorch.json"
pt = json.loads(egnn_pt.read_text(encoding="utf-8")) if egnn_pt.is_file() else {}
print(f"{'':10s} {'n':>6s} {'val':>8s} {'test':>8s} {'sec':>8s}")
print(f"{'PyTorch':10s} {pt.get('n_proteins'):>6} {pt.get('final_val_loss'):8.4f} {pt.get('final_test_mse'):8.4f} {pt.get('training_time_s'):8.1f}")
print(f"{'JAX':10s} {jx.get('n_proteins'):>6} {jx.get('final_val_loss'):8.4f} {jx.get('final_test_mse'):8.4f} {jx.get('training_time_s'):8.1f}")
print("compile JAX s:", jx.get("compile_s"))
""",
        ),
        cell(
            "markdown",
            """## 10. Fase 11 — tuning EGNN

Grid: `tune_hidden_dim` × `tune_num_layers` × `tune_learning_rate`, epoch lebih pendek (`tune_epochs`). Pemenang dipilih dari **val MSE**.
""",
        ),
        cell(
            "code",
            """tune_path = METRICS_DIR / "phase11_tune.json"
if RETRAIN_TUNE or not tune_path.is_file():
    run_script("tune_egnn.py", CFG_PATH)
tune = json.loads(tune_path.read_text(encoding="utf-8"))
print("best_val:", tune.get("best_val"))
print(f"{'id':28s} {'H':>4s} {'L':>3s} {'lr':>8s} {'val':>8s} {'test':>8s}")
hs, ls, vals = [], [], []
for r in tune["runs"]:
    print(f"{r['experiment_id']:28s} {r['hidden_dim']:4d} {r['num_layers']:3d} {r['learning_rate']:8.4g} {r['final_val_loss']:8.4f} {r['final_test_mse']:8.4f}")
    hs.append(r["hidden_dim"])
    ls.append(r["num_layers"])
    vals.append(r["final_val_loss"])
fig, ax = plt.subplots(figsize=(7, 3.6))
tags = [f"H{h} L{l}" for h, l in zip(hs, ls)]
ax.bar(tags, vals, color="#1f4e79")
ax.set_ylabel("val MSE")
ax.set_title("Fase 11 — EGNN grid")
fig.tight_layout()
plt.show()
""",
        ),
        cell(
            "markdown",
            """## 11. Gallery eksperimen (tombol)

Bukan hanya protein: EDA, model, kurva training, compare, tuning, results — digabung di **hub** dengan tombol navigasi.

- File: [`results/plots/hub.html`](../results/plots/hub.html)
- PNG: `results/plots/experiment/`
""",
        ),
        cell(
            "code",
            """FORCE_EXP_PLOTS = False
hub_path = PLOTS_DIR / "hub.html"
need_exp = FORCE_EXP_PLOTS or not (PLOTS_DIR / "experiment" / "compare_mse.png").is_file()
if need_exp:
    run_script("plot_experiment.py", "--force")
else:
    run_script("plot_experiment.py")

from IPython.display import IFrame
nb_hub = ROOT / "notebooks" / "experiment_hub.html"
print("hub:", hub_path)
print("notebook hub:", nb_hub)
display(Markdown("Gunakan tombol **EDA / Model / Training / Compare / Tuning / Results / Protein** di bawah."))
display(IFrame(src=str(nb_hub.as_posix()), width="100%", height=820))
""",
        ),
        cell(
            "markdown",
            """## 12. Visualisasi protein

PNG struktur di `results/plots/`. Gallery protein: [`gallery.html`](../results/plots/gallery.html).
""",
        ),
        cell(
            "code",
            """from src.visualization.plots import write_protein_gallery

n_scatter = len(list(PLOTS_DIR.glob("*_allatom_scatter.png")))
print(f"all-atom PNG: {n_scatter}  graf target: {n_graphs}")
if FORCE_VIZ or n_scatter < n_graphs:
    run_script("visualize.py", CFG_PATH)

trained = sorted(p.name.replace("_ca.npz", "") for p in STRUCTURES_DIR.glob("*_ca.npz"))
gallery = write_protein_gallery(trained, PLOTS_DIR / "gallery.html")
print("gallery:", gallery)

def show_protein_panels(pid: str, width: int = 220) -> None:
    display(Markdown(f"**{pid}**"))
    for fname, label in [
        (f"{pid}_cartoon.png", "cartoon"),
        (f"{pid}_allatom_scatter.png", "all-atom"),
        (f"{pid}_ca_backbone.png", "C-alpha"),
        (f"{pid}_graph.png", "graf"),
    ]:
        path = PLOTS_DIR / fname
        if path.is_file():
            display(Markdown(f"*{label}*"))
            display(Image(filename=str(path), width=width))

show_protein_panels(EXAMPLE_ID)
""",
        ),
        cell(
            "markdown",
            """### Gallery protein (lazy-load)
""",
        ),
        cell(
            "code",
            """nb_gallery = ROOT / "notebooks" / "protein_gallery.html"
write_protein_gallery(trained, nb_gallery, src_prefix="../results/plots/")
display(Markdown(f"{len(trained)} struktur — atau buka hub tombol Protein"))
display(HTML(nb_gallery.read_text(encoding="utf-8")))
""",
        ),
        cell(
            "markdown",
            """## Catatan

- **test MSE** = metrik banding model. `loss` CSV = batch train terakhir, bukan rata-rata epoch.
- Hub eksperimen: tombol EDA / Model / Training / Compare / Tuning / Results / Protein.
- PairMLP / GCN / GAT tanpa koordinat; SchNet jarak; EGNN E(3).
- Tombol sel 1 = subprocess. JAX tidak di-import di kernel notebook.
""",
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
