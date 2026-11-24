"""Experiment-level plots: EDA, models, training, compare, tune, results.

Saved under results/plots/experiment/ and linked from hub.html buttons.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.paths import EXP_PLOTS_DIR, METADATA_DIR, METRICS_DIR, PLOTS_DIR, ensure_dirs

_COLORS = {
    "blue": "#1f4e79",
    "orange": "#c45c26",
    "green": "#3d6b4f",
    "gray": "#5c5c5c",
    "teal": "#2a6f6f",
}


def _save(fig: plt.Figure, name: str, *, force: bool = False) -> Path:
    ensure_dirs()
    EXP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EXP_PLOTS_DIR / name
    if force or not out.is_file() or out.stat().st_size < 100:
        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _finite(vals: list[float | None]) -> list[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x:
            out.append(x)
    return out


def plot_eda(*, force: bool = False) -> list[Path]:
    csv_path = METADATA_DIR / "structure_summary.csv"
    rows = _load_csv_rows(csv_path)
    written: list[Path] = []
    if not rows:
        return written

    n_ca = np.array([float(r["n_ca"]) for r in rows if r.get("n_ca")], dtype=np.float64)
    resol = np.array(
        [float(r["resolution"]) for r in rows if r.get("resolution") not in ("", "None", None)],
        dtype=np.float64,
    )
    n_chains = np.array([float(r["n_chains"]) for r in rows if r.get("n_chains")], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(n_ca, bins=40, color=_COLORS["blue"], alpha=0.88)
    ax.axvline(float(np.median(n_ca)), color=_COLORS["orange"], ls="--", lw=1.4, label="median")
    ax.set_xlabel("N C-alpha")
    ax.set_ylabel("jumlah struktur")
    ax.set_title(f"EDA — distribusi N_CA (n={len(n_ca)})")
    ax.legend()
    written.append(_save(fig, "eda_nca_hist.png", force=force))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(resol, bins=35, color=_COLORS["green"], alpha=0.88)
    ax.set_xlabel("resolusi (Å)")
    ax.set_ylabel("jumlah struktur")
    ax.set_title("EDA — resolusi X-ray")
    written.append(_save(fig, "eda_resolution_hist.png", force=force))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if len(n_ca) == len(resol):
        ax.scatter(resol, n_ca, s=8, alpha=0.35, c=_COLORS["teal"])
    else:
        # align by filtering rows that have both
        pairs = [
            (float(r["resolution"]), float(r["n_ca"]))
            for r in rows
            if r.get("n_ca") and r.get("resolution") not in ("", "None", None)
        ]
        if pairs:
            rr, nn = zip(*pairs)
            ax.scatter(rr, nn, s=8, alpha=0.35, c=_COLORS["teal"])
    ax.set_xlabel("resolusi (Å)")
    ax.set_ylabel("N C-alpha")
    ax.set_title("EDA — N_CA vs resolusi")
    written.append(_save(fig, "eda_nca_vs_resolution.png", force=force))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(n_chains, bins=np.arange(0.5, n_chains.max() + 1.5, 1), color=_COLORS["gray"], alpha=0.9)
    ax.set_xlabel("jumlah rantai")
    ax.set_ylabel("jumlah struktur")
    ax.set_title("EDA — jumlah rantai polimer")
    written.append(_save(fig, "eda_nchains_hist.png", force=force))

    cmp_path = METRICS_DIR / "phase10_compare.json"
    if cmp_path.is_file():
        cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
        m0 = (cmp.get("models") or [{}])[0]
        sizes = [
            int(m0.get("n_train") or 0),
            int(m0.get("n_val") or 0),
            int(m0.get("n_test") or 0),
        ]
        labels = ["train", "val", "test"]
        if sum(sizes) > 0:
            fig, ax = plt.subplots(figsize=(5.8, 5.0))
            ax.pie(
                sizes,
                labels=[f"{lb}\n{n}" for lb, n in zip(labels, sizes)],
                colors=[_COLORS["blue"], _COLORS["orange"], _COLORS["green"]],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"EDA — split train/val/test (n={sum(sizes)})")
            written.append(_save(fig, "eda_split_pie.png", force=force))
    return written


def plot_model_overview(*, force: bool = False) -> list[Path]:
    written: list[Path] = []
    cmp_path = METRICS_DIR / "phase10_compare.json"
    if not cmp_path.is_file():
        return written
    cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
    models = cmp.get("models") or []
    if not models:
        return written

    labels = [str(m["model"]) for m in models]
    params = [int(m.get("n_params") or 0) for m in models]
    notes = {
        "PairMLP": "tanpa pesan graf",
        "BaselineGCN": "mean tetangga",
        "GAT": "attention α_ij",
        "SchNet": "RBF jarak",
        "EGNN": "E(3) equivariant",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(labels, params, color=_COLORS["blue"], alpha=0.9)
    ax.set_ylabel("jumlah parameter")
    ax.set_title("Model — ukuran parameter (fase 10)")
    ax.tick_params(axis="x", rotation=15)
    for bar, lab in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            notes.get(lab, ""),
            ha="center",
            va="bottom",
            fontsize=8,
            color=_COLORS["gray"],
        )
    written.append(_save(fig, "model_params.png", force=force))

    # Card-style text figure describing roles
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axis("off")
    lines = [
        "Model diagnostik — regresi jarak C-alpha per edge",
        "",
        "PairMLP     : embedding residu independen, tanpa message passing",
        "BaselineGCN : agregasi mean tetangga (topologi + tipe AA)",
        "GAT         : perhatian antar tetangga (tanpa koordinat)",
        "SchNet      : filter kontinu dari ||x_i − x_j|| (RBF)",
        "EGNN        : pesan ||x_i − x_j||² + update koordinat E(3)",
        "",
        "Head semua model: MLP([h_i || h_j]) → jarak edge (Å)",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11, transform=ax.transAxes)
    ax.set_title("Model — ringkasan arsitektur")
    written.append(_save(fig, "model_architecture.png", force=force))
    return written


def plot_training_curves(*, force: bool = False) -> list[Path]:
    written: list[Path] = []
    series = [
        ("mlp", "PairMLP", METRICS_DIR / "phase10_mlp_pytorch_epochs.csv", _COLORS["gray"]),
        ("gcn", "BaselineGCN", METRICS_DIR / "phase10_gcn_pytorch_epochs.csv", _COLORS["green"]),
        ("gat", "GAT", METRICS_DIR / "phase10_gat_pytorch_epochs.csv", "#8a5a2b"),
        ("schnet", "SchNet", METRICS_DIR / "phase10_schnet_pytorch_epochs.csv", _COLORS["teal"]),
        ("egnn", "EGNN PyTorch", METRICS_DIR / "phase10_egnn_pytorch_epochs.csv", _COLORS["blue"]),
        ("egnn_jax", "EGNN JAX", METRICS_DIR / "phase10_egnn_jax_epochs.csv", _COLORS["orange"]),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for slug, label, path, color in series:
        rows = _load_csv_rows(path)
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        val = []
        for r in rows:
            raw = (r.get("val_loss") or "").strip()
            val.append(float(raw) if raw else np.nan)
        ax.plot(epochs, val, label=label, color=color, lw=1.8)
        # per-model figure
        fig_m, ax_m = plt.subplots(figsize=(7.0, 3.8))
        tr = [float(r["loss"]) for r in rows]
        ax_m.plot(epochs, tr, label="train last-batch", color=_COLORS["blue"], lw=1.6)
        if np.isfinite(val).any():
            ax_m.plot(epochs, val, label="val MSE", color=_COLORS["orange"], lw=1.6)
        ax_m.set_xlabel("epoch")
        ax_m.set_ylabel("MSE (Å²)")
        ax_m.set_title(f"Training — {label}")
        ax_m.legend()
        written.append(_save(fig_m, f"train_{slug}.png", force=force))

    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE (Å²)")
    ax.set_title("Training — kurva validasi semua model")
    ax.legend(fontsize=8, ncol=2)
    written.append(_save(fig, "train_val_overlay.png", force=force))
    return written


def plot_compare(*, force: bool = False) -> list[Path]:
    written: list[Path] = []
    cmp_path = METRICS_DIR / "phase10_compare.json"
    if not cmp_path.is_file():
        return written
    cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
    models = cmp.get("models") or []
    if not models:
        return written

    labels = [str(m["model"]) for m in models]
    train = [m.get("final_train_mse") for m in models]
    val = [m.get("final_val_loss") for m in models]
    test = [m.get("final_test_mse") for m in models]
    times = [float(m.get("training_time_s") or 0) for m in models]
    peaks = [float(m.get("peak_allocated_mb") or 0) for m in models]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x - w, train, w, label="train", color=_COLORS["gray"])
    ax.bar(x, val, w, label="val", color=_COLORS["orange"])
    ax.bar(x + w, test, w, label="test", color=_COLORS["blue"])
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("MSE (Å²)")
    ax.set_title("Compare — error train / val / test")
    ax.legend()
    written.append(_save(fig, "compare_mse.png", force=force))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, times, color=_COLORS["green"], alpha=0.9)
    ax.set_ylabel("detik")
    ax.set_title("Compare — waktu training")
    ax.tick_params(axis="x", rotation=15)
    written.append(_save(fig, "compare_time.png", force=force))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, peaks, color=_COLORS["teal"], alpha=0.9)
    ax.set_ylabel("peak VRAM (MB)")
    ax.set_title("Compare — peak memory GPU")
    ax.tick_params(axis="x", rotation=15)
    written.append(_save(fig, "compare_memory.png", force=force))
    return written


def plot_tuning(*, force: bool = False) -> list[Path]:
    written: list[Path] = []
    tune_path = METRICS_DIR / "phase11_tune.json"
    if not tune_path.is_file():
        return written
    tune = json.loads(tune_path.read_text(encoding="utf-8"))
    runs = tune.get("runs") or []
    if not runs:
        return written

    tags = [f"H{r['hidden_dim']} L{r['num_layers']}" for r in runs]
    vals = [r.get("final_val_loss") for r in runs]
    tests = [r.get("final_test_mse") for r in runs]

    x = np.arange(len(tags))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - w / 2, vals, w, label="val", color=_COLORS["orange"])
    ax.bar(x + w / 2, tests, w, label="test", color=_COLORS["blue"])
    ax.set_xticks(x, tags)
    ax.set_ylabel("MSE (Å²)")
    ax.set_title("Tuning — EGNN grid (fase 11)")
    ax.legend()
    best = tune.get("best_val") or {}
    if best:
        ax.annotate(
            f"best val: H{best.get('hidden_dim')} L{best.get('num_layers')}",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            color=_COLORS["gray"],
        )
    written.append(_save(fig, "tune_grid.png", force=force))

    # overlay tune epoch curves if present
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    palette = [_COLORS["blue"], _COLORS["orange"], _COLORS["green"], _COLORS["teal"]]
    for i, r in enumerate(runs):
        path = METRICS_DIR / f"{r['experiment_id']}_epochs.csv"
        rows = _load_csv_rows(path)
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        val = []
        for row in rows:
            raw = (row.get("val_loss") or "").strip()
            val.append(float(raw) if raw else np.nan)
        ax.plot(epochs, val, label=tags[i], color=palette[i % len(palette)], lw=1.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE")
    ax.set_title("Tuning — kurva val per konfigurasi")
    ax.legend(fontsize=8)
    written.append(_save(fig, "tune_curves.png", force=force))
    return written


def plot_results(*, force: bool = False) -> list[Path]:
    written: list[Path] = []
    cmp_path = METRICS_DIR / "phase10_compare.json"
    bench_path = METRICS_DIR / "phase10_benchmark.json"
    tune_path = METRICS_DIR / "phase11_tune.json"

    # Ranking bar from compare
    if cmp_path.is_file():
        cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
        models = sorted(
            [m for m in (cmp.get("models") or []) if m.get("final_test_mse") is not None],
            key=lambda m: float(m["final_test_mse"]),
        )
        if models:
            labels = [str(m["model"]) for m in models]
            tests = [float(m["final_test_mse"]) for m in models]
            fig, ax = plt.subplots(figsize=(8.2, 4.4))
            colors = [_COLORS["green"]] + [_COLORS["blue"]] * (len(labels) - 1)
            ax.barh(labels[::-1], tests[::-1], color=colors[::-1])
            ax.set_xlabel("test MSE (Å²)")
            ax.set_title("Results — ranking test MSE (lebih kecil lebih baik)")
            best = cmp.get("best_test") or models[0]
            ax.text(
                0.98,
                0.05,
                f"terbaik: {best.get('model')} ({float(best.get('final_test_mse')):.3f})",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                color=_COLORS["gray"],
            )
            written.append(_save(fig, "result_ranking.png", force=force))

    # PyTorch vs JAX
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        pt = bench.get("pytorch") or {}
        jx = bench.get("jax") or {}
        metrics = ["final_val_loss", "final_test_mse", "training_time_s"]
        names = ["val MSE", "test MSE", "train s"]
        pt_vals = [pt.get(k) for k in metrics]
        jx_vals = [jx.get(k) for k in metrics]
        # split time to secondary axis-like: two panels
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        x = np.arange(2)
        axes[0].bar(x - 0.18, [pt_vals[0], pt_vals[1]], 0.36, label="PyTorch", color=_COLORS["blue"])
        axes[0].bar(x + 0.18, [jx_vals[0], jx_vals[1]], 0.36, label="JAX", color=_COLORS["orange"])
        axes[0].set_xticks(x, names[:2])
        axes[0].set_ylabel("MSE")
        axes[0].set_title("Results — EGNN PyTorch vs JAX (error)")
        axes[0].legend()
        axes[1].bar(["PyTorch", "JAX"], [pt_vals[2] or 0, jx_vals[2] or 0], color=[_COLORS["blue"], _COLORS["orange"]])
        axes[1].set_ylabel("detik")
        axes[1].set_title("Results — waktu train")
        written.append(_save(fig, "result_pytorch_vs_jax.png", force=force))

    # Summary board
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.axis("off")
    lines = ["RESULTS BOARD — edge C-alpha distance regression", ""]
    if cmp_path.is_file():
        cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
        best = cmp.get("best_test") or {}
        lines.append(
            f"Best test (PyTorch compare): {best.get('model')}  "
            f"test={best.get('final_test_mse')}  val={best.get('final_val_loss')}"
        )
        lines.append("")
        lines.append(f"{'model':12s} {'test':>8s} {'val':>8s} {'sec':>8s}")
        for m in cmp.get("models") or []:
            lines.append(
                f"{str(m.get('model')):12s} "
                f"{float(m.get('final_test_mse') or 0):8.4f} "
                f"{float(m.get('final_val_loss') or 0):8.4f} "
                f"{float(m.get('training_time_s') or 0):8.1f}"
            )
    if tune_path.is_file():
        tune = json.loads(tune_path.read_text(encoding="utf-8"))
        b = tune.get("best_val") or {}
        lines.extend(
            [
                "",
                f"Best tune (val): H{b.get('hidden_dim')} L{b.get('num_layers')}  "
                f"val={b.get('final_val_loss')}  test={b.get('final_test_mse')}",
            ]
        )
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                f"EGNN JAX test={bench.get('jax', {}).get('final_test_mse')}  "
                f"PyTorch test={bench.get('pytorch', {}).get('final_test_mse')}",
            ]
        )
    ax.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=10, transform=ax.transAxes)
    written.append(_save(fig, "result_board.png", force=force))
    return written


SECTION_SPECS: list[tuple[str, str, list[str]]] = [
    (
        "eda",
        "EDA",
        [
            "eda_nca_hist.png",
            "eda_resolution_hist.png",
            "eda_nca_vs_resolution.png",
            "eda_nchains_hist.png",
            "eda_split_pie.png",
        ],
    ),
    ("models", "Model", ["model_params.png", "model_architecture.png"]),
    (
        "training",
        "Training",
        [
            "train_val_overlay.png",
            "train_mlp.png",
            "train_gcn.png",
            "train_gat.png",
            "train_schnet.png",
            "train_egnn.png",
            "train_egnn_jax.png",
        ],
    ),
    ("compare", "Compare", ["compare_mse.png", "compare_time.png", "compare_memory.png"]),
    ("tuning", "Tuning", ["tune_grid.png", "tune_curves.png"]),
    (
        "results",
        "Results",
        ["result_ranking.png", "result_pytorch_vs_jax.png", "result_board.png"],
    ),
]


def write_experiment_hub(
    *,
    out_path: Path | None = None,
    img_prefix: str = "experiment/",
    protein_gallery_href: str = "gallery.html",
) -> Path:
    """Hub HTML with buttons for EDA / model / train / compare / tune / results / protein."""
    ensure_dirs()
    out_path = out_path or (PLOTS_DIR / "hub.html")

    panels: list[str] = [
        f"""
<section id="panel-protein" class="panel">
  <h2>Gallery protein</h2>
  <p>Cartoon, all-atom, C-alpha, graf per PDB ID.</p>
  <p><a class="open" href="{protein_gallery_href}" target="_blank" rel="noopener">Buka gallery protein (full page)</a></p>
  <iframe src="{protein_gallery_href}" title="protein gallery" loading="lazy"></iframe>
</section>
"""
    ]

    for key, title, files in SECTION_SPECS:
        figs = []
        for fname in files:
            p = EXP_PLOTS_DIR / fname
            if p.is_file():
                figs.append(
                    f'<figure><img src="{img_prefix}{fname}" alt="{fname}" loading="lazy">'
                    f"<figcaption>{fname.replace('_', ' ').removesuffix('.png')}</figcaption></figure>"
                )
            else:
                figs.append(f"<figure><figcaption>{fname}: belum digenerate</figcaption></figure>")
        panels.append(
            f'<section id="panel-{key}" class="panel">'
            f"<h2>{title}</h2>"
            f'<div class="grid">{"".join(figs)}</div></section>'
        )

    tab_bits = ['<button type="button" class="tab" data-panel="protein">Protein</button>']
    for key, title, _ in SECTION_SPECS:
        tab_bits.append(f'<button type="button" class="tab" data-panel="{key}">{title}</button>')
    buttons = "".join(tab_bits)
    known_js = json.dumps(["protein"] + [k for k, _, _ in SECTION_SPECS])

    html = f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>Experiment hub — RCSB GNN</title>
<style>
:root {{
  --bg: #f4f2ec; --ink: #1c1c1c; --card: #ffffff; --accent: #1f4e79;
  --muted: #666; --line: #d9d4c8;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
header {{
  position: sticky; top: 0; z-index: 5; background: #ebe7dc; border-bottom: 1px solid var(--line);
  padding: 12px 16px 10px;
}}
h1 {{ margin: 0 0 8px; font-size: 1.15rem; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.tab {{
  border: 1px solid var(--line); background: #faf8f3; color: var(--ink);
  padding: 8px 14px; cursor: pointer; font-size: 0.92rem;
}}
.tab:hover {{ border-color: var(--accent); }}
.tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
main {{ padding: 16px; }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}
h2 {{ font-size: 1.05rem; margin: 0 0 10px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
figure {{ margin: 0; background: var(--card); padding: 8px; border: 1px solid var(--line); }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 0.78rem; color: var(--muted); margin-top: 6px; }}
iframe {{ width: 100%; height: 75vh; border: 1px solid var(--line); background: #fff; }}
.open {{ color: var(--accent); }}
p {{ color: var(--muted); }}
</style></head><body>
<header>
  <h1>Gallery eksperimen — EDA · Model · Training · Compare · Tuning · Results · Protein</h1>
  <nav class="tabs">{buttons}</nav>
</header>
<main>
{"".join(panels)}
</main>
<script>
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const known = new Set({known_js});
function show(id) {{
  tabs.forEach(t => t.classList.toggle("active", t.dataset.panel === id));
  panels.forEach(p => p.classList.toggle("active", p.id === "panel-" + id));
  history.replaceState(null, "", "#" + id);
}}
tabs.forEach(t => t.addEventListener("click", () => show(t.dataset.panel)));
const initial = (location.hash || "#eda").replace("#", "");
show(known.has(initial) ? initial : "eda");
</script>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_all_experiment_plots(*, force: bool = False) -> dict[str, list[Path]]:
    return {
        "eda": plot_eda(force=force),
        "models": plot_model_overview(force=force),
        "training": plot_training_curves(force=force),
        "compare": plot_compare(force=force),
        "tuning": plot_tuning(force=force),
        "results": plot_results(force=force),
    }
