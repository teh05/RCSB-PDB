"""PyTorch vs JAX EGNN CUDA benchmark.

Does not import torch or jax — they must run in separate processes on Windows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import CONFIGS_DIR, GRAPHS_DIR, METRICS_DIR, STRUCTURES_DIR, ensure_dirs


def _run(args: list[str]) -> None:
    print("+", " ".join(args))
    proc = subprocess.run(args, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _ensure_graphs() -> None:
    if not list(STRUCTURES_DIR.glob("*_ca.npz")):
        _run([sys.executable, str(ROOT / "scripts" / "preprocess.py")])
    cfg = CONFIGS_DIR / "egnn.yaml"
    if not list(GRAPHS_DIR.glob("*_ca_r*.npz")):
        _run([sys.executable, str(ROOT / "scripts" / "build_graph.py"), str(cfg)])


def main() -> int:
    ensure_dirs()
    from_metrics = "--from-metrics" in sys.argv
    if not from_metrics:
        _ensure_graphs()
        py = sys.executable
        _run([py, str(ROOT / "scripts" / "check_e3_jax.py")])
        _run([py, "-m", "pytest", str(ROOT / "tests" / "test_e3_pytorch.py"), "-q"])
        _run([py, str(ROOT / "scripts" / "train_pytorch.py"), str(CONFIGS_DIR / "egnn.yaml")])
        _run([py, str(ROOT / "scripts" / "train_jax.py"), str(CONFIGS_DIR / "egnn.yaml")])

    pt9 = METRICS_DIR / "phase9_egnn_pytorch.json"
    jx9 = METRICS_DIR / "phase9_egnn_jax.json"
    if pt9.exists() and jx9.exists():
        pt_path, jx_path = pt9, jx9
        out_name = "phase9_benchmark.json"
        exp_id = "phase9_pytorch_vs_jax"
    else:
        pt_path = METRICS_DIR / "phase8_egnn_pytorch.json"
        jx_path = METRICS_DIR / "phase8_egnn_jax.json"
        out_name = "phase8_benchmark.json"
        exp_id = "phase8_pytorch_vs_jax"
    e3_path = METRICS_DIR / "phase8_e3_jax.json"
    for p in (pt_path, jx_path, e3_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run without --from-metrics first.")

    pt = json.loads(pt_path.read_text(encoding="utf-8"))
    jx = json.loads(jx_path.read_text(encoding="utf-8"))
    e3 = json.loads(e3_path.read_text(encoding="utf-8"))

    pt_step = pt.get("mean_step_ms_ex_first") or pt.get("training_time_s")
    jx_step = jx.get("mean_step_ms_ex_first") or jx.get("training_time_s")
    ratio = None
    if isinstance(pt_step, (int, float)) and isinstance(jx_step, (int, float)) and jx_step:
        ratio = float(pt_step) / float(jx_step)

    summary = {
        "experiment_id": exp_id,
        "model": "EGNN",
        "task": "edge_ca_distance_regression",
        "dataset": pt.get("dataset"),
        "n_proteins": pt.get("n_proteins"),
        "n_train": pt.get("n_train"),
        "n_val": pt.get("n_val"),
        "epochs": pt.get("epochs"),
        "note": "engineering run; not a published generalization benchmark. Loss not comparable across inits.",
        "e3_jax": e3,
        "pytorch": {
            "final_loss": pt.get("final_loss"),
            "final_val_loss": pt.get("final_val_loss"),
            "training_time_s": pt.get("training_time_s"),
            "first_step_ms": pt.get("first_step_ms"),
            "mean_step_ms_ex_first": pt.get("mean_step_ms_ex_first"),
            "memory": pt.get("memory"),
            "GPU": pt.get("GPU"),
        },
        "jax": {
            "final_loss": jx.get("final_loss"),
            "final_val_loss": jx.get("final_val_loss"),
            "training_time_s": jx.get("training_time_s"),
            "first_step_ms": jx.get("first_step_ms"),
            "mean_step_ms_ex_first": jx.get("mean_step_ms_ex_first"),
            "compile_s": jx.get("compile_s"),
            "memory": jx.get("memory"),
            "GPU": jx.get("GPU"),
        },
        "pytorch_over_jax_train_s": ratio,
    }
    out = METRICS_DIR / out_name
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
