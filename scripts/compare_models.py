"""Train all comparison models (PyTorch subprocesses only — no jax import)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.pytorch.factory import COMPARE_MODELS
from src.paths import CONFIGS_DIR, METRICS_DIR, ensure_dirs


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    proc = subprocess.run(args, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    ensure_dirs()
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS_DIR / "egnn.yaml"
    cfg = load_config(cfg_path)
    models = list(cfg.get("compare_models") or COMPARE_MODELS)
    py = sys.executable
    rows = []
    for name in models:
        exp = f"phase10_{name}_pytorch"
        _run(
            [
                py,
                "-u",
                str(ROOT / "scripts" / "train_pytorch.py"),
                str(cfg_path),
                "--model",
                name,
                "--experiment-id",
                exp,
                "--ckpt",
                str(ROOT / "results" / "checkpoints" / f"{name}_pytorch.pt"),
                "--metrics-json",
                str(METRICS_DIR / f"{exp}.json"),
                "--metrics-csv",
                str(METRICS_DIR / f"{exp}_epochs.csv"),
            ]
        )
        metrics = json.loads((METRICS_DIR / f"{exp}.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "model": metrics.get("model"),
                "slug": name,
                "n_params": metrics.get("n_params"),
                "n_proteins": metrics.get("n_proteins"),
                "n_train": metrics.get("n_train"),
                "n_val": metrics.get("n_val"),
                "n_test": metrics.get("n_test"),
                "final_train_mse": metrics.get("final_train_mse"),
                "final_val_loss": metrics.get("final_val_loss"),
                "final_test_mse": metrics.get("final_test_mse"),
                "training_time_s": metrics.get("training_time_s"),
                "peak_allocated_mb": (metrics.get("memory") or {}).get("peak_allocated_mb"),
                "epochs": metrics.get("epochs"),
            }
        )

    ranked = sorted(
        [r for r in rows if r.get("final_test_mse") is not None],
        key=lambda r: r["final_test_mse"],
    )
    summary = {
        "experiment_id": "phase10_compare",
        "task": "edge_ca_distance_regression",
        "note": "test MSE is the comparison metric; loss not a biological label",
        "models": rows,
        "best_test": ranked[0] if ranked else None,
    }
    out = METRICS_DIR / "phase10_compare.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
