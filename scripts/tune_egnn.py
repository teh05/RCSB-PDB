"""Small hyperparameter grid for EGNN (PyTorch). Phase 11."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
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
    hiddens = [int(x) for x in cfg.get("tune_hidden_dim") or [64]]
    layers_list = [int(x) for x in cfg.get("tune_num_layers") or [3]]
    lrs = [float(x) for x in cfg.get("tune_learning_rate") or [0.001]]
    epochs = int(cfg.get("tune_epochs", 8))
    py = sys.executable
    runs = []
    for hidden in hiddens:
        for layers in layers_list:
            for lr in lrs:
                tag = f"h{hidden}_l{layers}_lr{lr:g}".replace(".", "p")
                exp = f"phase11_egnn_{tag}"
                _run(
                    [
                        py,
                        "-u",
                        str(ROOT / "scripts" / "train_pytorch.py"),
                        str(cfg_path),
                        "--model",
                        "egnn",
                        "--hidden-dim",
                        str(hidden),
                        "--num-layers",
                        str(layers),
                        "--learning-rate",
                        str(lr),
                        "--epochs",
                        str(epochs),
                        "--experiment-id",
                        exp,
                        "--ckpt",
                        str(ROOT / "results" / "checkpoints" / f"{exp}.pt"),
                        "--metrics-json",
                        str(METRICS_DIR / f"{exp}.json"),
                        "--metrics-csv",
                        str(METRICS_DIR / f"{exp}_epochs.csv"),
                    ]
                )
                m = json.loads((METRICS_DIR / f"{exp}.json").read_text(encoding="utf-8"))
                runs.append(
                    {
                        "experiment_id": exp,
                        "hidden_dim": hidden,
                        "num_layers": layers,
                        "learning_rate": lr,
                        "epochs": epochs,
                        "final_val_loss": m.get("final_val_loss"),
                        "final_test_mse": m.get("final_test_mse"),
                        "training_time_s": m.get("training_time_s"),
                        "n_params": m.get("n_params"),
                    }
                )

    ranked = sorted(
        [r for r in runs if r.get("final_val_loss") is not None],
        key=lambda r: r["final_val_loss"],
    )
    summary = {
        "experiment_id": "phase11_tune",
        "model": "EGNN",
        "metric": "val MSE (select); test MSE reported",
        "runs": runs,
        "best_val": ranked[0] if ranked else None,
    }
    out = METRICS_DIR / "phase11_tune.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
