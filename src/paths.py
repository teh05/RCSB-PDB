from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
METADATA_DIR = ROOT / "metadata"
PROCESSED_DIR = ROOT / "processed"
STRUCTURES_DIR = PROCESSED_DIR / "structures"
GRAPHS_DIR = PROCESSED_DIR / "graphs"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
EXP_PLOTS_DIR = PLOTS_DIR / "experiment"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
CONFIGS_DIR = ROOT / "configs"


def ensure_dirs() -> None:
    for d in (
        METADATA_DIR,
        STRUCTURES_DIR,
        GRAPHS_DIR,
        PLOTS_DIR,
        EXP_PLOTS_DIR,
        METRICS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
