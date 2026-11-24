"""Generate EDA / model / training / compare / tuning / results plots + hub HTML."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import EXP_PLOTS_DIR, PLOTS_DIR, ensure_dirs
from src.visualization.experiment_plots import (
    generate_all_experiment_plots,
    write_experiment_hub,
)


def main() -> int:
    ensure_dirs()
    force = "--force" in sys.argv
    out = generate_all_experiment_plots(force=force)
    for section, paths in out.items():
        print(f"{section}: {len(paths)} png")
        for p in paths:
            print(f"  {p.name}")

    hub = write_experiment_hub(
        out_path=PLOTS_DIR / "hub.html",
        img_prefix="experiment/",
        protein_gallery_href="gallery.html",
    )
    nb_hub = ROOT / "notebooks" / "experiment_hub.html"
    write_experiment_hub(
        out_path=nb_hub,
        img_prefix="../results/plots/experiment/",
        protein_gallery_href="../results/plots/gallery.html",
    )
    print(f"hub: {hub}")
    print(f"notebook hub: {nb_hub}")
    print(f"dir: {EXP_PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
