#!/usr/bin/env python
"""Regenerate the paper's figures from a TCE results directory.

Usage:
    python scripts/make_plots.py --results-dir outputs/deberta --prefix tce_deberta
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tce.statistics import add_error_column  # noqa: E402
from tce.utils import setup_logging  # noqa: E402
from tce.visualization import plot_correlations, plot_overview  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate TCE plots.")
    p.add_argument("--results-dir", required=True, help="Directory with TCE CSVs")
    p.add_argument("--prefix", required=True, help="Filename prefix (e.g. tce_deberta)")
    p.add_argument("--plots-dir", default=None, help="Override output plots dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level="INFO")
    log = logging.getLogger("make_plots")

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = results_dir / f"{args.prefix}_pairs.csv"
    instances_path = results_dir / f"{args.prefix}_per_instance.csv"
    if not pairs_path.exists():
        raise SystemExit(f"Missing {pairs_path}")

    pairs = pd.read_csv(pairs_path)
    log.info("Loaded %d pair rows", len(pairs))

    p1 = plot_overview(pairs, plots_dir / "fig1_overview.png")
    log.info("Saved %s", p1)

    if instances_path.exists():
        instances = pd.read_csv(instances_path)
        keys = pairs[["row_idx", "gt_prediction", "gt_confidence"]].drop_duplicates("row_idx")
        merged = instances.merge(keys, on="row_idx", how="left")
        merged = add_error_column(merged, gt_col="gt_stance", pred_col="gt_prediction")
        p2 = plot_correlations(merged, plots_dir / "fig2_correlations.png")
        log.info("Saved %s", p2)
    else:
        log.warning("Per-instance CSV not found — skipping correlation plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
