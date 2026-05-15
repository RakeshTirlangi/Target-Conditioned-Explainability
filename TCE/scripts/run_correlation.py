#!/usr/bin/env python
"""Correlate per-instance TCE scores with prediction-quality indicators.

Inputs: the ``*_per_instance.csv`` produced by ``run_tce.py`` for one model.
Outputs: a correlation table (Pearson + Spearman, per metric, per target).

Usage:
    python scripts/run_correlation.py \\
        --instances outputs/deberta/tce_deberta_per_instance.csv \\
        --pairs     outputs/deberta/tce_deberta_pairs.csv \\
        --output    outputs/deberta/correlations.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tce.statistics import add_error_column, correlation_analysis  # noqa: E402
from tce.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate TCE metrics with prediction quality.")
    p.add_argument("--instances", required=True, help="Per-instance TCE CSV")
    p.add_argument("--pairs", required=True, help="Pair-level TCE CSV (for gt_prediction + confidence)")
    p.add_argument("--output", required=True, help="Output CSV path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level="INFO")
    log = logging.getLogger("run_correlation")

    instances = pd.read_csv(args.instances)
    pairs = pd.read_csv(args.pairs)

    # Pair-level rows carry gt_prediction + confidence per instance; the
    # per-instance aggregate carries the averaged TCE metrics. Merge them.
    keys_pair = (
        pairs[["row_idx", "gt_prediction", "gt_confidence"]]
        .drop_duplicates(subset="row_idx")
    )
    merged = instances.merge(keys_pair, on="row_idx", how="left")
    merged = add_error_column(merged, gt_col="gt_stance", pred_col="gt_prediction")

    corr_df = correlation_analysis(
        merged,
        metric_cols=("ESS", "CED", "TFR"),
        error_col="is_error",
        confidence_col="gt_confidence",
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(args.output, index=False)
    log.info("Wrote %d correlation rows to %s", len(corr_df), args.output)
    log.info("\n%s", corr_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
