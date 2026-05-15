#!/usr/bin/env python
"""Aggregate per-model TCE results into a single cross-model summary table.

Usage:
    python scripts/aggregate_results.py \\
        --pairs outputs/deberta/tce_deberta_pairs.csv \\
                outputs/bert/tce_bert_pairs.csv \\
                outputs/distilbert/tce_distilbert_pairs.csv \\
                outputs/roberta/tce_roberta_pairs.csv \\
        --names DeBERTa BERT DistilBERT RoBERTa \\
        --output outputs/cross_model_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tce.statistics import summary_table  # noqa: E402
from tce.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate TCE pairs across models.")
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.pairs) != len(args.names):
        raise SystemExit("--pairs and --names must have the same length")

    setup_logging(level="INFO")
    log = logging.getLogger("aggregate_results")

    rows = []
    for n, p in zip(args.names, args.pairs):
        df = pd.read_csv(p)
        s = summary_table(df, metric_cols=("ESS", "CED", "TFR"),
                          n_bootstrap=args.bootstrap)
        s["model"] = n
        rows.append(s)

    out = pd.concat(rows, ignore_index=True)
    pivot = out.pivot_table(
        index="model", columns="metric", values=["mean", "ci_low", "ci_high"]
    )
    pivot.columns = [f"{m}_{c}" for c, m in pivot.columns]
    pivot = pivot.reset_index()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.output, index=False)
    log.info("Wrote cross-model summary -> %s", args.output)
    log.info("\n%s", pivot.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
