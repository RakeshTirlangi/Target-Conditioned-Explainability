#!/usr/bin/env python
"""Pairwise Wilcoxon signed-rank tests across models for one TCE metric.

Inputs: a glob of per-instance TCE CSVs, one per model. Each CSV must contain
``row_idx`` and the metric column (e.g. CED).

Usage:
    python scripts/run_significance.py \\
        --metric CED \\
        --csvs outputs/deberta/tce_deberta_per_instance.csv \\
               outputs/bert/tce_bert_per_instance.csv \\
               outputs/distilbert/tce_distilbert_per_instance.csv \\
               outputs/roberta/tce_roberta_per_instance.csv \\
        --names DeBERTa BERT DistilBERT RoBERTa \\
        --output outputs/significance/CED_pairwise.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tce.statistics import pairwise_significance, summary_table  # noqa: E402
from tce.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise Wilcoxon significance across models.")
    p.add_argument("--metric", default="CED", choices=["ESS", "CED", "TFR"])
    p.add_argument("--csvs", nargs="+", required=True, help="Per-instance CSVs, one per model")
    p.add_argument("--names", nargs="+", required=True, help="Model names matching --csvs order")
    p.add_argument("--output", required=True, help="Output CSV path for the significance table")
    p.add_argument("--summary-output", default=None,
                   help="Optional: also write a per-model bootstrap-CI summary here")
    p.add_argument("--no-bonferroni", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.csvs) != len(args.names):
        raise SystemExit("--csvs and --names must have the same length")

    setup_logging(level="INFO")
    log = logging.getLogger("run_significance")

    # Load each model's per-instance file and align on row_idx so the tests are paired.
    dfs = {n: pd.read_csv(p) for n, p in zip(args.names, args.csvs)}
    common_rows = None
    for n, df in dfs.items():
        if "row_idx" not in df.columns:
            raise SystemExit(f"{n}: 'row_idx' column missing")
        rows = set(df["row_idx"].tolist())
        common_rows = rows if common_rows is None else (common_rows & rows)

    common_rows = sorted(common_rows or [])
    log.info("Found %d common row_idx values across %d models", len(common_rows), len(dfs))

    aligned: dict[str, list[float]] = {}
    for n, df in dfs.items():
        sub = (
            df[df["row_idx"].isin(common_rows)]
            .drop_duplicates(subset="row_idx")
            .set_index("row_idx")
            .loc[common_rows]
        )
        aligned[n] = sub[args.metric].astype(float).tolist()

    sig_df = pairwise_significance(aligned, bonferroni=not args.no_bonferroni)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(args.output, index=False)
    log.info("Wrote %d pairwise tests to %s", len(sig_df), args.output)
    log.info("\n%s", sig_df.to_string(index=False))

    if args.summary_output:
        summary_rows = []
        for n, vals in aligned.items():
            tmp = pd.DataFrame({args.metric: vals})
            s = summary_table(tmp, metric_cols=(args.metric,))
            s["model"] = n
            summary_rows.append(s)
        combined = pd.concat(summary_rows, ignore_index=True)
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.summary_output, index=False)
        log.info("Wrote model summary -> %s", args.summary_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
