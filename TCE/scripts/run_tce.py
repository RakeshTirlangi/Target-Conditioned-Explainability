#!/usr/bin/env python
"""Run the full TCE pipeline for one model.

Usage:
    python scripts/run_tce.py --config configs/deberta.yaml
    python scripts/run_tce.py --config configs/deberta.yaml --sample-size 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``src/`` importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tce.config import load_config, save_config  # noqa: E402
from tce.data import load_data  # noqa: E402
from tce.model import load_model  # noqa: E402
from tce.pipeline import run_tce, save_results  # noqa: E402
from tce.utils import ensure_dir, resolve_device, set_seed, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TCE evaluation for one model.")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Override config.data.sample_size (e.g. 50 for smoke test)")
    p.add_argument("--data-path", type=str, default=None,
                   help="Override config.data.path")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override config.experiment.output_dir")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.sample_size is not None:
        cfg.data.sample_size = args.sample_size
    if args.data_path is not None:
        cfg.data.path = args.data_path
    if args.output_dir is not None:
        cfg.experiment.output_dir = args.output_dir

    output_dir = ensure_dir(cfg.experiment.output_dir)
    setup_logging(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_file=cfg.logging.log_file,
        output_dir=output_dir,
    )
    log = logging.getLogger("run_tce")
    log.info("Experiment: %s", cfg.experiment.name)

    save_config(cfg, output_dir / "resolved_config.yaml")
    set_seed(cfg.experiment.seed)

    device = resolve_device(cfg.model.device)
    log.info("Device: %s", device)

    df = load_data(
        path=cfg.data.path,
        text_col=cfg.data.text_column,
        gt_target_col=cfg.data.gt_target_column,
        gt_stance_col=cfg.data.gt_stance_column,
        similar_targets_col=cfg.data.similar_targets_column,
        sample_size=cfg.data.sample_size,
        seed=cfg.experiment.seed,
    )

    bundle = load_model(
        cfg.model.name_or_path,
        device=device,
        label_names=cfg.data.stance_labels,
    )

    pairs_df = run_tce(
        df,
        bundle,
        text_col=cfg.data.text_column,
        gt_target_col=cfg.data.gt_target_column,
        gt_stance_col=cfg.data.gt_stance_column,
        similar_targets_col=cfg.data.similar_targets_column,
        n_similar=cfg.data.n_similar,
        max_evals=cfg.shap.max_evals,
        max_length=cfg.model.max_length,
        batch_size=cfg.model.batch_size,
        top_k=cfg.metrics.top_k,
        n_background_candidates=cfg.shap.n_background_candidates,
        masker_regex=cfg.shap.masker_regex,
        seed=cfg.experiment.seed,
        progress=not args.no_progress,
    )

    if pairs_df.empty:
        log.error("Pipeline produced no rows.")
        return 1

    paths = save_results(pairs_df, output_dir, prefix=cfg.experiment.name)
    for k, v in paths.items():
        log.info("  %-10s -> %s", k, v)

    log.info("=" * 60)
    log.info("Summary (mean over %d pairs):", len(pairs_df))
    log.info("  ESS = %.4f", pairs_df["ESS"].mean())
    log.info("  CED = %.4f", pairs_df["CED"].mean())
    log.info("  TFR = %.4f", pairs_df["TFR"].mean())
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
