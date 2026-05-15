#!/usr/bin/env python
"""Hyperparameter sensitivity sweeps (max_evals, background_size).

For the top-k sweep, run TCE once and rerun ``compute_all`` at different k
values from the cached pair vectors; that path is not exposed here because
it requires storing per-pair attribution vectors. Instead this script sweeps
``max_evals`` and ``background_size`` directly.

Usage:
    python scripts/run_sensitivity.py --config configs/deberta.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tce.config import load_config, save_config  # noqa: E402
from tce.data import load_data  # noqa: E402
from tce.model import load_model  # noqa: E402
from tce.sensitivity import (  # noqa: E402
    save_sensitivity_results,
    sweep_background_size,
    sweep_max_evals,
)
from tce.utils import ensure_dir, resolve_device, set_seed, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TCE hyperparameter sensitivity sweeps.")
    p.add_argument("--config", required=True)
    p.add_argument("--sweep", choices=["max_evals", "background_size", "both"], default="both")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(Path(cfg.experiment.output_dir) / "sensitivity")
    setup_logging(level=cfg.logging.level, log_to_file=True,
                  log_file="sensitivity.log", output_dir=output_dir)
    log = logging.getLogger("run_sensitivity")

    save_config(cfg, output_dir / "resolved_config.yaml")
    set_seed(cfg.experiment.seed)

    device = resolve_device(cfg.model.device)
    df = load_data(
        path=cfg.data.path,
        text_col=cfg.data.text_column,
        gt_target_col=cfg.data.gt_target_column,
        gt_stance_col=cfg.data.gt_stance_column,
        similar_targets_col=cfg.data.similar_targets_column,
        sample_size=cfg.sensitivity.sensitivity_sample_size,
        seed=cfg.experiment.seed,
    )

    bundle = load_model(
        cfg.model.name_or_path,
        device=device,
        label_names=cfg.data.stance_labels,
    )

    base_kwargs = dict(
        n_similar=cfg.data.n_similar,
        max_evals=cfg.shap.max_evals,
        max_length=cfg.model.max_length,
        batch_size=cfg.model.batch_size,
        top_k=cfg.metrics.top_k,
        n_background_candidates=cfg.shap.n_background_candidates,
        masker_regex=cfg.shap.masker_regex,
        seed=cfg.experiment.seed,
    )

    results: dict[str, "pandas.DataFrame"] = {}

    if args.sweep in {"max_evals", "both"}:
        log.info("Running max_evals sweep: %s", cfg.sensitivity.max_evals_values)
        results["max_evals"] = sweep_max_evals(
            df, bundle,
            eval_budgets=cfg.sensitivity.max_evals_values,
            text_col=cfg.data.text_column,
            gt_target_col=cfg.data.gt_target_column,
            gt_stance_col=cfg.data.gt_stance_column,
            similar_targets_col=cfg.data.similar_targets_column,
            base_kwargs=base_kwargs,
        )

    if args.sweep in {"background_size", "both"}:
        log.info("Running background_size sweep: %s", cfg.sensitivity.background_sizes)
        results["background_size"] = sweep_background_size(
            df, bundle,
            background_sizes=cfg.sensitivity.background_sizes,
            text_col=cfg.data.text_column,
            gt_target_col=cfg.data.gt_target_column,
            gt_stance_col=cfg.data.gt_stance_column,
            similar_targets_col=cfg.data.similar_targets_column,
            base_kwargs=base_kwargs,
        )

    paths = save_sensitivity_results(results, output_dir)
    for k, v in paths.items():
        log.info("Saved %s -> %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
