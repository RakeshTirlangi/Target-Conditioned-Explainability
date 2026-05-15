"""Hyperparameter sensitivity sweeps for the TCE pipeline.

These produce the Section VI.C tables in the paper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from tce.metrics import compute_all
from tce.model import ModelBundle
from tce.pipeline import run_tce

logger = logging.getLogger(__name__)


def sweep_k(
    pairs_df: pd.DataFrame,
    k_values: Sequence[int] = (3, 5, 10, 20),
    phi_gt_col: str = "phi_gt",
    phi_sim_col: str = "phi_sim",
) -> pd.DataFrame:
    """Recompute TFR at multiple k values, given cached SHAP vectors.

    If your pairs DataFrame already stores per-pair attribution vectors,
    pass their column names; otherwise this is a no-op fallback that returns
    only the precomputed TFR column.
    """
    has_vectors = phi_gt_col in pairs_df.columns and phi_sim_col in pairs_df.columns
    if not has_vectors:
        logger.warning(
            "Pairs DataFrame has no cached attribution vectors — sweep_k cannot "
            "recompute TFR at different k. Returning original TFR only."
        )
        return pairs_df[["gt_stance", "TFR"]].copy()

    rows = []
    for k in k_values:
        for _, row in pairs_df.iterrows():
            phi_gt = np.asarray(row[phi_gt_col])
            phi_sim = np.asarray(row[phi_sim_col])
            m = compute_all(phi_gt, phi_sim, k=k)
            rows.append({"k": k, "gt_stance": row["gt_stance"], "TFR": m["TFR"]})
    out = pd.DataFrame(rows)
    return out.groupby(["gt_stance", "k"], as_index=False)["TFR"].mean()


def sweep_max_evals(
    df_subset: pd.DataFrame,
    bundle: ModelBundle,
    eval_budgets: Sequence[int] = (25, 50, 100, 250, 500),
    *,
    text_col: str,
    gt_target_col: str,
    gt_stance_col: str,
    similar_targets_col: str,
    base_kwargs: dict,
) -> pd.DataFrame:
    """For each ``max_evals`` value, run TCE and report mean (ESS, CED, TFR)."""
    rows = []
    for budget in eval_budgets:
        kwargs = dict(base_kwargs)
        kwargs["max_evals"] = budget
        kwargs["progress"] = False
        logger.info("Sensitivity: max_evals=%d", budget)
        pairs = run_tce(
            df_subset,
            bundle,
            text_col=text_col,
            gt_target_col=gt_target_col,
            gt_stance_col=gt_stance_col,
            similar_targets_col=similar_targets_col,
            **kwargs,
        )
        rows.append(
            {
                "max_evals": budget,
                "n_pairs": int(len(pairs)),
                "ESS": float(pairs["ESS"].mean()),
                "CED": float(pairs["CED"].mean()),
                "TFR": float(pairs["TFR"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sweep_background_size(
    df_subset: pd.DataFrame,
    bundle: ModelBundle,
    background_sizes: Sequence[int] = (1, 5, 15, 50),
    *,
    text_col: str,
    gt_target_col: str,
    gt_stance_col: str,
    similar_targets_col: str,
    base_kwargs: dict,
) -> pd.DataFrame:
    """For each background size, run TCE and report mean (ESS, CED, TFR).

    Note: increasing background size raises ``n_background_candidates`` to
    at least that size.
    """
    rows = []
    for size in background_sizes:
        kwargs = dict(base_kwargs)
        kwargs["n_background_candidates"] = max(size, kwargs.get("n_background_candidates", 15))
        kwargs["progress"] = False
        logger.info("Sensitivity: background_size=%d", size)
        pairs = run_tce(
            df_subset,
            bundle,
            text_col=text_col,
            gt_target_col=gt_target_col,
            gt_stance_col=gt_stance_col,
            similar_targets_col=similar_targets_col,
            **kwargs,
        )
        rows.append(
            {
                "background_size": size,
                "n_pairs": int(len(pairs)),
                "ESS": float(pairs["ESS"].mean()),
                "CED": float(pairs["CED"].mean()),
                "TFR": float(pairs["TFR"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_sensitivity_results(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist each sensitivity table to a separate CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in results.items():
        p = output_dir / f"sensitivity_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths
