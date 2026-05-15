"""Statistical analysis: bootstrap CIs, paired tests, correlations.

Used by ``scripts/run_significance.py`` and ``scripts/run_correlation.py``.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns ``(mean, lower, upper)``."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return float(arr.mean()), lower, upper


def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns ``(statistic, p_value)``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # If all paired differences are zero, Wilcoxon is degenerate.
    if np.all(a == b):
        return 0.0, 1.0

    res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def pairwise_significance(
    model_scores: dict[str, Sequence[float]],
    bonferroni: bool = True,
) -> pd.DataFrame:
    """All pairwise paired Wilcoxon tests across models.

    Returns a long-format DataFrame: ``model_a, model_b, statistic, p_value,
    p_corrected, significant``.
    """
    names = list(model_scores.keys())
    pairs = list(combinations(names, 2))
    n_comparisons = max(1, len(pairs))

    rows = []
    for a, b in pairs:
        stat, p = paired_wilcoxon(model_scores[a], model_scores[b])
        p_corr = min(1.0, p * n_comparisons) if bonferroni else p
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "statistic": stat,
                "p_value": p,
                "p_corrected": p_corr,
                "significant": p_corr < 0.05,
            }
        )
    return pd.DataFrame(rows)


def correlation_analysis(
    metrics_df: pd.DataFrame,
    metric_cols: Sequence[str] = ("ESS", "CED", "TFR"),
    error_col: str = "is_error",
    confidence_col: Optional[str] = "gt_confidence",
) -> pd.DataFrame:
    """Compute Pearson + Spearman correlations between TCE metrics and
    prediction-quality indicators.

    Returns a long DataFrame: ``metric, target, pearson_r, pearson_p,
    spearman_r, spearman_p``.
    """
    rows = []
    targets = {"misclassification": error_col}
    if confidence_col is not None and confidence_col in metrics_df.columns:
        targets["confidence_margin"] = confidence_col

    for metric in metric_cols:
        if metric not in metrics_df.columns:
            continue
        x = metrics_df[metric].to_numpy(dtype=np.float64)
        valid_x = ~np.isnan(x)
        for target_name, target_col in targets.items():
            if target_col not in metrics_df.columns:
                continue
            y = metrics_df[target_col].to_numpy(dtype=np.float64)
            mask = valid_x & ~np.isnan(y)
            if mask.sum() < 3:
                continue
            pr = stats.pearsonr(x[mask], y[mask])
            sr = stats.spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "metric": metric,
                    "target": target_name,
                    "n": int(mask.sum()),
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                    "spearman_r": float(sr.statistic),
                    "spearman_p": float(sr.pvalue),
                }
            )
    return pd.DataFrame(rows)


def add_error_column(
    df: pd.DataFrame,
    gt_col: str = "gt_stance",
    pred_col: str = "gt_prediction",
    out_col: str = "is_error",
) -> pd.DataFrame:
    """Append a 0/1 misclassification indicator column."""
    df = df.copy()
    df[out_col] = (
        df[gt_col].astype(str).str.upper().str.strip()
        != df[pred_col].astype(str).str.upper().str.strip()
    ).astype(int)
    return df


def summary_table(
    metrics_df: pd.DataFrame,
    metric_cols: Sequence[str] = ("ESS", "CED", "TFR"),
    group_col: Optional[str] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Mean + bootstrap CI for each metric (overall or grouped)."""
    rows = []
    if group_col is None:
        groups: dict[str, pd.DataFrame] = {"OVERALL": metrics_df}
    else:
        groups = {g: sub for g, sub in metrics_df.groupby(group_col)}

    for g, sub in groups.items():
        for m in metric_cols:
            if m not in sub.columns:
                continue
            mean, lo, hi = bootstrap_ci(sub[m].tolist(), n_bootstrap=n_bootstrap, ci=ci, seed=seed)
            rows.append(
                {
                    "group": g,
                    "metric": m,
                    "n": int(len(sub)),
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)
