"""Plots: overview dashboard, correlations, model comparison.

All plot functions accept a pre-computed DataFrame and save a single PNG.
Run ``scripts/make_plots.py`` for the full set used in the paper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper")


def plot_overview(pairs_df: pd.DataFrame, output_path: str | Path) -> Path:
    """4-panel dashboard: ESS/CED/TFR histograms + per-stance bars."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, m, color in zip(
        axes.flat[:3],
        ["ESS", "CED", "TFR"],
        ["#1f77b4", "#ff7f0e", "#2ca02c"],
    ):
        sns.histplot(pairs_df[m], bins=40, kde=True, color=color, ax=ax)
        ax.set_title(f"Distribution of {m}")
        ax.set_xlabel(m)

    per_stance = pairs_df.groupby("gt_stance")[["ESS", "CED", "TFR"]].mean().reset_index()
    per_stance_melt = per_stance.melt(
        id_vars="gt_stance", var_name="metric", value_name="value"
    )
    sns.barplot(
        data=per_stance_melt,
        x="gt_stance",
        y="value",
        hue="metric",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Mean TCE Metrics by GT Stance")
    axes[1, 1].set_xlabel("Stance")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend(title="Metric", loc="upper left")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_correlations(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric_cols: Sequence[str] = ("ESS", "CED", "TFR"),
    error_col: str = "is_error",
    confidence_col: str = "gt_confidence",
) -> Path:
    """Scatter plot grid: each TCE metric vs prediction quality."""
    cols_present = [c for c in metric_cols if c in metrics_df.columns]
    targets = []
    if error_col in metrics_df.columns:
        targets.append((error_col, "Misclassification (0/1)"))
    if confidence_col in metrics_df.columns:
        targets.append((confidence_col, "Confidence margin"))
    if not targets or not cols_present:
        logger.warning("plot_correlations: required columns missing — skipping")
        return Path(output_path)

    fig, axes = plt.subplots(
        len(cols_present),
        len(targets),
        figsize=(4.5 * len(targets), 3.5 * len(cols_present)),
        squeeze=False,
    )
    for i, m in enumerate(cols_present):
        for j, (tc, tl) in enumerate(targets):
            ax = axes[i, j]
            data = metrics_df[[m, tc]].dropna()
            if tc == error_col:
                sns.boxplot(
                    data=data, x=tc, y=m, ax=ax,
                    boxprops={"alpha": 0.7},
                )
                ax.set_xlabel(tl)
            else:
                sns.regplot(
                    data=data, x=tc, y=m, ax=ax,
                    scatter_kws={"alpha": 0.3, "s": 10},
                    line_kws={"color": "red"},
                )
                ax.set_xlabel(tl)
            ax.set_ylabel(m)
            ax.set_title(f"{m} vs {tl}")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    metric_cols: Sequence[str] = ("ESS", "CED", "TFR"),
) -> Path:
    """Grouped bar plot comparing models on the headline TCE metrics."""
    if "model" not in summary_df.columns:
        raise ValueError("summary_df must contain a 'model' column")

    melted = summary_df.melt(id_vars="model", value_vars=list(metric_cols),
                             var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("TCE Metrics Across Models (lower is better)")
    ax.set_ylabel("Mean metric value")
    ax.legend(title="Model", loc="upper left")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_sensitivity_curve(
    sensitivity_df: pd.DataFrame,
    x_col: str,
    output_path: str | Path,
    metric_cols: Sequence[str] = ("ESS", "CED", "TFR"),
) -> Path:
    """Line plot of metric vs hyperparameter value."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in metric_cols:
        if m not in sensitivity_df.columns:
            continue
        ax.plot(sensitivity_df[x_col], sensitivity_df[m], marker="o", label=m)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Metric value")
    ax.set_title(f"TCE Sensitivity to {x_col}")
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
