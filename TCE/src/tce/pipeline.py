"""TCE pipeline: per-instance ESS / CED / TFR over (GT, similar-target) pairs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tce.inference import confidence_margin, predict_label, predict_proba
from tce.metrics import compute_all
from tce.model import ModelBundle
from tce.preprocessing import format_input, pad_similar_targets, parse_similar_targets
from tce.shap_explainer import (
    attribution_for_predicted_class,
    build_explainer,
    explain,
    select_background,
)

logger = logging.getLogger(__name__)


def _none_stance_tweets(
    df: pd.DataFrame, text_col: str, stance_col: str, label: str = "NONE"
) -> list[str]:
    mask = df[stance_col].astype(str).str.upper().str.strip() == label.upper()
    return df.loc[mask, text_col].astype(str).tolist()


def run_tce(
    df: pd.DataFrame,
    bundle: ModelBundle,
    *,
    text_col: str,
    gt_target_col: str,
    gt_stance_col: str,
    similar_targets_col: str,
    n_similar: int = 3,
    max_evals: int = 50,
    max_length: int = 128,
    batch_size: int = 16,
    top_k: int = 5,
    n_background_candidates: int = 15,
    masker_regex: str = r"\W+",
    seed: int = 42,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the full TCE pipeline.

    For each row, the pipeline:
      1. Formats four inputs: the tweet with its GT target plus ``n_similar`` rephrased targets.
      2. Predicts the stance label for each.
      3. Computes SHAP attribution vectors for all four inputs.
      4. Compares the GT vector against each similar-target vector and computes ESS, CED, TFR.

    Returns a long-format DataFrame: one row per (instance, similar-target) pair.
    Columns: ``row_idx, tweet, gt_target, gt_stance, gt_prediction, gt_confidence,
    similar_target, sim_prediction, sim_confidence, ESS, CED, TFR``.
    """
    bg_pool = _none_stance_tweets(df, text_col, gt_stance_col)
    background = select_background(
        bundle,
        bg_pool,
        n_candidates=n_background_candidates,
        max_length=max_length,
        seed=seed,
    )
    explainer = build_explainer(
        bundle,
        background=background,
        masker_regex=masker_regex,
        max_length=max_length,
        batch_size=batch_size,
    )

    records: list[dict] = []
    iterator: Iterable[int] = range(len(df))
    if progress:
        iterator = tqdm(iterator, total=len(df), desc="TCE", ncols=80)

    for i in iterator:
        row = df.iloc[i]
        tweet = str(row[text_col])
        gt_target = str(row[gt_target_col])
        gt_stance = str(row[gt_stance_col])
        sim_targets = pad_similar_targets(
            parse_similar_targets(row[similar_targets_col], n=n_similar),
            n=n_similar,
            fill=gt_target,
        )

        all_targets = [gt_target] + sim_targets
        all_inputs = [format_input(tweet, t) for t in all_targets]

        try:
            probs = predict_proba(
                all_inputs, bundle, max_length=max_length, batch_size=batch_size
            )
            labels = predict_label(probs, bundle.label_names)
            margins = confidence_margin(probs)

            shap_vals = explain(explainer, all_inputs, max_evals=max_evals)

            # Take the attribution for the GT-predicted class so all four
            # vectors are directly comparable.
            gt_class = int(np.argmax(probs[0]))
            phi_gt = attribution_for_predicted_class(shap_vals, gt_class, row_idx=0)

            for j, sim_target in enumerate(sim_targets, start=1):
                phi_sim = attribution_for_predicted_class(shap_vals, gt_class, row_idx=j)
                m = compute_all(phi_gt, phi_sim, k=top_k)
                records.append(
                    {
                        "row_idx": i,
                        "tweet": tweet,
                        "gt_target": gt_target,
                        "gt_stance": gt_stance,
                        "gt_prediction": labels[0],
                        "gt_confidence": float(margins[0]),
                        "similar_target": sim_target,
                        "sim_prediction": labels[j],
                        "sim_confidence": float(margins[j]),
                        "ESS": m["ESS"],
                        "CED": m["CED"],
                        "TFR": m["TFR"],
                    }
                )

        except Exception as e:  # noqa: BLE001
            logger.warning("Row %d failed: %s", i, e)
            continue

    return pd.DataFrame.from_records(records)


def aggregate_per_instance(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Average ESS / CED / TFR over similar targets per row."""
    agg = (
        pairs_df.groupby(
            ["row_idx", "gt_target", "gt_stance", "gt_prediction", "gt_confidence"],
            as_index=False,
        )[["ESS", "CED", "TFR"]]
        .mean()
    )
    return agg


def aggregate_per_target(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Average ESS / CED / TFR per GT target (across all rows)."""
    return (
        pairs_df.groupby("gt_target", as_index=False)[["ESS", "CED", "TFR"]]
        .mean()
        .sort_values("TFR", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_per_stance(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Average ESS / CED / TFR per GT stance label."""
    return (
        pairs_df.groupby("gt_stance", as_index=False)[["ESS", "CED", "TFR"]]
        .mean()
        .reset_index(drop=True)
    )


def save_results(
    pairs_df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "tce",
) -> dict[str, Path]:
    """Save pair-level + three aggregates to CSVs. Returns the file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["pairs"] = output_dir / f"{prefix}_pairs.csv"
    paths["instances"] = output_dir / f"{prefix}_per_instance.csv"
    paths["targets"] = output_dir / f"{prefix}_per_target.csv"
    paths["stances"] = output_dir / f"{prefix}_per_stance.csv"

    pairs_df.to_csv(paths["pairs"], index=False)
    aggregate_per_instance(pairs_df).to_csv(paths["instances"], index=False)
    aggregate_per_target(pairs_df).to_csv(paths["targets"], index=False)
    aggregate_per_stance(pairs_df).to_csv(paths["stances"], index=False)

    logger.info("Saved 4 result files to %s", output_dir)
    return paths
