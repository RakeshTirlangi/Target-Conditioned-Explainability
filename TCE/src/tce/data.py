"""Data loading and validation for stance detection datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    path: str | Path,
    text_col: str,
    gt_target_col: str,
    gt_stance_col: str,
    similar_targets_col: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load and validate the stance detection dataset.

    Args:
        path: Path to the CSV file.
        text_col: Column name for the (cleaned) tweet text.
        gt_target_col: Column name for the ground-truth target.
        gt_stance_col: Column name for the ground-truth stance label.
        similar_targets_col: Column name holding the list of similar targets.
        sample_size: If set, randomly subsample to this many rows.
        seed: RNG seed used when sampling.

    Returns:
        A DataFrame with the required columns and missing rows dropped.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if any required column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = [text_col, gt_target_col, gt_stance_col, similar_targets_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    before = len(df)
    df = df[required].dropna(subset=[text_col, gt_target_col, gt_stance_col]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing text/target/stance", dropped)

    df[gt_stance_col] = df[gt_stance_col].astype(str).str.upper().str.strip()

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d rows (seed=%d)", sample_size, seed)

    logger.info("Loaded %d rows", len(df))
    logger.info("Stance distribution:\n%s", df[gt_stance_col].value_counts().to_string())

    return df
