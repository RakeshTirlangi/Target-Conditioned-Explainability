"""Input formatting and similar-target parsing."""

from __future__ import annotations

import ast
from typing import Iterable

import pandas as pd

INPUT_TEMPLATE = "{tweet} [SEP] Target: {target}"


def format_input(tweet: str, target: str) -> str:
    """Build the tweet + target input string used during fine-tuning."""
    return INPUT_TEMPLATE.format(tweet=str(tweet).strip(), target=str(target).strip())


def parse_similar_targets(cell_value, n: int = 3) -> list[str]:
    """Parse the `similar targets` cell, which may be a list-as-string or CSV-as-string.

    Returns up to `n` non-empty target strings.
    """
    if pd.isna(cell_value):
        return []

    raw = str(cell_value).strip()
    if not raw:
        return []

    # Preferred: Python-list literal
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()][:n]
    except (ValueError, SyntaxError):
        pass

    # Fallback: comma-separated
    return [x.strip() for x in raw.split(",") if x.strip()][:n]


def pad_similar_targets(targets: Iterable[str], n: int, fill: str = "") -> list[str]:
    """Ensure exactly `n` targets, padding with `fill` if fewer were parsed."""
    targets = list(targets)
    if len(targets) >= n:
        return targets[:n]
    return targets + [fill] * (n - len(targets))
