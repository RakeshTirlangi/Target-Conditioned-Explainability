"""SHAP PartitionExplainer wrapper for stance detection models.

Computes word-level attribution vectors using a regex masker. The explainer
is built once per run and reused across rows for consistency.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from tce.inference import predict_proba
from tce.model import ModelBundle

logger = logging.getLogger(__name__)


def select_background(
    bundle: ModelBundle,
    none_tweets: Sequence[str],
    n_candidates: int = 15,
    max_length: int = 128,
    seed: int = 42,
) -> str:
    """Pick the single ``none``-stance tweet whose probability vector is
    closest (in L2) to the mean of ``n_candidates`` random samples.

    This becomes the single background reference fed to SHAP.
    """
    rng = np.random.default_rng(seed)
    candidates = list(none_tweets)
    if len(candidates) > n_candidates:
        idx = rng.choice(len(candidates), size=n_candidates, replace=False)
        candidates = [candidates[i] for i in idx]
    if not candidates:
        return "this is a neutral statement"

    probs = predict_proba(candidates, bundle, max_length=max_length)
    mean_prob = probs.mean(axis=0)
    dists = np.linalg.norm(probs - mean_prob, axis=1)
    best = int(np.argmin(dists))
    logger.info(
        "Selected background reference (idx=%d of %d candidates, L2=%.4f)",
        best, len(candidates), dists[best],
    )
    return candidates[best]


def build_explainer(
    bundle: ModelBundle,
    background: str | Sequence[str],
    masker_regex: str = r"\W+",
    max_length: int = 128,
    batch_size: int = 16,
):
    """Construct a SHAP PartitionExplainer over the model's softmax output.

    The masker is a word-boundary regex (``\\W+`` by default), which gives
    word-level rather than token-level attributions.
    """
    import shap

    def _pred(texts: Sequence[str]) -> np.ndarray:
        return predict_proba(
            list(texts), bundle, max_length=max_length, batch_size=batch_size
        )

    masker = shap.maskers.Text(re.compile(masker_regex))
    bg = background if isinstance(background, (list, tuple)) else [background]
    explainer = shap.PartitionExplainer(_pred, masker, output_names=bundle.label_names)
    explainer._tce_background = list(bg)
    return explainer


def explain(
    explainer,
    texts: Sequence[str],
    max_evals: int = 50,
    silent: bool = True,
):
    """Run the explainer over a batch of inputs and return a SHAP Explanation."""
    import shap  # noqa: F401  (ensures shap import error surfaces early)

    return explainer(list(texts), max_evals=max_evals, silent=silent)


def attribution_for_predicted_class(
    shap_values,
    predicted_class_idx: int,
    row_idx: int = 0,
) -> np.ndarray:
    """Slice a SHAP Explanation to the attribution vector for one row + class.

    SHAP returns shape ``(n_inputs, n_tokens, n_classes)`` for multi-class.
    """
    values = shap_values.values
    if values.ndim == 3:
        return np.asarray(values[row_idx, :, predicted_class_idx], dtype=np.float64)
    if values.ndim == 2:
        return np.asarray(values[row_idx, :], dtype=np.float64)
    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")
