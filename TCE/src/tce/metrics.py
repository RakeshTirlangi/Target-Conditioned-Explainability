"""TCE stability metrics: ESS, CED, TFR.

All three operate on a pair of SHAP attribution vectors:
    phi_gt  - attribution vector under the ground-truth target
    phi_sim - attribution vector under a semantically similar target
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-8


def _align(v1: np.ndarray, v2: np.ndarray, strategy: str = "pad") -> tuple[np.ndarray, np.ndarray]:
    """Make two attribution vectors the same length.

    ``pad`` zero-pads the shorter vector; ``truncate`` clips the longer one.
    Tokenisation under different targets can yield different lengths, so the
    aligned form is required before any pointwise comparison.
    """
    v1 = np.asarray(v1, dtype=np.float64).ravel()
    v2 = np.asarray(v2, dtype=np.float64).ravel()

    if len(v1) == len(v2):
        return v1, v2

    if strategy == "pad":
        n = max(len(v1), len(v2))
        return (
            np.pad(v1, (0, n - len(v1))),
            np.pad(v2, (0, n - len(v2))),
        )
    if strategy == "truncate":
        n = min(len(v1), len(v2))
        return v1[:n], v2[:n]
    raise ValueError(f"Unknown align strategy: {strategy!r}")


def explanation_shift_score(
    phi_gt: np.ndarray,
    phi_sim: np.ndarray,
    align: str = "pad",
) -> float:
    """Explanation Shift Score (ESS): mean absolute attribution shift.

    .. math::
       \\mathrm{ESS} = \\frac{1}{n} \\sum_{i=1}^{n} |\\phi^{GT}_i - \\phi^{sim}_i|

    Lower is better. ESS = 0 iff every token's attribution is identical.
    """
    a, b = _align(phi_gt, phi_sim, align)
    if len(a) == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def cosine_explanation_divergence(
    phi_gt: np.ndarray,
    phi_sim: np.ndarray,
    align: str = "pad",
) -> float:
    """Cosine Explanation Divergence (CED): 1 - cosine similarity.

    Range :math:`[0, 2]`. 0 = identical direction; 2 = exactly opposite.
    Lower is better.
    """
    a, b = _align(phi_gt, phi_sim, align)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPSILON
    return float(1.0 - np.dot(a, b) / denom)


def token_flip_rate(
    phi_gt: np.ndarray,
    phi_sim: np.ndarray,
    k: int = 5,
    align: str = "pad",
) -> float:
    """Token Flip Rate (TFR): fraction of top-k tokens that change.

    .. math::
       \\mathrm{TFR} = 1 - \\frac{|T_k(\\phi^{GT}) \\cap T_k(\\phi^{sim})|}{k}

    where :math:`T_k(\\cdot)` is the set of indices with the largest absolute
    attribution. Lower is better.
    """
    a, b = _align(phi_gt, phi_sim, align)
    if len(a) == 0:
        return 0.0
    k_eff = min(k, len(a))
    if k_eff == 0:
        return 0.0
    top_a = set(np.argsort(np.abs(a))[-k_eff:].tolist())
    top_b = set(np.argsort(np.abs(b))[-k_eff:].tolist())
    return float(1.0 - len(top_a & top_b) / k_eff)


def compute_all(
    phi_gt: np.ndarray,
    phi_sim: np.ndarray,
    k: int = 5,
    align: str = "pad",
) -> dict[str, float]:
    """Compute (ESS, CED, TFR) in one call."""
    return {
        "ESS": explanation_shift_score(phi_gt, phi_sim, align=align),
        "CED": cosine_explanation_divergence(phi_gt, phi_sim, align=align),
        "TFR": token_flip_rate(phi_gt, phi_sim, k=k, align=align),
    }
