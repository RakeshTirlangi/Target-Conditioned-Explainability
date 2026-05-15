"""TCE: Target-Conditioned Explainability.

A framework for quantifying explanation stability in stance detection
under target variation.
"""

from tce.metrics import (
    cosine_explanation_divergence,
    explanation_shift_score,
    token_flip_rate,
)

__version__ = "1.0.0"

__all__ = [
    "explanation_shift_score",
    "cosine_explanation_divergence",
    "token_flip_rate",
    "__version__",
]
