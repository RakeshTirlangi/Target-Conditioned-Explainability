# metrics.py

import numpy as np

def _align(v1, v2):
    """Pad vectors to same length."""
    L = max(len(v1), len(v2))
    return np.pad(v1, (0, L - len(v1))), np.pad(v2, (0, L - len(v2)))


def explanation_shift_score(phi_i, phi_j):
    """ESS: mean absolute difference."""
    a, b = _align(phi_i, phi_j)
    return float(np.mean(np.abs(a - b)))


def cosine_explanation_divergence(phi_i, phi_j):
    """CED: 1 - cosine similarity."""
    a, b = _align(phi_i, phi_j)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1 - np.dot(a, b) / (denom + 1e-8))


def top_k_flip_rate(phi_i, phi_j, k=5):
    """TFR: difference in top-k tokens."""
    a, b = _align(phi_i, phi_j)
    
    k = min(k, len(a))
    
    top_a = set(np.argsort(np.abs(a))[-k:])
    top_b = set(np.argsort(np.abs(b))[-k:])
    
    return float(1 - len(top_a & top_b) / k)