"""Batched softmax-probability inference utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tce.model import ModelBundle


def predict_proba(
    texts: Sequence[str],
    bundle: ModelBundle,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    """Return softmax probability matrix of shape ``(len(texts), num_labels)``.

    Empty/invalid inputs are replaced with a neutral placeholder string so
    tokenization never fails on the SHAP perturbation path.
    """
    import torch

    cleaned = [
        t if isinstance(t, str) and t.strip() else "neutral" for t in texts
    ]

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            encoded = bundle.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
            logits = bundle.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def predict_label(probs: np.ndarray, label_names: Sequence[str]) -> list[str]:
    """Argmax over the class axis, mapped to label names."""
    idx = probs.argmax(axis=-1)
    return [label_names[i] for i in idx]


def confidence_margin(probs: np.ndarray) -> np.ndarray:
    """Top-prob minus second-prob — a calibrated certainty signal."""
    sorted_probs = np.sort(probs, axis=-1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]
