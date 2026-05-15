"""Model loading wrapper around HuggingFace Transformers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Container for tokenizer + model + device + label mapping."""

    tokenizer: Any
    model: Any
    device: str
    label_names: list[str]
    num_labels: int

    def __post_init__(self):
        if not self.label_names:
            self.label_names = [f"LABEL_{i}" for i in range(self.num_labels)]


def load_model(
    name_or_path: str,
    device: str = "cpu",
    label_names: list[str] | None = None,
) -> ModelBundle:
    """Load a tokenizer + sequence-classification model from a HF hub ID or local path.

    Args:
        name_or_path: HuggingFace model ID (e.g. ``"bert-base-uncased"``) or local dir.
        device: ``"cpu"`` or ``"cuda"``.
        label_names: Stance label order, e.g. ``["AGAINST", "NONE", "FAVOR"]``.
            If None, falls back to model config or generic LABEL_i names.

    Returns:
        A ``ModelBundle`` with everything downstream code needs.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading tokenizer from %s", name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    logger.info("Loading model from %s", name_or_path)
    num_labels = len(label_names) if label_names else 3
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()

    if label_names is None:
        id2label = getattr(model.config, "id2label", None)
        if id2label and len(id2label) == model.config.num_labels:
            label_names = [id2label[i] for i in sorted(id2label)]
        else:
            label_names = [f"LABEL_{i}" for i in range(model.config.num_labels)]

    logger.info("Model loaded on %s with labels: %s", device, label_names)
    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        label_names=list(label_names),
        num_labels=model.config.num_labels,
    )
