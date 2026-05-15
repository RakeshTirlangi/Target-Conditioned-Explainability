"""End-to-end smoke test on a 3-row synthetic dataset.

This test downloads a small model from HuggingFace, so requires network
access on first run. To skip when offline, run pytest with ``-k "not smoke"``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow forcing the smoke test to skip in restricted environments.
_SKIP = os.environ.get("TCE_SKIP_SMOKE", "").lower() in {"1", "true", "yes"}


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cleaned_tweet": "Climate is changing rapidly.",
                "GT Target": "climate change",
                "GT Stance": "FAVOR",
                "similar targets": "['global warming', 'environmental policy', 'climate crisis']",
            },
            {
                "cleaned_tweet": "I oppose this idea completely.",
                "GT Target": "tax reform",
                "GT Stance": "AGAINST",
                "similar targets": "['fiscal policy', 'taxation', 'taxes']",
            },
            {
                "cleaned_tweet": "I am not sure either way.",
                "GT Target": "atheism",
                "GT Stance": "NONE",
                "similar targets": "['religion', 'belief systems', 'faith']",
            },
        ]
    )


@pytest.mark.skipif(_SKIP, reason="TCE_SKIP_SMOKE=1 set")
def test_pipeline_end_to_end(tiny_df, tmp_path):
    """Pipeline runs to completion and produces expected columns + ranges."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("shap")

    from tce.model import load_model
    from tce.pipeline import run_tce, save_results
    from tce.utils import resolve_device, set_seed

    set_seed(0)
    device = resolve_device("auto")
    bundle = load_model(
        "sshleifer/tiny-distilbert-base-cased-distilled-squad",
        device=device,
        label_names=["AGAINST", "NONE", "FAVOR"],
    )

    pairs = run_tce(
        tiny_df, bundle,
        text_col="cleaned_tweet",
        gt_target_col="GT Target",
        gt_stance_col="GT Stance",
        similar_targets_col="similar targets",
        n_similar=3,
        max_evals=10,
        max_length=64,
        batch_size=4,
        top_k=5,
        n_background_candidates=2,
        seed=0,
        progress=False,
    )

    assert len(pairs) > 0
    for col in ("ESS", "CED", "TFR", "gt_target", "similar_target",
                "gt_prediction", "sim_prediction"):
        assert col in pairs.columns

    # All metrics are bounded
    assert (pairs["ESS"] >= 0).all()
    assert (pairs["CED"] >= 0).all() and (pairs["CED"] <= 2.0 + 1e-6).all()
    assert (pairs["TFR"] >= 0).all() and (pairs["TFR"] <= 1.0 + 1e-6).all()

    paths = save_results(pairs, tmp_path, prefix="smoke")
    for p in paths.values():
        assert Path(p).exists()
