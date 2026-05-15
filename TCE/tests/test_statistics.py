"""Unit tests for statistical analysis helpers."""

import numpy as np
import pandas as pd
import pytest

from tce.statistics import (
    add_error_column,
    bootstrap_ci,
    correlation_analysis,
    paired_wilcoxon,
    pairwise_significance,
)


class TestBootstrapCI:
    def test_returns_three_floats(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0.5, 0.1, size=200).tolist()
        mean, lo, hi = bootstrap_ci(vals, n_bootstrap=200, ci=0.95, seed=0)
        assert lo <= mean <= hi
        assert 0.45 < mean < 0.55

    def test_empty_input_returns_nan(self):
        mean, lo, hi = bootstrap_ci([])
        assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


class TestPairedWilcoxon:
    def test_identical_inputs_yield_p_one(self):
        a = [0.1, 0.2, 0.3]
        stat, p = paired_wilcoxon(a, a)
        assert p == 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([1, 2], [1, 2, 3])

    def test_significant_difference(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.5, 0.1, 100)
        b = a + 0.2
        _, p = paired_wilcoxon(a.tolist(), b.tolist())
        assert p < 0.001


class TestPairwiseSignificance:
    def test_two_models_one_row(self):
        scores = {
            "A": [0.1, 0.2, 0.3, 0.4, 0.5],
            "B": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
        df = pairwise_significance(scores, bonferroni=False)
        assert len(df) == 1
        assert df.iloc[0]["model_a"] == "A"
        assert df.iloc[0]["model_b"] == "B"


class TestAddErrorColumn:
    def test_basic(self):
        df = pd.DataFrame({
            "gt_stance": ["AGAINST", "FAVOR", "NONE"],
            "gt_prediction": ["AGAINST", "AGAINST", "NONE"],
        })
        out = add_error_column(df)
        assert out["is_error"].tolist() == [0, 1, 0]


class TestCorrelationAnalysis:
    def test_returns_rows_for_each_metric(self):
        df = pd.DataFrame({
            "ESS": [0.1, 0.2, 0.3, 0.4],
            "CED": [0.2, 0.3, 0.4, 0.5],
            "TFR": [0.0, 0.5, 0.5, 1.0],
            "is_error": [0, 0, 1, 1],
            "gt_confidence": [0.9, 0.8, 0.3, 0.1],
        })
        corr = correlation_analysis(df)
        assert len(corr) > 0
        assert set(corr["metric"]).issubset({"ESS", "CED", "TFR"})
