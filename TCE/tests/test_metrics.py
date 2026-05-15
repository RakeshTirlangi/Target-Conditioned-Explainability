"""Unit tests for TCE metric definitions."""

import numpy as np
import pytest

from tce.metrics import (
    compute_all,
    cosine_explanation_divergence,
    explanation_shift_score,
    token_flip_rate,
)


class TestESS:
    def test_identical_vectors_yield_zero(self):
        v = np.array([0.1, 0.2, -0.3, 0.4])
        assert explanation_shift_score(v, v) == pytest.approx(0.0)

    def test_known_difference(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        # mean |1-0|, |0-1| = 1.0
        assert explanation_shift_score(a, b) == pytest.approx(1.0)

    def test_pads_shorter_vector(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        # After padding b -> [1, 2, 0]; mean(|0|, |0|, |3|) = 1.0
        assert explanation_shift_score(a, b) == pytest.approx(1.0)


class TestCED:
    def test_identical_direction_yields_zero(self):
        v = np.array([0.5, 0.5, 0.0])
        assert cosine_explanation_divergence(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_direction_yields_two(self):
        v = np.array([1.0, 0.0])
        w = np.array([-1.0, 0.0])
        assert cosine_explanation_divergence(v, w) == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_yields_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_explanation_divergence(a, b) == pytest.approx(1.0, abs=1e-6)


class TestTFR:
    def test_identical_top_k_yields_zero(self):
        v = np.array([0.1, 0.9, 0.5, 0.3])
        assert token_flip_rate(v, v, k=2) == pytest.approx(0.0)

    def test_disjoint_top_k_yields_one(self):
        # Top-2 of a is {2, 3}; top-2 of b is {0, 1} (by |value|)
        a = np.array([0.0, 0.0, 0.9, 0.8])
        b = np.array([0.9, 0.8, 0.0, 0.0])
        assert token_flip_rate(a, b, k=2) == pytest.approx(1.0)

    def test_k_clipped_when_larger_than_vector(self):
        v = np.array([0.1, 0.2])
        # Vector has length 2 so effective k=2
        assert 0.0 <= token_flip_rate(v, v, k=10) <= 1.0


class TestComputeAll:
    def test_returns_all_three_metrics(self):
        a = np.array([0.5, 0.1, -0.2, 0.3])
        b = np.array([0.4, 0.2, -0.1, 0.5])
        result = compute_all(a, b, k=2)
        assert set(result.keys()) == {"ESS", "CED", "TFR"}
        assert all(isinstance(v, float) for v in result.values())

    def test_empty_vectors_are_safe(self):
        a = np.array([])
        b = np.array([])
        result = compute_all(a, b)
        assert result["ESS"] == 0.0
        assert result["TFR"] == 0.0
