"""Unit tests for input formatting and target parsing."""

import pandas as pd

from tce.preprocessing import (
    format_input,
    pad_similar_targets,
    parse_similar_targets,
)


class TestFormatInput:
    def test_concatenates_with_sep_and_target(self):
        result = format_input("Climate is changing.", "climate change")
        assert result == "Climate is changing. [SEP] Target: climate change"

    def test_strips_whitespace(self):
        result = format_input("  hello  ", "  target  ")
        assert "hello" in result and "target" in result


class TestParseSimilarTargets:
    def test_python_list_literal(self):
        s = "['climate change', 'global warming', 'environment']"
        assert parse_similar_targets(s) == ["climate change", "global warming", "environment"]

    def test_comma_separated_fallback(self):
        s = "tax reform, fiscal policy, taxes"
        assert parse_similar_targets(s) == ["tax reform", "fiscal policy", "taxes"]

    def test_clips_to_n(self):
        s = "['a', 'b', 'c', 'd', 'e']"
        assert parse_similar_targets(s, n=2) == ["a", "b"]

    def test_handles_empty_and_nan(self):
        assert parse_similar_targets("") == []
        assert parse_similar_targets(pd.NA) == []
        assert parse_similar_targets(None) == []


class TestPadSimilarTargets:
    def test_pads_short_list(self):
        assert pad_similar_targets(["a", "b"], n=4, fill="x") == ["a", "b", "x", "x"]

    def test_truncates_long_list(self):
        assert pad_similar_targets(["a", "b", "c", "d"], n=2) == ["a", "b"]

    def test_exact_match(self):
        assert pad_similar_targets(["a", "b", "c"], n=3) == ["a", "b", "c"]
