"""Test YAML config loading with `_base_` inheritance."""

import tempfile
from pathlib import Path

import pytest
import yaml

from tce.config import load_config


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


class TestConfigLoading:
    def test_loads_flat_yaml(self, tmp_path):
        cfg_path = tmp_path / "x.yaml"
        _write_yaml(cfg_path, {"a": 1, "nested": {"b": 2}})
        cfg = load_config(cfg_path)
        assert cfg.a == 1
        assert cfg.nested.b == 2

    def test_base_inheritance_merges(self, tmp_path):
        base = tmp_path / "base.yaml"
        child = tmp_path / "child.yaml"
        _write_yaml(base, {"a": 1, "b": {"c": 10, "d": 20}})
        _write_yaml(child, {"_base_": "base.yaml", "b": {"d": 99}})
        cfg = load_config(child)
        assert cfg.a == 1
        assert cfg.b.c == 10  # inherited
        assert cfg.b.d == 99  # overridden

    def test_missing_key_raises_attribute_error(self, tmp_path):
        cfg_path = tmp_path / "x.yaml"
        _write_yaml(cfg_path, {"a": 1})
        cfg = load_config(cfg_path)
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent

    def test_save_then_reload_round_trip(self, tmp_path):
        """save_config must handle Config (dict subclass) and Path values."""
        from tce.config import save_config

        original_path = tmp_path / "orig.yaml"
        _write_yaml(original_path, {
            "experiment": {"name": "exp1", "seed": 7},
            "model": {"name": "bert"},
            "nested": {"sensitivity": {"k_values": [3, 5, 10]}},
        })
        cfg = load_config(original_path)
        # Touch a nested attribute so it's wrapped as a Config subclass.
        _ = cfg.nested.sensitivity.k_values

        out_path = tmp_path / "resolved.yaml"
        save_config(cfg, out_path)

        reloaded = load_config(out_path)
        assert reloaded.experiment.name == "exp1"
        assert reloaded.experiment.seed == 7
        assert reloaded.nested.sensitivity.k_values == [3, 5, 10]
