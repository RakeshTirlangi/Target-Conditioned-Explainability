"""Configuration loading with YAML inheritance (`_base_`)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict-like configuration with attribute access and YAML inheritance."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins on conflict)."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    """Load a YAML config file. Resolves `_base_` chains relative to the file."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return Config(raw)

    base_path = (path.parent / base_ref).resolve()
    base_cfg = load_config(base_path)
    merged = _deep_merge(base_cfg, raw)
    return Config(merged)


def _to_plain(obj: Any) -> Any:
    """Recursively convert Config (dict subclass) and Paths to YAML-safe types."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_config(cfg: Config | dict, path: str | Path) -> None:
    """Persist a config alongside experiment outputs for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False, default_flow_style=False)
