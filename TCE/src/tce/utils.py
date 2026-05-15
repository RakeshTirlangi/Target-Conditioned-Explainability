"""Utilities: seeding, device selection, output directory management."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """Return 'cuda' when requested/available, else 'cpu'."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = False,
    log_file: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Configure root logger. Idempotent — safe to call multiple times."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_to_file and log_file:
        log_path = Path(output_dir) / log_file if output_dir else Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
