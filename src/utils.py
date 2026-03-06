"""Utilities for the NL2SQL pipeline."""

import logging
import os
import random

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a dictionary."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalize_sql(sql: str) -> str:
    """Normalize a SQL string for comparison.

    Lowercases keywords, collapses whitespace, and strips leading/trailing
    whitespace so that trivial formatting differences do not affect evaluation.
    """
    import sqlparse
    from sqlparse.sql import Token
    from sqlparse.tokens import Keyword, DML

    if not sql or not sql.strip():
        return ""

    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return sql.strip().lower()

    flat_tokens: list[str] = []
    for statement in parsed:
        for token in statement.flatten():
            value = token.value
            if token.ttype in (Keyword, DML) or (
                token.ttype is not None and token.ttype in Keyword
            ):
                value = value.upper()
            flat_tokens.append(value)

    normalized = " ".join(flat_tokens)
    # collapse multiple spaces
    import re

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def ensure_dir(path: str) -> None:
    """Create *path* (and all intermediate directories) if it does not exist."""
    os.makedirs(path, exist_ok=True)
