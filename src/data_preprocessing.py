"""Dataset preprocessing for the NL2SQL pipeline.

Supports the Spider benchmark dataset.  Each example is converted into a
schema-aware prompt that is fed to the T5 model.

Prompt format
-------------
``translate English to SQL: <question> | tables: <table>: <col1>, <col2> | ...``

The target is the gold SQL query.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def load_tables(tables_file: str) -> dict[str, dict[str, Any]]:
    """Load Spider ``tables.json`` and return a mapping from *db_id* to its
    schema dictionary.

    Each schema dictionary has the keys:
    * ``table_names`` – list of lower-cased table name strings
    * ``columns`` – mapping of table index → list of (column_name, column_type) tuples
    * ``primary_keys`` – list of global column indices that are primary keys
    * ``foreign_keys`` – list of (child_col_index, parent_col_index) pairs
    """
    with open(tables_file, "r", encoding="utf-8") as fh:
        raw_tables = json.load(fh)

    schemas: dict[str, dict[str, Any]] = {}
    for db in raw_tables:
        db_id: str = db["db_id"]
        table_names: list[str] = [t.lower() for t in db["table_names_original"]]

        # column_names_original is a list of [table_index, column_name]
        columns: dict[int, list[tuple[str, str]]] = {
            i: [] for i in range(len(table_names))
        }
        col_types: list[str] = db.get("column_types", [])
        for col_idx, (tbl_idx, col_name) in enumerate(db["column_names_original"]):
            if tbl_idx < 0:
                # wildcard column (*) – skip
                continue
            col_type = col_types[col_idx] if col_idx < len(col_types) else "text"
            columns[tbl_idx].append((col_name.lower(), col_type.lower()))

        schemas[db_id] = {
            "table_names": table_names,
            "columns": columns,
            "primary_keys": db.get("primary_keys", []),
            "foreign_keys": db.get("foreign_keys", []),
        }
    return schemas


def format_schema_prompt(schema: dict[str, Any]) -> str:
    """Convert a schema dictionary into the schema portion of the prompt.

    Example output::

        tables: singer: singer_id (number), name (text) | concert: concert_id (number), theme (text)
    """
    parts: list[str] = []
    for tbl_idx, tbl_name in enumerate(schema["table_names"]):
        cols = schema["columns"].get(tbl_idx, [])
        col_strs = [f"{col_name} ({col_type})" for col_name, col_type in cols]
        parts.append(f"{tbl_name}: {', '.join(col_strs)}" if col_strs else tbl_name)
    return "tables: " + " | ".join(parts)


def build_prompt(question: str, schema: dict[str, Any]) -> str:
    """Build the full T5 input prompt for a single NL2SQL example.

    Format::

        translate English to SQL: <question> | tables: <schema>
    """
    schema_str = format_schema_prompt(schema)
    return f"translate English to SQL: {question.strip()} | {schema_str}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_spider_examples(
    data_file: str,
    schemas: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Load a Spider train/dev JSON file and return a list of preprocessed examples.

    Each returned dictionary contains:
    * ``input`` – the formatted prompt (question + schema)
    * ``target`` – the gold SQL query
    * ``db_id`` – the database identifier
    * ``question`` – the original natural-language question
    """
    with open(data_file, "r", encoding="utf-8") as fh:
        raw_examples = json.load(fh)

    examples: list[dict[str, str]] = []
    missing_schemas = 0

    for item in raw_examples:
        db_id: str = item["db_id"]
        question: str = item["question"]
        gold_sql: str = item["query"]

        if db_id not in schemas:
            logger.warning("Schema not found for db_id '%s' – skipping.", db_id)
            missing_schemas += 1
            continue

        prompt = build_prompt(question, schemas[db_id])
        examples.append(
            {
                "input": prompt,
                "target": gold_sql,
                "db_id": db_id,
                "question": question,
            }
        )

    if missing_schemas:
        logger.warning(
            "Skipped %d examples due to missing schemas.", missing_schemas
        )

    logger.info("Loaded %d examples from '%s'.", len(examples), data_file)
    return examples


# ---------------------------------------------------------------------------
# Tokenisation helpers (Hugging Face datasets-compatible)
# ---------------------------------------------------------------------------


def make_tokenize_fn(
    tokenizer,
    max_input_length: int = 512,
    max_target_length: int = 256,
):
    """Return a function that tokenizes a batch of NL2SQL examples.

    The returned function can be passed directly to
    ``datasets.Dataset.map()``.
    """

    def tokenize_fn(batch: dict[str, list]) -> dict[str, list]:
        model_inputs = tokenizer(
            batch["input"],
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=max_target_length,
                padding="max_length",
                truncation=True,
            )

        # Replace padding token ids with -100 so they are ignored in the loss
        label_ids = [
            [
                token_id if token_id != tokenizer.pad_token_id else -100
                for token_id in ids
            ]
            for ids in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize_fn


def preprocess_dataset(
    train_file: str,
    dev_file: str,
    tables_file: str,
    tokenizer,
    max_input_length: int = 512,
    max_target_length: int = 256,
    cache_dir: str | None = None,
):
    """End-to-end preprocessing pipeline.

    Loads raw Spider files, builds schema-aware prompts, tokenizes them, and
    returns Hugging Face ``Dataset`` objects ready for training/evaluation.

    Parameters
    ----------
    train_file:
        Path to the Spider ``train_spider.json`` (or combined train file).
    dev_file:
        Path to the Spider ``dev.json``.
    tables_file:
        Path to the Spider ``tables.json``.
    tokenizer:
        A Hugging Face tokenizer (e.g. ``T5Tokenizer``).
    max_input_length:
        Maximum number of tokens for the input sequence.
    max_target_length:
        Maximum number of tokens for the target (SQL) sequence.
    cache_dir:
        Optional directory for caching the tokenized dataset.

    Returns
    -------
    tuple[Dataset, Dataset]
        ``(train_dataset, dev_dataset)``
    """
    from datasets import Dataset

    schemas = load_tables(tables_file)

    train_examples = load_spider_examples(train_file, schemas)
    dev_examples = load_spider_examples(dev_file, schemas)

    train_ds = Dataset.from_list(train_examples)
    dev_ds = Dataset.from_list(dev_examples)

    tokenize_fn = make_tokenize_fn(tokenizer, max_input_length, max_target_length)

    kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": ["input", "target", "db_id", "question"],
    }
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    train_ds = train_ds.map(tokenize_fn, **kwargs)
    dev_ds = dev_ds.map(tokenize_fn, **kwargs)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, dev_ds
