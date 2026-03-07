"""
preprocess.py — SQL conversion helpers and training-pair builder.
"""

import json
from typing import List, Tuple

from config import DATA_DIR, MAX_TRAIN_SAMPLES

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def convert_sql(sample: dict, table: dict) -> str:
    """Convert a WikiSQL sample + table dict into a plain SQL string."""
    col_names = table["header"]

    select_col = col_names[sample["sql"]["sel"]]
    agg = AGG_OPS[sample["sql"]["agg"]]
    select_part = f"{agg}({select_col})" if agg else select_col

    where_clauses = []
    for col_idx, op_idx, value in sample["sql"]["conds"]:
        column = col_names[col_idx]
        op = COND_OPS[op_idx]
        where_clauses.append(f"{column}{op}'{value}'")

    query = f"SELECT {select_part} FROM table"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    return query


def build_schema(table: dict) -> str:
    """Return a compact schema string, e.g. 'name(text), age(real)'."""
    return ", ".join(
        f"{col}({typ})"
        for col, typ in zip(table["header"], table["types"])
    )


def create_training_examples() -> List[Tuple[str, str]]:
    """
    Read the WikiSQL train split and return a list of (input, target) pairs.
    Input format:  'Schema: <schema>, Question: <question>'
    Target format: plain SQL string
    """
    tables: dict = {}
    with open(f"{DATA_DIR}/train.tables.jsonl") as f_tables:
        for line in f_tables:
            table_obj = json.loads(line)
            tables[table_obj["id"]] = table_obj

    pairs: List[Tuple[str, str]] = []
    with open(f"{DATA_DIR}/train.jsonl") as f_data:
        for line in f_data:
            sample = json.loads(line)
            table = tables[sample["table_id"]]
            schema = build_schema(table)
            sql = convert_sql(sample, table)
            inp = f"Schema: {schema}, Question: {sample['question']}"
            pairs.append((inp, sql))

    return pairs[:MAX_TRAIN_SAMPLES]
