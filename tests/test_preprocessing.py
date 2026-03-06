"""Unit tests for src/data_preprocessing.py."""

import json
import os
import tempfile

import pytest

from src.data_preprocessing import (
    build_prompt,
    format_schema_prompt,
    load_spider_examples,
    load_tables,
    make_tokenize_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TABLES = [
    {
        "db_id": "concert_singer",
        "table_names_original": ["singer", "concert"],
        "column_names_original": [
            [-1, "*"],
            [0, "Singer_ID"],
            [0, "Name"],
            [1, "Concert_ID"],
            [1, "Theme"],
        ],
        "column_types": ["text", "number", "text", "number", "text"],
        "primary_keys": [1, 3],
        "foreign_keys": [],
    }
]

SAMPLE_EXAMPLES = [
    {
        "db_id": "concert_singer",
        "question": "How many singers are there?",
        "query": "SELECT count(*) FROM singer",
    },
    {
        "db_id": "concert_singer",
        "question": "List all concert themes.",
        "query": "SELECT Theme FROM concert",
    },
]


@pytest.fixture()
def tables_file(tmp_path):
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(SAMPLE_TABLES), encoding="utf-8")
    return str(path)


@pytest.fixture()
def examples_file(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps(SAMPLE_EXAMPLES), encoding="utf-8")
    return str(path)


@pytest.fixture()
def schemas(tables_file):
    return load_tables(tables_file)


# ---------------------------------------------------------------------------
# load_tables
# ---------------------------------------------------------------------------


class TestLoadTables:
    def test_keys_present(self, schemas):
        assert "concert_singer" in schemas

    def test_table_names(self, schemas):
        assert schemas["concert_singer"]["table_names"] == ["singer", "concert"]

    def test_columns_loaded(self, schemas):
        cols_singer = schemas["concert_singer"]["columns"][0]
        col_names = [c[0] for c in cols_singer]
        assert "singer_id" in col_names
        assert "name" in col_names

    def test_wildcard_column_excluded(self, schemas):
        # The wildcard (*) column at table_index -1 must not appear in any table
        for cols in schemas["concert_singer"]["columns"].values():
            assert "*" not in [c[0] for c in cols]

    def test_primary_keys_preserved(self, schemas):
        assert schemas["concert_singer"]["primary_keys"] == [1, 3]


# ---------------------------------------------------------------------------
# format_schema_prompt
# ---------------------------------------------------------------------------


class TestFormatSchemaPrompt:
    def test_starts_with_tables(self, schemas):
        prompt = format_schema_prompt(schemas["concert_singer"])
        assert prompt.startswith("tables: ")

    def test_contains_table_names(self, schemas):
        prompt = format_schema_prompt(schemas["concert_singer"])
        assert "singer" in prompt
        assert "concert" in prompt

    def test_contains_column_names(self, schemas):
        prompt = format_schema_prompt(schemas["concert_singer"])
        assert "singer_id" in prompt
        assert "theme" in prompt

    def test_contains_column_types(self, schemas):
        prompt = format_schema_prompt(schemas["concert_singer"])
        assert "number" in prompt or "text" in prompt

    def test_pipe_separator_between_tables(self, schemas):
        prompt = format_schema_prompt(schemas["concert_singer"])
        assert " | " in prompt


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_starts_with_translate(self, schemas):
        prompt = build_prompt("How many singers?", schemas["concert_singer"])
        assert prompt.startswith("translate English to SQL:")

    def test_prompt_contains_question(self, schemas):
        question = "How many singers?"
        prompt = build_prompt(question, schemas["concert_singer"])
        assert question in prompt

    def test_prompt_contains_schema(self, schemas):
        prompt = build_prompt("q?", schemas["concert_singer"])
        assert "tables:" in prompt
        assert "singer" in prompt


# ---------------------------------------------------------------------------
# load_spider_examples
# ---------------------------------------------------------------------------


class TestLoadSpiderExamples:
    def test_loads_correct_count(self, examples_file, schemas):
        examples = load_spider_examples(examples_file, schemas)
        assert len(examples) == len(SAMPLE_EXAMPLES)

    def test_example_has_required_keys(self, examples_file, schemas):
        examples = load_spider_examples(examples_file, schemas)
        for ex in examples:
            assert "input" in ex
            assert "target" in ex
            assert "db_id" in ex
            assert "question" in ex

    def test_target_matches_gold_sql(self, examples_file, schemas):
        examples = load_spider_examples(examples_file, schemas)
        gold_queries = [e["query"] for e in SAMPLE_EXAMPLES]
        for ex, gold in zip(examples, gold_queries):
            assert ex["target"] == gold

    def test_input_contains_question(self, examples_file, schemas):
        examples = load_spider_examples(examples_file, schemas)
        for ex, raw in zip(examples, SAMPLE_EXAMPLES):
            assert raw["question"] in ex["input"]

    def test_unknown_db_skipped(self, tmp_path, schemas):
        bad_examples = [{"db_id": "unknown_db", "question": "q", "query": "SELECT 1"}]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_examples), encoding="utf-8")
        result = load_spider_examples(str(path), schemas)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# make_tokenize_fn (without heavy model weights – uses a mock tokenizer)
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer stub for testing make_tokenize_fn."""

    pad_token_id = 0

    def __call__(self, texts, max_length=512, padding=None, truncation=None):
        ids = [[1, 2, 3, 0] for _ in texts]
        return {"input_ids": ids, "attention_mask": [[1, 1, 1, 0] for _ in texts]}

    def as_target_tokenizer(self):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            yield

        return _ctx()


class TestMakeTokenizeFn:
    def test_output_has_expected_keys(self):
        tokenizer = MockTokenizer()
        fn = make_tokenize_fn(tokenizer)
        batch = {
            "input": ["translate English to SQL: q | tables: t: c (text)"],
            "target": ["SELECT c FROM t"],
        }
        result = fn(batch)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_padding_replaced_with_minus_100(self):
        tokenizer = MockTokenizer()
        fn = make_tokenize_fn(tokenizer)
        batch = {
            "input": ["translate English to SQL: q | tables: t: c (text)"],
            "target": ["SELECT c FROM t"],
        }
        result = fn(batch)
        labels = result["labels"][0]
        # pad_token_id is 0 – must be replaced with -100
        assert 0 not in labels
        assert -100 in labels
