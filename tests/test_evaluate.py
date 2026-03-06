"""Unit tests for src/evaluate.py."""

import pytest

from src.evaluate import (
    clause_set_match,
    evaluate_predictions,
    exact_set_match,
    extract_clauses,
    token_f1,
)


# ---------------------------------------------------------------------------
# extract_clauses
# ---------------------------------------------------------------------------


class TestExtractClauses:
    def test_simple_select(self):
        clauses = extract_clauses("SELECT name FROM singer")
        assert "SELECT" in clauses
        assert "FROM" in clauses
        assert clauses["SELECT"].lower() == "name"
        assert clauses["FROM"].lower() == "singer"

    def test_where_clause(self):
        clauses = extract_clauses("SELECT name FROM singer WHERE age > 30")
        assert "WHERE" in clauses
        assert "age" in clauses["WHERE"].lower()

    def test_group_by_clause(self):
        clauses = extract_clauses("SELECT country, COUNT(*) FROM singer GROUP BY country")
        assert "GROUP BY" in clauses
        assert "country" in clauses["GROUP BY"].lower()

    def test_order_by_clause(self):
        clauses = extract_clauses("SELECT name FROM singer ORDER BY age DESC")
        assert "ORDER BY" in clauses

    def test_empty_string(self):
        assert extract_clauses("") == {}

    def test_no_keywords(self):
        # A string with no recognized clause keywords
        clauses = extract_clauses("1 = 1")
        # Falls back to treating the whole string as SELECT body
        assert clauses.get("SELECT") == "1 = 1"

    def test_case_insensitive(self):
        clauses = extract_clauses("select name from singer")
        assert "SELECT" in clauses
        assert "FROM" in clauses


# ---------------------------------------------------------------------------
# clause_set_match
# ---------------------------------------------------------------------------


class TestClauseSetMatch:
    def test_identical_queries(self):
        sql = "SELECT name FROM singer WHERE age > 30"
        results = clause_set_match(sql, sql)
        assert all(results.values())

    def test_different_select(self):
        pred = "SELECT age FROM singer"
        gold = "SELECT name FROM singer"
        results = clause_set_match(pred, gold)
        assert results.get("SELECT") is False

    def test_matching_select_different_where(self):
        pred = "SELECT name FROM singer WHERE age > 20"
        gold = "SELECT name FROM singer WHERE age > 30"
        results = clause_set_match(pred, gold)
        assert results.get("SELECT") is True
        assert results.get("WHERE") is False


# ---------------------------------------------------------------------------
# exact_set_match
# ---------------------------------------------------------------------------


class TestExactSetMatch:
    def test_identical_queries(self):
        sql = "SELECT name FROM singer"
        assert exact_set_match(sql, sql) is True

    def test_different_queries(self):
        assert exact_set_match("SELECT a FROM t", "SELECT b FROM t") is False

    def test_extra_clause_in_prediction(self):
        pred = "SELECT name FROM singer ORDER BY name"
        gold = "SELECT name FROM singer"
        assert exact_set_match(pred, gold) is False


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------


class TestTokenF1:
    def test_perfect_match(self):
        sql = "SELECT name FROM singer"
        assert token_f1(sql, sql) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("SELECT a", "FROM b") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = token_f1("SELECT name age FROM singer", "SELECT name FROM singer")
        assert 0.0 < f1 < 1.0

    def test_empty_both(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_empty_prediction(self):
        assert token_f1("", "SELECT name FROM singer") == pytest.approx(0.0)

    def test_empty_gold(self):
        assert token_f1("SELECT name FROM singer", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_all_correct(self):
        queries = [
            "SELECT name FROM singer",
            "SELECT count(*) FROM concert",
        ]
        results = evaluate_predictions(queries, queries)
        assert results["exact_match"] == pytest.approx(1.0)
        assert results["token_f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = ["SELECT a FROM x", "SELECT b FROM y"]
        golds = ["SELECT c FROM z", "SELECT d FROM w"]
        results = evaluate_predictions(preds, golds)
        assert results["exact_match"] == pytest.approx(0.0)

    def test_partial_correct(self):
        preds = ["SELECT name FROM singer", "SELECT a FROM x"]
        golds = ["SELECT name FROM singer", "SELECT b FROM y"]
        results = evaluate_predictions(preds, golds)
        assert results["exact_match"] == pytest.approx(0.5)

    def test_num_examples(self):
        preds = ["SELECT a FROM x"] * 5
        golds = ["SELECT a FROM x"] * 5
        results = evaluate_predictions(preds, golds)
        assert results["num_examples"] == 5

    def test_empty_lists(self):
        results = evaluate_predictions([], [])
        assert results["num_examples"] == 0
        assert results["exact_match"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_predictions(["SELECT a"], ["SELECT a", "SELECT b"])

    def test_per_clause_match_keys(self):
        preds = ["SELECT name FROM singer WHERE age > 30"]
        golds = ["SELECT name FROM singer WHERE age > 30"]
        results = evaluate_predictions(preds, golds)
        assert "SELECT" in results["per_clause_match"]
        assert "FROM" in results["per_clause_match"]
        assert "WHERE" in results["per_clause_match"]

    def test_normalization_effect(self):
        # Queries that differ only in whitespace should be considered equal
        pred = "SELECT  name  FROM  singer"
        gold = "SELECT name FROM singer"
        results = evaluate_predictions([pred], [gold])
        assert results["exact_match"] == pytest.approx(1.0)
