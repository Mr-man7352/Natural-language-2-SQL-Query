"""Structured evaluation for NL2SQL predictions.

Implements:
* **Exact Set Match** – component-wise matching of SQL clauses (SELECT, FROM,
  WHERE, GROUP BY, ORDER BY, HAVING, LIMIT) following the Spider evaluation
  methodology.
* **Token-level F1** – lightweight token overlap metric.
* ``evaluate_predictions`` – convenience function that computes all metrics
  for a list of (prediction, gold) pairs and returns a summary dictionary.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from src.utils import get_logger, normalize_sql

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL clause extraction
# ---------------------------------------------------------------------------

# Ordered list of major SQL clauses (higher-priority clauses come first so
# that the splitting regex works correctly).
_CLAUSE_KEYWORDS: list[str] = [
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "LIMIT",
    "INTERSECT",
    "UNION",
    "EXCEPT",
]

_CLAUSE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _CLAUSE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def extract_clauses(sql: str) -> dict[str, str]:
    """Split a SQL query into its constituent clauses.

    Returns a mapping from clause keyword (upper-cased) to the clause body
    (stripped).  Only clauses that are present in *sql* are included.
    """
    sql = sql.strip()
    if not sql:
        return {}

    # Find all clause boundaries
    boundaries: list[tuple[int, str]] = []
    for match in _CLAUSE_PATTERN.finditer(sql):
        boundaries.append((match.start(), match.group(0).upper()))

    if not boundaries:
        return {"SELECT": sql}

    clauses: dict[str, str] = {}
    for i, (start, keyword) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(sql)
        body = sql[start + len(keyword) : end].strip()
        clauses[keyword] = body

    return clauses


def _tokenize_clause(clause_body: str) -> list[str]:
    """Split a clause body into comparable tokens.

    Tokens are lower-cased and punctuation is stripped so that trivial
    formatting differences are ignored.
    """
    # Remove parentheses used for sub-queries or function calls for a
    # lightweight comparison
    body = clause_body.lower()
    body = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", body)
    return [t for t in body.split() if t]


def clause_set_match(predicted_sql: str, gold_sql: str) -> dict[str, bool]:
    """Compute per-clause set match between *predicted_sql* and *gold_sql*.

    Returns a dict mapping each clause keyword that appears in either query to
    a boolean indicating whether the predicted clause matches the gold clause.
    """
    pred_clauses = extract_clauses(predicted_sql)
    gold_clauses = extract_clauses(gold_sql)

    all_keywords = set(pred_clauses) | set(gold_clauses)
    results: dict[str, bool] = {}
    for kw in all_keywords:
        pred_tokens = set(_tokenize_clause(pred_clauses.get(kw, "")))
        gold_tokens = set(_tokenize_clause(gold_clauses.get(kw, "")))
        results[kw] = pred_tokens == gold_tokens
    return results


# ---------------------------------------------------------------------------
# Exact set match (full query)
# ---------------------------------------------------------------------------


def exact_set_match(predicted_sql: str, gold_sql: str) -> bool:
    """Return ``True`` if every SQL clause in *predicted_sql* matches the
    corresponding clause in *gold_sql* (set-based comparison)."""
    per_clause = clause_set_match(predicted_sql, gold_sql)
    return all(per_clause.values())


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------


def token_f1(predicted_sql: str, gold_sql: str) -> float:
    """Compute token-level F1 between two SQL strings.

    Tokens are lower-cased; punctuation is stripped before comparison.
    Returns a float in ``[0, 1]``.
    """

    def _tokens(sql: str) -> list[str]:
        sql = sql.lower()
        sql = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", sql)
        return sql.split()

    pred_tokens = _tokens(predicted_sql)
    gold_tokens = _tokens(gold_sql)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: list[str],
    gold_queries: list[str],
) -> dict[str, Any]:
    """Evaluate a list of predicted SQL queries against gold queries.

    Parameters
    ----------
    predictions:
        List of predicted SQL strings (one per example).
    gold_queries:
        List of gold SQL strings (one per example).

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``exact_match`` – proportion of predictions with an exact string
          match after normalization.
        * ``exact_set_match`` – proportion of predictions with a full
          clause-level set match.
        * ``token_f1`` – average token-level F1.
        * ``per_clause_match`` – dict mapping each clause keyword to the
          fraction of examples where that clause matched.
        * ``num_examples`` – total number of examples evaluated.
    """
    if len(predictions) != len(gold_queries):
        raise ValueError(
            f"predictions and gold_queries must have the same length, "
            f"got {len(predictions)} vs {len(gold_queries)}."
        )

    n = len(predictions)
    if n == 0:
        return {
            "exact_match": 0.0,
            "exact_set_match": 0.0,
            "token_f1": 0.0,
            "per_clause_match": {},
            "num_examples": 0,
        }

    exact_match_count = 0
    exact_set_match_count = 0
    token_f1_total = 0.0
    clause_match_counts: dict[str, int] = {}
    clause_totals: dict[str, int] = {}

    for pred, gold in zip(predictions, gold_queries):
        norm_pred = normalize_sql(pred)
        norm_gold = normalize_sql(gold)

        # Exact string match (post-normalization)
        if norm_pred == norm_gold:
            exact_match_count += 1

        # Clause-level set match
        per_clause = clause_set_match(pred, gold)
        if all(per_clause.values()):
            exact_set_match_count += 1
        for kw, matched in per_clause.items():
            clause_match_counts[kw] = clause_match_counts.get(kw, 0) + int(matched)
            clause_totals[kw] = clause_totals.get(kw, 0) + 1

        # Token F1
        token_f1_total += token_f1(pred, gold)

    per_clause_match = {
        kw: clause_match_counts[kw] / clause_totals[kw]
        for kw in clause_totals
    }

    results = {
        "exact_match": exact_match_count / n,
        "exact_set_match": exact_set_match_count / n,
        "token_f1": token_f1_total / n,
        "per_clause_match": per_clause_match,
        "num_examples": n,
    }

    logger.info(
        "Evaluation results: exact_match=%.4f, exact_set_match=%.4f, token_f1=%.4f "
        "(n=%d)",
        results["exact_match"],
        results["exact_set_match"],
        results["token_f1"],
        n,
    )
    return results
