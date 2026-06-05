"""Tests for the self-consistency module.

Covers entropy computation, per-row consistency, and aggregate statistics.
Uses a small synthetic dataset to avoid external dependencies.
"""

import math
import tempfile
from pathlib import Path

import pytest

from sentiment_benchmark.self_consistency import (
    RowSelfConsistency,
    SelfConsistencyResult,
    compute_row_consistency,
    compute_self_consistency,
    entropy_from_counts,
    majority_fraction,
    majority_label,
)


# ---------------------------------------------------------------------------
# entropy_from_counts
# ---------------------------------------------------------------------------


def test_entropy_empty_returns_zero() -> None:
    assert entropy_from_counts({}) == 0.0


def test_entropy_single_label() -> None:
    # Only one category => entropy = 0 (since k=1, normalization gives 0/0=0? No, k>1 check)
    assert entropy_from_counts({"positive": 5}) == 0.0


def test_entropy_uniform_distribution() -> None:
    # 3 labels equal => normalized entropy = 1.0
    ent = entropy_from_counts({"positive": 4, "negative": 4, "neutral": 4})
    assert math.isclose(ent, 1.0, abs_tol=1e-4)


def test_entropy_skewed_distribution() -> None:
    # 8/10 on one label => low entropy
    ent = entropy_from_counts({"positive": 8, "negative": 1, "neutral": 1}, normalize=True)
    assert 0.0 < ent < 0.8  # not uniform, not 0


def test_entropy_un_normalized() -> None:
    # Without normalization, values > 1 are possible for >2 categories
    ent = entropy_from_counts({"positive": 2, "negative": 2, "neutral": 2}, normalize=False)
    # log2(3) ≈ 1.585 with uniform distribution
    assert math.isclose(ent, math.log2(3), abs_tol=1e-4)


def test_entropy_partial_valid() -> None:
    counts = {"positive": 5, "negative": 3}
    ent = entropy_from_counts(counts, normalize=True)
    assert 0.0 < ent <= 1.0


# ---------------------------------------------------------------------------
# majority_label / majority_fraction
# ---------------------------------------------------------------------------


def test_majority_label_tie_break() -> None:
    # Lexical order tie-break: "negative" < "neutral" < "positive"
    assert majority_label({"negative": 3, "positive": 3, "neutral": 3}) == "negative"


def test_majority_label_winner() -> None:
    assert majority_label({"positive": 5, "negative": 2, "neutral": 1}) == "positive"


def test_majority_label_empty() -> None:
    assert majority_label({}) == ""


def test_majority_fraction_perfect() -> None:
    assert majority_fraction({"positive": 10}) == 1.0


def test_majority_fraction_half() -> None:
    assert math.isclose(majority_fraction({"positive": 5, "negative": 5}), 0.5)


def test_majority_fraction_empty() -> None:
    assert majority_fraction({}) == 0.0


# ---------------------------------------------------------------------------
# compute_row_consistency
# ---------------------------------------------------------------------------


def test_row_consistency_perfect() -> None:
    result = compute_row_consistency(
        ["positive", "positive", "positive", "positive", "positive"],
        row_number=1,
        sentence="A great day.",
        hidden_label="positive",
        is_conflicting_duplicate=False,
    )
    assert result.n_valid == 5
    assert result.entropy == 0.0
    assert result.majority_label == "positive"
    assert result.majority_fraction == 1.0
    assert result.correct_majority is True


def test_row_consistency_split() -> None:
    result = compute_row_consistency(
        ["positive", "positive", "negative", "negative", "neutral"],
        row_number=2,
        sentence="Mixed signals.",
        hidden_label="positive",
        is_conflicting_duplicate=True,
    )
    assert result.n_valid == 5
    assert 0.0 < result.entropy < 1.0
    # Tie between positive(2) and negative(2): lexical order -> "negative"
    assert result.majority_label == "negative"
    assert result.correct_majority is False  # majority says negative, hidden is positive
    assert result.is_conflicting_duplicate is True


def test_row_consistency_invalid_labels_filtered() -> None:
    """Samples with invalid parse results are filtered before computing consistency."""
    result = compute_row_consistency(
        ["positive", "__invalid__", "positive", "__error__", "positive"],
        row_number=3,
        sentence="Some bad parses.",
        hidden_label="positive",
        is_conflicting_duplicate=False,
    )
    assert result.n_valid == 3  # only the 3 valid "positive" labels count
    assert result.entropy == 0.0  # all 3 valid ones agree
    assert result.majority_label == "positive"
    assert result.correct_majority is True


def test_row_consistency_all_invalid() -> None:
    result = compute_row_consistency(
        ["__invalid__", "__error__", "__invalid__"],
        row_number=4,
        sentence="All bad.",
        hidden_label="positive",
        is_conflicting_duplicate=False,
    )
    assert result.n_valid == 0
    assert result.entropy == 0.0
    assert result.majority_label == ""
    assert result.correct_majority is False


# ---------------------------------------------------------------------------
# compute_self_consistency
# ---------------------------------------------------------------------------


def _make_sc_rows(
    *entropy_label_triples: tuple[float, str, bool],
) -> list[RowSelfConsistency]:
    """Helper: build RowSelfConsistency objects from (entropy, label, is_conflict) shorthand."""
    rows: list[RowSelfConsistency] = []
    for idx, (entropy, label, is_conflict) in enumerate(entropy_label_triples):
        rows.append(
            RowSelfConsistency(
                row_number=idx,
                sentence=f"Sentence {idx}",
                hidden_label=label,
                is_conflicting_duplicate=is_conflict,
                n_samples=5,
                label_counts={label: 5},
                majority_label=label,
                majority_fraction=1.0,
                entropy=entropy,
                correct_majority=True,
                n_valid=5,
            )
        )
    return rows


def test_self_consistency_empty() -> None:
    result = compute_self_consistency(
        "test-model", 0.7, 5, "all", [], total_cost=0.0
    )
    assert result.n_rows == 0
    assert result.mean_entropy == 0.0
    assert result.majority_vote_accuracy == 0.0


def test_self_consistency_perfect() -> None:
    rows = _make_sc_rows(
        (0.0, "positive", False),
        (0.0, "negative", False),
        (0.0, "neutral", False),
    )
    result = compute_self_consistency("test-model", 0.7, 5, "all", rows)
    assert result.n_rows == 3
    assert result.mean_entropy == 0.0
    assert result.consistency_rate == 1.0
    assert result.majority_vote_accuracy == 1.0
    assert result.majority_vote_balanced_accuracy == 1.0


def test_self_consistency_mixed() -> None:
    rows = _make_sc_rows(
        (0.0, "positive", False),
        (0.5, "positive", True),  # conflicting + higher entropy
        (0.0, "negative", False),
        (0.8, "neutral", True),  # conflicting + higher entropy
        (0.0, "positive", False),
    )
    result = compute_self_consistency("test-model", 0.7, 5, "all", rows)
    assert result.n_rows == 5
    assert result.conflicting_rows == 2
    assert result.non_conflicting_rows == 3
    assert result.conflicting_entropy is not None
    assert result.non_conflicting_entropy is not None
    # Conflicting rows here have higher entropy
    assert result.conflicting_entropy > result.non_conflicting_entropy
    # Entropy gap should be negative (conflicting higher)
    assert result.entropy_gap is not None and result.entropy_gap < 0


def test_self_consistency_no_conflicting() -> None:
    rows = _make_sc_rows(
        (0.0, "positive", False),
        (0.3, "negative", False),
    )
    result = compute_self_consistency("test-model", 0.7, 5, "all", rows)
    assert result.conflicting_entropy is None
    assert result.non_conflicting_entropy is not None
    assert result.entropy_gap is None


def test_self_consistency_all_conflicting() -> None:
    rows = _make_sc_rows(
        (0.5, "positive", True),
        (0.7, "negative", True),
    )
    result = compute_self_consistency("test-model", 0.7, 5, "all", rows)
    assert result.conflicting_entropy is not None
    assert result.non_conflicting_entropy is None
    assert result.entropy_gap is None


def test_self_consistency_cost() -> None:
    rows = _make_sc_rows((0.0, "positive", False))
    result = compute_self_consistency("test-model", 0.7, 5, "all", rows, total_cost=1.23)
    assert result.total_cost == 1.23


def test_self_consistency_majority_vote_mismatches() -> None:
    """Test that majority_vote_accuracy correctly counts incorrect majority votes."""
    rows = [
        RowSelfConsistency(
            row_number=0,
            sentence="Correct majority",
            hidden_label="positive",
            is_conflicting_duplicate=False,
            n_samples=5,
            label_counts={"positive": 3, "negative": 2},
            majority_label="positive",
            majority_fraction=0.6,
            entropy=0.971,
            correct_majority=True,
            n_valid=5,
        ),
        RowSelfConsistency(
            row_number=1,
            sentence="Wrong majority",
            hidden_label="negative",
            is_conflicting_duplicate=False,
            n_samples=5,
            label_counts={"positive": 3, "negative": 2},
            majority_label="positive",
            majority_fraction=0.6,
            entropy=0.971,
            correct_majority=False,  # majority picked "positive" but hidden = "negative"
            n_valid=5,
        ),
    ]
    result = compute_self_consistency("test-model", 0.7, 5, "all", rows)
    assert result.majority_vote_accuracy == 0.5
    assert result.n_rows == 2


# ---------------------------------------------------------------------------
# RowSelfConsistency dataclass properties
# ---------------------------------------------------------------------------


def test_row_self_consistency_fields() -> None:
    row = RowSelfConsistency(
        row_number=10,
        sentence="Test sentence.",
        hidden_label="positive",
        is_conflicting_duplicate=True,
        n_samples=5,
        label_counts={"positive": 3, "neutral": 2},
        majority_label="positive",
        majority_fraction=0.6,
        entropy=0.971,
        correct_majority=True,
        n_valid=5,
    )
    assert row.row_number == 10
    assert row.sentence == "Test sentence."
    assert row.hidden_label == "positive"
    assert row.is_conflicting_duplicate is True
    assert row.majority_fraction == 0.6
    assert row.n_valid == 5