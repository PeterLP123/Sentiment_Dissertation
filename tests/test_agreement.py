import math

from sentiment_benchmark.agreement import (
    AgreementResult,
    cohen_kappa,
    compute_agreement,
    fleiss_kappa,
    krippendorff_alpha_nominal,
)


def test_cohen_kappa_empty_returns_zero() -> None:
    assert cohen_kappa([], []) == 0.0


def test_cohen_kappa_perfect_agreement() -> None:
    labels = ["positive", "negative", "neutral", "positive", "negative"]
    assert cohen_kappa(labels, labels) == 1.0


def test_cohen_kappa_single_category_returns_one() -> None:
    # Both raters always say "positive": p_e == 1.0 edge case.
    labels = ["positive"] * 5
    assert cohen_kappa(labels, labels) == 1.0


def test_cohen_kappa_textbook_value() -> None:
    # Classic 2x2 table: a=20 both yes, b=5, c=10, d=15 both no (n=50).
    # p_o = 0.70, p_e = 0.50 -> kappa = 0.4.
    y1 = ["yes"] * 25 + ["no"] * 25
    y2 = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert math.isclose(cohen_kappa(y1, y2), 0.4, abs_tol=1e-6)


def test_cohen_kappa_can_be_negative() -> None:
    # Raters systematically disagree; agreement below chance -> negative kappa.
    y1 = ["a", "a", "b", "b"]
    y2 = ["b", "b", "a", "a"]
    assert cohen_kappa(y1, y2) < 0.0


def test_fleiss_kappa_empty_returns_zero() -> None:
    assert fleiss_kappa([]) == 0.0


def test_fleiss_kappa_perfect_agreement() -> None:
    # All 3 raters agree on every item.
    counts = [[3, 0], [0, 3], [3, 0]]
    assert fleiss_kappa(counts) == 1.0


def test_fleiss_kappa_known_value() -> None:
    # Classic Fleiss worked example (Wikipedia "Fleiss' kappa"): 10 subjects,
    # 14 raters, 5 categories. Published kappa is approximately 0.210.
    counts = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    assert math.isclose(fleiss_kappa(counts), 0.210, abs_tol=1e-3)


def test_fleiss_kappa_unequal_rows_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        fleiss_kappa([[2, 0], [1, 0]])


def test_krippendorff_alpha_perfect_agreement() -> None:
    data = [
        ["a", "b", "c", "a"],
        ["a", "b", "c", "a"],
        ["a", "b", "c", "a"],
    ]
    assert krippendorff_alpha_nominal(data) == 1.0


def test_krippendorff_alpha_no_usable_units() -> None:
    # Only one rating per unit -> nothing pairable.
    data = [["a", None, None], [None, "b", None], [None, None, "c"]]
    assert krippendorff_alpha_nominal(data) == 0.0


def test_krippendorff_alpha_canonical_nominal_example() -> None:
    # Krippendorff-style nominal reliability-data matrix (3 coders, 12 units, with
    # missing values). The coincidence-matrix computation has n = 28 pairable
    # values and sum_c o_cc = 21, giving the hand-verified alpha = 0.67526.
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    ]
    string_data = [[None if value is None else str(value) for value in coder] for coder in data]
    assert math.isclose(krippendorff_alpha_nominal(string_data), 0.675258, abs_tol=1e-4)


def test_compute_agreement_three_raters() -> None:
    predictions = {
        "model-c": {1: "positive", 2: "negative", 3: "neutral", 4: "positive"},
        "model-a": {1: "positive", 2: "negative", 3: "neutral", 4: "negative"},
        "model-b": {1: "positive", 2: "positive", 3: "neutral", 4: "positive"},
    }
    result = compute_agreement(predictions)

    assert isinstance(result, AgreementResult)
    assert result.n_raters == 3
    assert result.raters == ["model-a", "model-b", "model-c"]
    # C(3, 2) = 3 pairs, all overlapping.
    assert len(result.pairwise_cohen_kappa) == 3
    for pair in result.pairwise_cohen_kappa:
        assert pair["rater_a"] < pair["rater_b"]
        assert pair["n"] == 4
        assert set(pair) == {"rater_a", "rater_b", "kappa", "n"}
    assert 0.0 <= result.observed_agreement <= 1.0
    assert result.n_units == 4
    assert result.n_units_all_raters == 4
    assert result.fleiss_kappa is not None
    assert result.krippendorff_alpha is not None


def test_compute_agreement_skips_empty_overlap_pairs() -> None:
    predictions = {
        "a": {1: "positive", 2: "negative"},
        "b": {1: "positive", 2: "neutral"},
        "c": {3: "neutral", 4: "positive"},
    }
    result = compute_agreement(predictions)

    # Only the a/b pair shares items; a/c and b/c overlap is empty.
    assert len(result.pairwise_cohen_kappa) == 1
    pair = result.pairwise_cohen_kappa[0]
    assert pair["rater_a"] == "a"
    assert pair["rater_b"] == "b"
    assert pair["n"] == 2
    # No item is rated by all three raters.
    assert result.n_units_all_raters == 0
    assert result.fleiss_kappa is None


def test_compute_agreement_single_rater_returns_none() -> None:
    predictions = {"only": {1: "positive", 2: "negative", 3: "neutral"}}
    result = compute_agreement(predictions)

    assert result.n_raters == 1
    assert result.raters == ["only"]
    assert result.pairwise_cohen_kappa == []
    assert result.observed_agreement == 0.0
    assert result.fleiss_kappa is None
    assert result.krippendorff_alpha is None
