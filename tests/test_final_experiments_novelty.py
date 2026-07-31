"""Focused tests for novelty, repetition, and blinded-audit helpers."""

from __future__ import annotations

import json

import pandas as pd

from final_experiments.lib.novelty import (
    annotate_novelty,
    classify_novelty,
    draw_audit_sample,
    jaccard,
    mix_by_firm_month,
    mix_overall,
    normalize_headline,
    token_set,
)


def test_normalize_headline_strips_punct_and_case() -> None:
    assert normalize_headline("  Apple's Shares Jump 5%! ") == "apple s shares jump 5"


def test_jaccard_identical_and_disjoint() -> None:
    a = token_set("apple raises guidance")
    b = token_set("apple raises guidance")
    c = token_set("oil prices fall")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, c) == 0.0


def test_classify_repetition_screen() -> None:
    assert classify_novelty(exact_repeat_in_window=True, near_dup_in_window=False) == "repetition"
    assert classify_novelty(exact_repeat_in_window=False, near_dup_in_window=False) == "novel_or_unclassified"


def test_annotate_first_mention_exact_repeat_and_near_dup() -> None:
    stories = pd.DataFrame(
        {
            "event_index": [1, 2, 3, 4],
            "event_key": ["a", "b", "c", "d"],
            "symbol": ["AAA"] * 4,
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-10", "2020-03-01"]),
            "headline": [
                "Acme wins contract",
                "Acme wins contract",  # exact repeat within 30d
                "Acme wins big contract",  # near-dup tokens
                "Acme wins contract",  # outside 30d window → not exact_repeat
            ],
            "is_recap": [False, False, False, False],
        }
    )
    out = annotate_novelty(stories)
    assert list(out["is_first_mention"]) == [True, False, True, False]
    assert list(out["exact_repeat_in_window"]) == [False, True, False, False]
    # "acme wins big contract" vs "acme wins contract" → 3/4 = 0.75 < 0.80 default
    assert bool(out.loc[2, "near_dup_in_window"]) is False
    assert list(out["novelty_class"]) == [
        "novel_or_unclassified",
        "repetition",
        "novel_or_unclassified",
        "novel_or_unclassified",
    ]
    assert out["family_revision_count"].isna().all()
    assert "headline" not in out.columns
    assert out.loc[1, "days_since_prior_story"] == 1.0


def test_annotate_near_dup_threshold() -> None:
    stories = pd.DataFrame(
        {
            "event_index": [1, 2],
            "event_key": ["a", "b"],
            "symbol": ["AAA", "AAA"],
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "headline": [
                "acme raises full year guidance today",
                "acme raises full year guidance now",
            ],
            "is_recap": [False, False],
        }
    )
    out = annotate_novelty(stories)
    # tokens: 6 vs 6, intersection 5 → 5/7 ≈ 0.714 < 0.80
    # Make them closer:
    stories.loc[1, "headline"] = "acme raises full year guidance today again"
    out = annotate_novelty(stories)
    # "acme raises full year guidance today" vs "... today again"
    # 6 tokens vs 7; inter 6 → 6/7 ≈ 0.857 >= 0.80
    assert bool(out.loc[1, "near_dup_in_window"]) is True
    assert out.loc[1, "novelty_class"] == "repetition"


def test_market_recap_is_orthogonal_to_novelty() -> None:
    stories = pd.DataFrame(
        {
            "event_index": [1],
            "event_key": ["a"],
            "symbol": ["AAA"],
            "session_date": pd.to_datetime(["2020-01-02"]),
            "headline": ["Shares jump 5% in pre-market"],
            "is_recap": [True],
        }
    )
    out = annotate_novelty(stories)
    assert out.loc[0, "novelty_class"] == "novel_or_unclassified"
    assert bool(out.loc[0, "is_market_recap"]) is True


def test_same_day_input_order_does_not_change_labels() -> None:
    headlines = {
        "a": "one two three four five six seven eight",
        "b": "one two three four five six seven eight nine",
        "c": "one two three four five six seven nine",
    }

    def annotate(order: str) -> dict[str, str]:
        stories = pd.DataFrame(
            {
                "event_index": range(1, 4),
                "event_key": list(order),
                "symbol": ["AAA"] * 3,
                "session_date": pd.to_datetime(["2020-01-02"] * 3),
                "headline": [headlines[key] for key in order],
                "is_recap": [False] * 3,
            }
        )
        out = annotate_novelty(stories)
        return dict(zip(out["event_key"], out["novelty_class"].astype(str), strict=True))

    assert annotate("abc") == annotate("acb")
    assert set(annotate("abc").values()) == {"novel_or_unclassified"}


def test_mix_overall_and_firm_month() -> None:
    annotated = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-15", "2020-01-03"]),
            "novelty_class": [
                "novel_or_unclassified",
                "novel_or_unclassified",
                "repetition",
            ],
            "is_market_recap": [False, True, False],
        }
    )
    overall = mix_overall(annotated)
    assert int(overall.loc[overall["novelty_class"] == "novel_or_unclassified", "n"].iloc[0]) == 2
    assert abs(float(overall["share"].sum()) - 1.0) < 1e-12
    by_fm = mix_by_firm_month(annotated)
    aaa = by_fm.loc[(by_fm["symbol"] == "AAA") & (by_fm["year_month"] == "2020-01")]
    assert int(aaa["n_stories"].iloc[0]) == 2
    assert abs(float(aaa["share_novel_or_unclassified"].iloc[0]) - 1.0) < 1e-12
    assert abs(float(aaa["share_market_recap"].iloc[0]) - 0.5) < 1e-12


def test_draw_audit_sample_stratified_and_seeded() -> None:
    stories = pd.DataFrame(
        {
            "event_index": range(9),
            "event_key": [f"k{i}" for i in range(9)],
            "symbol": ["AAA"] * 9,
            "session_date": pd.to_datetime(["2020-01-02"] * 9),
            "headline": [f"headline {i}" for i in range(9)],
            "is_recap": [False] * 9,
        }
    )
    annotated = pd.DataFrame(
        {
            "event_key": [f"k{i}" for i in range(9)],
            "novelty_class": ["novel_or_unclassified"] * 6 + ["repetition"] * 3,
            "is_market_recap": [False] * 9,
            "is_first_mention": [True] * 9,
            "exact_repeat_in_window": [False] * 9,
            "near_dup_in_window": [False] * 9,
            "max_jaccard_in_window": [0.0] * 9,
            "days_since_prior_story": [float("nan")] * 9,
        }
    )
    a, a_key = draw_audit_sample(stories, annotated, n=6, seed=7)
    b, b_key = draw_audit_sample(stories, annotated, n=6, seed=7)
    assert len(a) == 6
    assert a["event_key"].tolist() == b["event_key"].tolist()
    assert a_key["event_key"].tolist() == b_key["event_key"].tolist()
    assert set(a_key["novelty_class"]) == {"novel_or_unclassified", "repetition"}
    assert "headline" in a.columns
    assert "novelty_class" not in a.columns
    assert "max_jaccard_in_window" not in a.columns
    assert "headline" not in a_key.columns
    assert all(isinstance(json.loads(value), list) for value in a["prior_30d_headlines_json"])
    weighted = a_key.groupby("novelty_class")["sample_weight"].sum()
    assert weighted["novel_or_unclassified"] == 6
    assert weighted["repetition"] == 3
    assert a["human_novelty_class"].isna().all()


def test_draw_audit_sample_allows_n_smaller_than_class_count() -> None:
    stories = pd.DataFrame(
        {
            "event_index": [1, 2],
            "event_key": ["a", "b"],
            "symbol": ["AAA", "AAA"],
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "headline": ["first", "second"],
            "is_recap": [False, False],
        }
    )
    annotated = pd.DataFrame(
        {
            "event_key": ["a", "b"],
            "novelty_class": ["novel_or_unclassified", "repetition"],
            "is_market_recap": [False, False],
            "is_first_mention": [True, True],
            "exact_repeat_in_window": [False, False],
            "near_dup_in_window": [False, True],
            "max_jaccard_in_window": [0.0, 0.9],
            "days_since_prior_story": [float("nan"), 1.0],
        }
    )
    worksheet, key = draw_audit_sample(stories, annotated, n=1, seed=7)
    assert len(worksheet) == len(key) == 1
