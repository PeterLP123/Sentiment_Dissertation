"""Tests for entity masking, including the generic-alias over-masking guard."""

from __future__ import annotations

from sentiment_benchmark.entity_masking import (
    COMPANY_PLACEHOLDER,
    TICKER_PLACEHOLDER,
    mask_entities,
)


def test_masks_name_ticker_and_alias() -> None:
    result = mask_entities(
        "Apple Inc. (AAPL) beat estimates; analysts love Apple.",
        name="Apple Inc.",
        ticker="AAPL",
        aliases=("Apple",),
    )
    assert "Apple" not in result.masked_text
    assert "AAPL" not in result.masked_text
    assert result.masked_text.count(COMPANY_PLACEHOLDER) == 2  # "Apple Inc." + "Apple"
    assert result.masked_text.count(TICKER_PLACEHOLDER) == 1
    assert result.n_masked == 3


def test_multiword_name_masked_before_constituents() -> None:
    # "Apple Inc." collapses to one [COMPANY], not [COMPANY] Inc.
    result = mask_entities("Apple Inc. rose", name="Apple Inc.", aliases=("Apple",))
    assert result.masked_text == f"{COMPANY_PLACEHOLDER} rose"


def test_does_not_mask_substrings() -> None:
    # The generic-alias trap: "Apple" must not touch "pineapple" or "Applebee's".
    result = mask_entities(
        "A pineapple at Applebee's is not Apple.",
        name="Apple Inc.",
        ticker="AAPL",
        aliases=("Apple",),
    )
    assert "pineapple" in result.masked_text
    assert "Applebee's" in result.masked_text
    assert result.masked_text.endswith(f"is not {COMPANY_PLACEHOLDER}.")
    assert result.n_masked == 1


def test_case_insensitive_match() -> None:
    result = mask_entities("tesla and TESLA and Tesla", name="Tesla", aliases=())
    assert result.masked_text == f"{COMPANY_PLACEHOLDER} and {COMPANY_PLACEHOLDER} and {COMPANY_PLACEHOLDER}"
    assert result.n_masked == 3


def test_no_match_leaves_text_unchanged() -> None:
    result = mask_entities("The market rallied broadly.", name="Apple Inc.", ticker="AAPL", aliases=("Apple",))
    assert result.masked_text == "The market rallied broadly."
    assert result.n_masked == 0
