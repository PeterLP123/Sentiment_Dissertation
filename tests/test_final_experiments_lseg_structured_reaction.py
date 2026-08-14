from __future__ import annotations

import pytest

from final_experiments.lib.lseg_structured_reaction import (
    conservative_pending_cost_upper,
    load_structured_reaction_prompt,
    openrouter_request_contract,
    parse_reaction_labels,
)


def test_prompt_renders_no_prior_headline_or_newness_field() -> None:
    prompt = load_structured_reaction_prompt()
    rendered = prompt.render(
        {
            "target_company_name": "Example Corp",
            "symbol": "EXM",
            "headline": "Example reports a larger-than-expected profit",
            "prior_context": "MUST NOT BE RENDERED",
        }
    )
    assert "MUST NOT BE RENDERED" not in rendered
    assert "earlier" not in rendered.casefold()
    assert "new_information" not in openrouter_request_contract()["response_format"]["json_schema"]["schema"]["properties"]


def test_parse_reaction_labels_builds_direct_score() -> None:
    parsed = parse_reaction_labels(
        {
            "target_specific": True,
            "expectation_revision": "better_than_expected",
            "persistence": "persistent",
            "first_tradable_reaction": "up",
            "reaction_score": 0.75,
        }
    )
    assert parsed["trade_signal"] == pytest.approx(0.75)


def test_parse_reaction_labels_rejects_direction_score_mismatch() -> None:
    with pytest.raises(ValueError, match="down requires"):
        parse_reaction_labels(
            {
                "target_specific": True,
                "expectation_revision": "unclear",
                "persistence": "temporary",
                "first_tradable_reaction": "down",
                "reaction_score": 0.25,
            }
        )


def test_conservative_cost_uses_only_current_request() -> None:
    prompt = load_structured_reaction_prompt()
    event = {
        "target_company_name": "Example Corp",
        "symbol": "EXM",
        "headline": "Example wins a contract",
        "prior_context": "X" * 10_000,
    }
    cost = conservative_pending_cost_upper([event], prompt)
    event["prior_context"] = "Y" * 50_000
    assert conservative_pending_cost_upper([event], prompt) == pytest.approx(cost)
