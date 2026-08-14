from __future__ import annotations

import pytest

from final_experiments.lib.lseg_prompt_value import (
    EXPECTED_PROMPTS,
    conservative_pending_cost_upper,
    load_prompt_variants,
    parse_impact_labels,
)


def test_prompt_family_has_fixed_identity_and_horizons() -> None:
    variants = load_prompt_variants()
    assert {variant.prompt_id: variant.horizon_sessions for variant in variants} == EXPECTED_PROMPTS
    assert len({variant.prompt_hash for variant in variants}) == len(variants)


def test_parse_impact_labels_builds_fixed_trade_signal() -> None:
    parsed = parse_impact_labels(
        {
            "p_negative": 0.1,
            "p_neutral": 0.2,
            "p_positive": 0.7,
            "target_specific": True,
            "new_information": True,
            "materiality": "high",
        }
    )
    assert parsed["signed_probability"] == pytest.approx(0.6)
    assert parsed["trade_signal"] == pytest.approx(0.45)


def test_parse_impact_labels_fails_closed_on_inconsistent_neutral() -> None:
    with pytest.raises(ValueError, match="fully neutral"):
        parse_impact_labels(
            {
                "p_negative": 0.1,
                "p_neutral": 0.8,
                "p_positive": 0.1,
                "target_specific": False,
                "new_information": False,
                "materiality": "none",
            }
        )


def test_conservative_cost_upper_is_positive_and_scales_with_events() -> None:
    variant = load_prompt_variants()[0]
    event = {
        "target_company_name": "Example Corp",
        "symbol": "EXM",
        "headline": "Example announces a material contract",
        "prior_context": "(No prior headline.)",
    }
    one = conservative_pending_cost_upper([event], variant)
    two = conservative_pending_cost_upper([event, event], variant)
    assert one > 0
    assert two == pytest.approx(2 * one)
