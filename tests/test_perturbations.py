from __future__ import annotations

import pytest

from sentiment_benchmark.constants import ALLOWED_LABELS
from sentiment_benchmark.perturbations import (
    generate_prompt_suite,
    label_order_variants,
    paraphrase_variants,
)
from sentiment_benchmark.prompts import make_prompt


def _base():
    return make_prompt(
        prompt_id="base",
        system_prompt="Classify the sentiment as positive, negative, or neutral.",
        user_template="Sentence: {sentence}",
        output_mode="label_only",
    )


def _names_all_labels(text: str) -> bool:
    return all(label in text for label in ALLOWED_LABELS)


def test_label_order_variants_default():
    base = _base()
    variants = label_order_variants(base)
    assert len(variants) == 6
    assert len({v.prompt_hash for v in variants}) == 6
    for v in variants:
        assert _names_all_labels(v.system_prompt)
        assert "{sentence}" in v.user_template
        assert v.output_mode == base.output_mode


def test_paraphrase_variants_default():
    base = _base()
    variants = paraphrase_variants(base)
    assert len(variants) >= 4
    assert len({v.prompt_hash for v in variants}) == len(variants)
    for v in variants:
        assert _names_all_labels(v.system_prompt)
        assert v.output_mode == base.output_mode
        assert "{sentence}" in v.user_template


def test_generate_prompt_suite_default_union_unique():
    base = _base()
    suite = generate_prompt_suite(base)
    order = label_order_variants(base)
    paraphrase = paraphrase_variants(base)
    assert len(suite) == len(order) + len(paraphrase)
    assert len({v.prompt_hash for v in suite}) == len(suite)


def test_generate_prompt_suite_single_family():
    base = _base()
    suite = generate_prompt_suite(base, include=("label_order",))
    assert len(suite) == len(label_order_variants(base))
    assert all("__order-" in v.prompt_id for v in suite)


def test_generate_prompt_suite_unknown_family():
    base = _base()
    with pytest.raises(ValueError):
        generate_prompt_suite(base, include=("label_order", "bogus"))


def test_generate_prompt_suite_deterministic():
    base = _base()
    first = [(v.prompt_id, v.prompt_hash) for v in generate_prompt_suite(base)]
    second = [(v.prompt_id, v.prompt_hash) for v in generate_prompt_suite(base)]
    assert first == second
