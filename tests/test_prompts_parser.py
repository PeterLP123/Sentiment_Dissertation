from pathlib import Path

import pytest

from sentiment_benchmark.models import BlindExample, DatasetRow
from sentiment_benchmark.parser import parse_model_response
from sentiment_benchmark.prompts import load_prompts, make_prompt, render_messages


def test_render_messages_accepts_only_blind_examples() -> None:
    prompt = make_prompt(
        "test",
        "Return only a label.",
        "Sentence:\n{sentence}\n\nSentiment label:",
        "label_only",
    )
    row = DatasetRow(2, "Example sentence", "positive", False, False, 1)
    with pytest.raises(TypeError):
        render_messages(prompt, row)  # type: ignore[arg-type]

    messages = render_messages(prompt, BlindExample(row_number=row.row_number, sentence=row.sentence))
    rendered = str(messages)
    assert "Example sentence" in rendered
    assert "positive" not in rendered
    assert "hidden_label" not in rendered


def test_label_only_parser_valid_cases() -> None:
    assert parse_model_response("positive", "label_only").normalized_label == "positive"
    assert parse_model_response(" Negative. ", "label_only").normalized_label == "negative"
    assert parse_model_response("`neutral`", "label_only").normalized_label == "neutral"


def test_label_only_parser_invalid_cases() -> None:
    assert parse_model_response("positive sentiment", "label_only").parse_status == "invalid"
    assert parse_model_response("mixed", "label_only").parse_status == "invalid"
    assert parse_model_response("", "label_only").parse_status == "invalid"


def test_explanation_parser() -> None:
    parsed = parse_model_response("neutral\nThe wording is factual.", "explanation")
    assert parsed.parse_status == "valid"
    assert parsed.normalized_label == "neutral"
    assert parsed.explanation == "The wording is factual."



def test_soft_label_parser_plain_json() -> None:
    parsed = parse_model_response('{"positive": 0.7, "negative": 0.1, "neutral": 0.2}', "soft_label")
    assert parsed.parse_status == "valid"
    assert parsed.normalized_label == "positive"
    assert parsed.label_probabilities == {"positive": 0.7, "negative": 0.1, "neutral": 0.2}


def test_soft_label_parser_accepts_fences_and_prose() -> None:
    fenced = '```json\n{"positive": 0.1, "negative": 0.8, "neutral": 0.1}\n```'
    parsed = parse_model_response(fenced, "soft_label")
    assert parsed.normalized_label == "negative"

    prose = 'Here is my estimate: {"positive": 0.2, "negative": 0.2, "neutral": 0.6} based on the wording.'
    parsed = parse_model_response(prose, "soft_label")
    assert parsed.normalized_label == "neutral"


def test_soft_label_parser_rejects_imperfect_sums() -> None:
    parsed = parse_model_response('{"positive": 0.7, "negative": 0.2, "neutral": 0.2}', "soft_label")
    assert parsed.parse_status == "invalid"
    assert parsed.label_probabilities is None


def test_soft_label_parser_rejects_missing_label() -> None:
    parsed = parse_model_response('{"positive": 0.75, "negative": 0.25}', "soft_label")
    assert parsed.parse_status == "invalid"
    assert parsed.label_probabilities is None


def test_soft_label_parser_tie_breaks_in_allowed_label_order() -> None:
    parsed = parse_model_response('{"positive": 0.4, "negative": 0.4, "neutral": 0.2}', "soft_label")
    assert parsed.normalized_label == "positive"


def test_soft_label_parser_invalid_cases() -> None:
    # Unknown key signals the model misunderstood the task.
    assert parse_model_response('{"positive": 0.5, "mixed": 0.5}', "soft_label").parse_status == "invalid"
    # Negative, boolean, and non-numeric values are rejected.
    assert parse_model_response('{"positive": -0.5, "negative": 1.5, "neutral": 0.0}', "soft_label").parse_status == "invalid"
    assert parse_model_response('{"positive": true, "negative": 0.5, "neutral": 0.5}', "soft_label").parse_status == "invalid"
    assert parse_model_response('{"positive": "high", "negative": 0.1, "neutral": 0.1}', "soft_label").parse_status == "invalid"
    # All-zero distribution carries no information.
    assert parse_model_response('{"positive": 0, "negative": 0, "neutral": 0}', "soft_label").parse_status == "invalid"
    # No JSON object at all.
    assert parse_model_response("positive", "soft_label").parse_status == "invalid"
    assert parse_model_response("", "soft_label").parse_status == "invalid"
    assert parse_model_response(None, "soft_label").parse_status == "invalid"


def test_soft_label_parser_skips_bad_objects_and_uses_first_valid() -> None:
    content = 'Scores: {"confidence": 0.9} then {"positive": 0.6, "negative": 0.3, "neutral": 0.1}'
    parsed = parse_model_response(content, "soft_label")
    assert parsed.parse_status == "valid"
    assert parsed.normalized_label == "positive"


def test_make_prompt_accepts_soft_label_mode() -> None:
    prompt = make_prompt(
        "soft",
        "Return probability JSON.",
        "Sentence:\n{sentence}\n\nProbability JSON:",
        "soft_label",
    )
    assert prompt.output_mode == "soft_label"


def test_crossed_soft_label_prompt_suites_are_distinct() -> None:
    prompts = load_prompts(Path("configs/default_prompts.toml"))
    financial = [prompts[f"financial_soft_label_{name}"] for name in ("base", "label_order", "paraphrase")]
    target = [prompts[f"target_company_soft_label_{name}"] for name in ("base", "label_order", "paraphrase")]

    assert all(prompt.output_mode == "soft_label" for prompt in [*financial, *target])
    assert len({prompt.prompt_hash for prompt in financial}) == 3
    assert len({prompt.prompt_hash for prompt in target}) == 3
