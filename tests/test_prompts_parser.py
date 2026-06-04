import pytest

from sentiment_benchmark.models import BlindExample, DatasetRow
from sentiment_benchmark.parser import parse_model_response
from sentiment_benchmark.prompts import make_prompt, render_messages


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

