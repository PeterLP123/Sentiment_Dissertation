from __future__ import annotations

from pathlib import Path

import pytest

from sentiment_benchmark.agreement import krippendorff_alpha_nominal
from sentiment_benchmark.demonstrations import demonstration_pool, select_demonstrations
from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.parser import parse_model_response
from sentiment_benchmark.prompt_sensitivity import prompt_sensitivity
from sentiment_benchmark.prompts import make_prompt, render_messages, with_demonstrations
from sentiment_benchmark.reliability import collect_predictions, run_agreement
from sentiment_benchmark.storage import BenchmarkStore

LABELS = ("positive", "negative", "neutral")


def _row(row_number: int, label: str, sentence: str | None = None, conflicting: bool = False) -> DatasetRow:
    return DatasetRow(
        row_number=row_number,
        sentence=sentence or f"sentence {row_number}",
        hidden_label=label,
        is_duplicate=conflicting,
        has_conflicting_duplicate=conflicting,
        duplicate_group_size=2 if conflicting else 1,
    )


def _response(row_number: int, label: str | None, status: str = "success", parse_status: str = "valid") -> dict:
    return {
        "row_number": row_number,
        "status": status,
        "parse_status": parse_status,
        "normalized_label": label,
        "latency_ms": 5.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


# --- metrics: MCC + balanced accuracy ----------------------------------------


def test_evaluate_includes_balanced_accuracy_and_mcc() -> None:
    rows = [_row(1, "positive"), _row(2, "negative"), _row(3, "neutral"), _row(4, "positive")]
    responses = [
        _response(1, "positive"),
        _response(2, "negative"),
        _response(3, "neutral"),
        _response(4, "positive"),
    ]
    result = evaluate_responses(rows, responses, "m", scope="primary")
    assert result.accuracy == 1.0
    assert result.balanced_accuracy == 1.0
    assert result.mcc == 1.0


def test_mcc_zero_for_constant_prediction() -> None:
    rows = [_row(i, LABELS[i % 3]) for i in range(1, 7)]
    responses = [_response(i, "positive") for i in range(1, 7)]
    result = evaluate_responses(rows, responses, "m", scope="primary")
    # A degenerate constant predictor has MCC 0 (no correlation with the truth).
    assert result.mcc == 0.0
    assert 0.0 <= result.balanced_accuracy <= 1.0


# --- parser: chain-of-thought ------------------------------------------------


@pytest.mark.parametrize(
    "content,expected",
    [
        ("The tone is upbeat.\nAnswer: positive", "positive"),
        ("Step 1: losses widen.\nFinal answer: negative", "negative"),
        ("Reasoning about the figures.\nneutral", "neutral"),
        ("It reads as POSITIVE overall in the end.", "positive"),
        ("Label: neutral\nbecause it is factual", "neutral"),
    ],
)
def test_cot_parses_trailing_label(content: str, expected: str) -> None:
    parsed = parse_model_response(content, "cot")
    assert parsed.parse_status == "valid"
    assert parsed.normalized_label == expected
    assert parsed.explanation is not None


def test_cot_invalid_when_no_label_present() -> None:
    parsed = parse_model_response("I cannot decide either way.", "cot")
    assert parsed.parse_status == "invalid"
    assert parsed.normalized_label is None


def test_cot_prefers_marked_answer_over_earlier_mention() -> None:
    parsed = parse_model_response("This looks positive at first.\nAnswer: negative", "cot")
    assert parsed.normalized_label == "negative"


def test_cot_ignores_label_words_in_reasoning_before_final_line() -> None:
    parsed = parse_model_response("This is not negative overall.\nAnswer: positive", "cot")
    assert parsed.normalized_label == "positive"


# --- few-shot demonstrations -------------------------------------------------


def _balanced_pool() -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    number = 1
    for label in LABELS:
        for _ in range(4):
            rows.append(_row(number, label))
            number += 1
    return rows


def test_demonstration_pool_excludes_eval_and_conflicting() -> None:
    rows = [_row(1, "positive"), _row(2, "negative"), _row(3, "neutral", conflicting=True)]
    pool = demonstration_pool(rows, exclude_row_numbers=[1])
    numbers = {row.row_number for row in pool}
    assert numbers == {2}  # 1 excluded as eval row, 3 excluded as conflicting


def test_select_demonstrations_is_balanced_and_deterministic() -> None:
    pool = _balanced_pool()
    first = select_demonstrations(pool, k_per_class=2, seed=42)
    second = select_demonstrations(pool, k_per_class=2, seed=42)
    assert first == second
    assert len(first) == 6
    label_counts = {label: sum(1 for _, demo_label in first if demo_label == label) for label in LABELS}
    assert label_counts == {"positive": 2, "negative": 2, "neutral": 2}


def test_select_demonstrations_raises_when_insufficient() -> None:
    pool = [_row(1, "positive"), _row(2, "positive"), _row(3, "negative"), _row(4, "neutral")]
    with pytest.raises(ValueError, match="negative"):
        select_demonstrations(pool, k_per_class=2, seed=1)


def test_with_demonstrations_changes_hash_and_renders_turns() -> None:
    base = make_prompt("p", "system", "Sentence:\n{sentence}", "label_only")
    demos = [("good news", "positive"), ("bad news", "negative")]
    few_shot = with_demonstrations(base, demos, few_shot_seed=7)
    assert few_shot.prompt_hash != base.prompt_hash
    assert few_shot.demonstrations == (("good news", "positive"), ("bad news", "negative"))

    messages = render_messages(few_shot, _row(99, "neutral").blind())
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == "positive"
    assert messages[-1]["content"].endswith("sentence 99")


def test_make_prompt_rejects_unknown_demo_label() -> None:
    with pytest.raises(ValueError, match="Demonstration label"):
        make_prompt("p", "s", "{sentence}", "label_only", demonstrations=[("x", "bullish")])


# --- Krippendorff independent cross-check ------------------------------------


def test_krippendorff_matches_hand_computed_value() -> None:
    # Fully-crossed balanced 2-coder, 2-category design. Worked by hand two ways
    # (coincidence formula and 1 - D_o/D_e), both give exactly 0.125.
    data = [
        ["A", "A", "B", "B"],
        ["A", "B", "A", "B"],
    ]
    assert abs(krippendorff_alpha_nominal(data) - 0.125) < 1e-9


# --- reliability + prompt-sensitivity integration ----------------------------


def _seed_two_model_run(tmp_path: Path) -> tuple[BenchmarkStore, int]:
    store = BenchmarkStore(tmp_path / "bench.sqlite")
    store.initialize()
    rows = [_row(1, "positive"), _row(2, "negative"), _row(3, "neutral"), _row(4, "positive")]
    store.upsert_dataset(rows, str(tmp_path / "data.csv"))
    prompt = make_prompt("p", "s", "{sentence}", "label_only")
    store.save_prompt(prompt)
    config = RunConfig(
        models=["model-a", "model-b"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(tmp_path / "data.csv"),
        db_path=str(store.db_path),
        base_url="https://x/api/v1",
        sample_per_class=1,
    )
    run_id = store.create_run(config, [1, 2, 3, 4])
    predictions = {
        "model-a": {1: "positive", 2: "negative", 3: "neutral", 4: "positive"},
        "model-b": {1: "positive", 2: "positive", 3: "neutral", 4: "positive"},
    }
    for model_id, labels in predictions.items():
        for row_number, label in labels.items():
            store.save_response(
                run_id,
                LLMResponseRecord(
                    row_number=row_number,
                    model_id=model_id,
                    prompt_hash=prompt.prompt_hash,
                    raw_content=label,
                    normalized_label=label,
                    parse_status="valid",
                    status="success",
                ),
            )
        result = evaluate_responses(rows, store.fetch_responses(run_id, model_id), model_id, scope="primary")
        store.save_metrics(run_id, result)
    store.mark_run_complete(run_id)
    return store, run_id


def test_collect_predictions_and_agreement(tmp_path: Path) -> None:
    store, run_id = _seed_two_model_run(tmp_path)
    predictions = collect_predictions(store, run_id, scope="primary")
    assert set(predictions) == {"model-a", "model-b"}

    agreement = run_agreement(store, run_id, scope="primary")
    assert agreement is not None
    assert agreement.n_raters == 2
    assert len(agreement.pairwise_cohen_kappa) == 1
    assert 0.0 <= agreement.observed_agreement <= 1.0


def test_prompt_sensitivity_aggregates_across_runs(tmp_path: Path) -> None:
    store, run_id = _seed_two_model_run(tmp_path)
    # Reuse the same run twice as a stand-in for two prompt variants; the
    # aggregation only needs metric rows keyed by run/model/scope.
    result = prompt_sensitivity(store, [run_id, run_id], "model-a", scope="primary", metric="accuracy")
    assert result.n == 2
    assert result.spread == 0.0
    assert result.minimum == result.maximum == result.mean


def test_prompt_sensitivity_raises_when_run_missing_metrics(tmp_path: Path) -> None:
    store, run_id = _seed_two_model_run(tmp_path)
    with pytest.raises(ValueError, match="run\\(s\\)"):
        prompt_sensitivity(store, [run_id, 99999], "model-a", scope="primary", metric="accuracy")
