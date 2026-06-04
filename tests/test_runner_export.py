import asyncio
import sqlite3
from pathlib import Path

from sentiment_benchmark.exporter import export_run
from sentiment_benchmark.models import BlindExample, LLMResponseRecord, RunConfig
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.runner import BenchmarkRunner
from sentiment_benchmark.storage import BenchmarkStore


class FakeClient:
    async def classify(
        self,
        model_id: str,
        prompt,
        example: BlindExample,
        temperature: float = 0.0,
        max_completion_tokens: int = 8,
        retries: int = 3,
    ) -> LLMResponseRecord:
        label = example.sentence.split()[0]
        return LLMResponseRecord(
            row_number=example.row_number,
            model_id=model_id,
            prompt_hash=prompt.prompt_hash,
            raw_content=label,
            normalized_label=label,
            parse_status="valid",
            status="success",
            raw_response_json={"id": f"gen-{example.row_number}", "choices": [{"message": {"content": label}}]},
            generation_id=f"gen-{example.row_number}",
            prompt_tokens=5,
            completion_tokens=1,
            total_tokens=6,
            latency_ms=10,
        )

    async def get_generation_metadata(self, generation_id: str, retries: int = 3):
        return {"id": generation_id, "total_cost": 0.0001, "provider_name": "Fake", "latency": 10}


def test_runner_creates_metrics_and_exports(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "\n".join(
            [
                "Sentence,Sentiment",
                "positive example,positive",
                "negative example,negative",
                "neutral example,neutral",
            ]
        ),
        encoding="utf-8",
    )
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
    db_path = tmp_path / "benchmark.sqlite"
    config = RunConfig(
        models=["fake/model"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
    )
    store = BenchmarkStore(db_path)
    runner = BenchmarkRunner(client=FakeClient(), store=store)  # type: ignore[arg-type]

    import asyncio

    summary = asyncio.run(runner.run(config))
    assert summary.selected_row_count == 3
    responses = store.fetch_responses(summary.run_id, "fake/model")
    assert len(responses) == 3

    paths = export_run(db_path, summary.run_id, output_dir=tmp_path / "exports")
    exported_names = {Path(path).name for path in paths}
    assert {"responses.csv", "responses.json", "metrics.json", "run.json", "summary.md"} == exported_names


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "\n".join(
            [
                "Sentence,Sentiment",
                "positive example,positive",
                "negative example,negative",
                "neutral example,neutral",
            ]
        ),
        encoding="utf-8",
    )
    return dataset


def test_runner_emits_structured_events(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
    db_path = tmp_path / "events.sqlite"
    config = RunConfig(
        models=["fake/model"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
    )
    runner = BenchmarkRunner(client=FakeClient(), store=BenchmarkStore(db_path))  # type: ignore[arg-type]

    events: list[dict] = []
    asyncio.run(runner.run(config, event_callback=lambda event: events.append(event)))

    types = [event["type"] for event in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    assert "model_started" in types
    assert types.count("row_completed") == 3
    assert "model_completed" in types
    assert events[-1]["status"] == "completed"


def test_runner_cancels_between_models(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
    db_path = tmp_path / "cancel.sqlite"
    config = RunConfig(
        models=["model-a", "model-b"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
    )
    runner = BenchmarkRunner(client=FakeClient(), store=BenchmarkStore(db_path))  # type: ignore[arg-type]

    cancel_event = asyncio.Event()

    def on_event(event: dict) -> None:
        if event["type"] == "model_completed" and event["model_id"] == "model-a":
            cancel_event.set()

    summary = asyncio.run(runner.run(config, event_callback=on_event, cancel_event=cancel_event))
    assert summary.status == "cancelled"

    with sqlite3.connect(db_path) as connection:
        run_row = connection.execute("SELECT status FROM runs WHERE id = ?", (summary.run_id,)).fetchone()
    assert run_row[0] == "cancelled"

    responses_a = BenchmarkStore(db_path).fetch_responses(summary.run_id, "model-a")
    responses_b = BenchmarkStore(db_path).fetch_responses(summary.run_id, "model-b")
    assert len(responses_a) == 3
    assert len(responses_b) == 0
