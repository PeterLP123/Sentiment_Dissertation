import asyncio
import json
import sqlite3
from pathlib import Path

from sentiment_benchmark.exporter import dataset_sha256, export_run
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


class MetadataFailingClient(FakeClient):
    async def get_generation_metadata(self, generation_id: str, retries: int = 3):
        raise RuntimeError("metadata unavailable")


class CountingClient(FakeClient):
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def classify(
        self,
        model_id: str,
        prompt,
        example: BlindExample,
        temperature: float = 0.0,
        max_completion_tokens: int = 8,
        retries: int = 3,
    ) -> LLMResponseRecord:
        self.calls.append(example.row_number)
        return await super().classify(model_id, prompt, example, temperature, max_completion_tokens, retries)


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
    run_payload = json.loads((tmp_path / "exports" / "run.json").read_text(encoding="utf-8"))
    metadata = run_payload["metadata"]
    assert metadata["package_version"] == "0.1.0"
    assert metadata["dataset_path"] == str(dataset)
    assert metadata["dataset_sha256"] == dataset_sha256(dataset)
    assert metadata["prompt_hash"] == prompt.prompt_hash
    assert metadata["seed"] == 42
    assert metadata["mode"] == "pilot"
    assert metadata["models"] == ["fake/model"]
    assert metadata["base_url"] == "https://openrouter.test/api/v1"
    assert metadata["request_settings"]["sample_per_class"] == 1


def test_dataset_sha256_is_stable_for_same_content(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    content = "Sentence,Sentiment\npositive example,positive\n"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    assert dataset_sha256(first) == dataset_sha256(second)


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


def test_runner_continues_when_generation_metadata_fails(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
    db_path = tmp_path / "metadata-fail.sqlite"
    config = RunConfig(
        models=["fake/model"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
    )
    messages: list[str] = []
    runner = BenchmarkRunner(client=MetadataFailingClient(), store=BenchmarkStore(db_path))  # type: ignore[arg-type]

    summary = asyncio.run(runner.run(config, callback=lambda message: messages.append(message)))

    assert summary.status == "completed"
    assert len(BenchmarkStore(db_path).fetch_responses(summary.run_id, "fake/model")) == 3
    assert any("Metadata lookup failed" in message for message in messages)


def test_runner_cancels_during_model_without_scheduling_remaining_rows(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
    db_path = tmp_path / "cancel-during-model.sqlite"
    config = RunConfig(
        models=["fake/model"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
        concurrency=1,
    )
    client = CountingClient()
    runner = BenchmarkRunner(client=client, store=BenchmarkStore(db_path))  # type: ignore[arg-type]
    cancel_event = asyncio.Event()

    def on_event(event: dict) -> None:
        if event["type"] == "row_completed":
            cancel_event.set()

    summary = asyncio.run(runner.run(config, event_callback=on_event, cancel_event=cancel_event))

    assert summary.status == "cancelled"
    assert client.calls == [2]
    with sqlite3.connect(db_path) as connection:
        run_row = connection.execute("SELECT status FROM runs WHERE id = ?", (summary.run_id,)).fetchone()
        model_row = connection.execute(
            "SELECT status FROM run_models WHERE run_id = ? AND model_id = ?",
            (summary.run_id, "fake/model"),
        ).fetchone()
    assert run_row[0] == "cancelled"
    assert model_row[0] == "cancelled"
