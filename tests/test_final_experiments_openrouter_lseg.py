import csv
import json
from pathlib import Path

from final_experiments.lib.openrouter_lseg import score_lseg_openrouter
from sentiment_benchmark.headline_value import ScorableHeadline


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def classify(self, row, retries: int = 3):
        del retries
        self.calls.append(row.sentence)
        if row.sentence == "Second headline" and self.calls.count(row.sentence) == 1:
            return {"status": "api_error", "error": "temporary 429", "attempt_count": 4}
        return {
            "status": "success",
            "normalized_label": "positive",
            "probabilities": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            "returned_provider": "DeepInfra",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "reported_cost_usd": 0.001,
            "generation_id": f"generation-{len(self.calls)}",
            "attempt_count": 1,
        }


def _records() -> list[ScorableHeadline]:
    return [
        ScorableHeadline("a" * 64, "First headline", ("AAA",), "2026-01-01T00:00:00Z", True, False, False),
        ScorableHeadline("b" * 64, "Second headline", ("BBB",), "2026-01-02T00:00:00Z", False, True, False),
    ]


def test_lseg_scoring_is_append_only_and_retries_failures(monkeypatch, tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "headlines.jsonl").write_text('{"headline":"source"}\n', encoding="utf-8")
    (collection / "manifest.json").write_text(json.dumps({"source": "test"}), encoding="utf-8")
    monkeypatch.setattr(
        "final_experiments.lib.openrouter_lseg.collect_scorable_headline_records",
        lambda root: _records(),
    )
    output = tmp_path / "scores.csv"
    client = FakeClient()

    import asyncio

    first = asyncio.run(score_lseg_openrouter(collection, output, concurrency=2, max_population=2, client=client))
    second = asyncio.run(score_lseg_openrouter(collection, output, concurrency=2, max_population=2, client=client))

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["success", "api_error", "success"]
    assert len({row["headline_sha256"] for row in rows if row["status"] == "success"}) == 2
    assert rows[-1]["score"] == "0.7000000000000001"
    assert first["status"] == "completed_partial"
    assert first["counts"]["remaining_after_run"] == 1
    assert second["status"] == "completed"
    assert second["counts"]["already_successful"] == 1
    assert second["counts"]["attempted_this_run"] == 1
    assert client.calls == ["First headline", "Second headline", "Second headline"]
