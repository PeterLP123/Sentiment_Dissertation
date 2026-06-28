import asyncio
import csv
import json
from pathlib import Path

from sentiment_benchmark.corpus_scoring import (
    frozen_design_call_counts,
    load_matrix_config,
    load_matrix_items,
    matrix_plan,
    score_corpus_matrix,
)
from sentiment_benchmark.models import LLMResponseRecord


class FakeClient:
    async def model_digest(self, model_id, retries=3):
        return f"digest-{model_id}"

    async def classify(self, model_id, prompt, example, **kwargs):
        return LLMResponseRecord(
            row_number=example.row_number,
            model_id=model_id,
            prompt_hash=prompt.prompt_hash,
            raw_content='{"positive":0.7,"negative":0.1,"neutral":0.2}',
            normalized_label="positive",
            label_probabilities={"positive": 0.7, "negative": 0.1, "neutral": 0.2},
            parse_status="valid",
            status="success",
        )


def _csv(path: Path, count: int) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "Sentence", "Sentiment"])
        writer.writeheader()
        for index in range(count):
            writer.writerow({"item_id": f"i{index}", "Sentence": f"Text {index}", "Sentiment": "positive"})
    return path


def test_frozen_design_counts() -> None:
    assert frozen_design_call_counts() == {
        "lseg": 100000,
        "benchmark": 47500,
        "hosted": 88500,
        "local": 59000,
        "total": 147500,
    }


def test_matrix_plan_and_resume(tmp_path: Path) -> None:
    main = _csv(tmp_path / "main.csv", 3)
    subset = _csv(tmp_path / "subset.csv", 1)
    config = load_matrix_config("configs/crossed_scoring.toml")
    items = load_matrix_items(main, "benchmark")
    plan = matrix_plan(config, items, {"i0"})
    assert plan.total_calls == 125
    clients = {"openrouter": FakeClient(), "ollama": FakeClient()}
    output = tmp_path / "output"
    first = asyncio.run(
        score_corpus_matrix(
            input_path=main,
            subset_path=subset,
            input_kind="benchmark",
            config_path="configs/crossed_scoring.toml",
            prompts_path="configs/default_prompts.toml",
            output_dir=output,
            clients=clients,
        )
    )
    second = asyncio.run(
        score_corpus_matrix(
            input_path=main,
            subset_path=subset,
            input_kind="benchmark",
            config_path="configs/crossed_scoring.toml",
            prompts_path="configs/default_prompts.toml",
            output_dir=output,
            clients=clients,
        )
    )
    assert first == second
    assert len((output / "scores.jsonl").read_text().splitlines()) == 125
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
