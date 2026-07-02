import asyncio
import csv
import json
from pathlib import Path

from sentiment_benchmark.headline_scoring import score_headlines
from sentiment_benchmark.headline_value import (
    analyze_headline_value,
    collect_scorable_headlines,
    headline_norm_sha256,
)
from sentiment_benchmark.models import BlindExample, LLMResponseRecord
from sentiment_benchmark.prompts import make_prompt


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fixture_collection(tmp_path: Path) -> Path:
    root = tmp_path / "collection"
    raw = root / "raw" / "lseg_fixture"
    raw.mkdir(parents=True)
    manifest = {
        "status": "in_progress",
        "config": {
            "collection": {"id": "fixture", "start": "2026-06-01T00:00:00Z", "end": "2026-06-03T00:00:00Z"},
            "companies": [
                {"symbol": "AAPL", "name": "Apple Inc.", "aliases": ["Apple", "Apple Inc.", "AAPL"]},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "aliases": ["Microsoft", "MSFT"]},
            ],
        },
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_jsonl(
        raw / "headlines.jsonl",
        [
            {
                "story_id": "urn:newsml:reuters.com:20260601:a:1",
                "headline": "Apple beats earnings estimates",
                "version_created": "2026-06-01T12:00:00",
                "source_code": "NS:RTRS",
                "matched_symbols": ["AAPL"],
            },
            {
                # Syndicated duplicate of the same headline: one unique norm.
                "story_id": "urn:web:20260601:b:1",
                "headline": "Apple beats earnings estimates",
                "version_created": "2026-06-01T13:00:00Z",
                "source_code": "NS:AAA",
                "matched_symbols": ["AAPL"],
            },
            {
                "story_id": "urn:web:20260601:d:1",
                "headline": "Microsoft faces antitrust probe",
                "version_created": "2026-06-01T15:00:00Z",
                "source_code": "NS:BBB",
                "matched_symbols": ["MSFT"],
            },
            {
                # Outside the config window: excluded from the scorable set.
                "story_id": "urn:web:20260603:f:1",
                "headline": "Apple launches new product",
                "version_created": "2026-06-03T01:00:00Z",
                "source_code": "NS:AAA",
                "matched_symbols": ["AAPL"],
            },
        ],
    )
    return root


class FakeClient:
    """Labels headlines by keyword; counts calls for resume assertions."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def classify(
        self,
        model_id: str,
        prompt,
        example: BlindExample,
        temperature: float = 0.0,
        max_completion_tokens: int = 8,
        retries: int = 3,
    ) -> LLMResponseRecord:
        self.calls.append(example.sentence)
        label = "positive" if "beats" in example.sentence else "negative"
        return LLMResponseRecord(
            row_number=example.row_number,
            model_id=model_id,
            prompt_hash=prompt.prompt_hash,
            raw_content=label,
            normalized_label=label,
            parse_status="valid",
            status="success",
            latency_ms=5,
        )


def _prompt():
    return make_prompt("test", "Return a label.", "Headline:\n{sentence}\n\nSentiment label:", "label_only")


def test_collect_scorable_headlines_dedupes_and_windows(tmp_path: Path) -> None:
    root = _fixture_collection(tmp_path)
    headlines = collect_scorable_headlines(root)
    texts = sorted(text for _sha, text in headlines)
    # Duplicate collapsed, out-of-window row excluded.
    assert texts == ["Apple beats earnings estimates", "Microsoft faces antitrust probe"]
    assert headlines[0][0] == headline_norm_sha256(dict(headlines)[headlines[0][0]])


def test_score_headlines_writes_scores_and_resumes(tmp_path: Path) -> None:
    root = _fixture_collection(tmp_path)
    output = tmp_path / "scores.csv"
    client = FakeClient()

    summary = asyncio.run(
        score_headlines(
            client,
            collection_root=root,
            model_id="fake/model",
            prompt=_prompt(),
            output_path=output,
            concurrency=2,
        )
    )
    assert summary.attempted == 2
    assert summary.succeeded == 2
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["label"] for row in rows} == {"positive", "negative"}
    assert {row["score"] for row in rows} == {"1.0", "-1.0"}
    assert all(row["status"] == "success" for row in rows)

    # Second invocation skips everything already scored.
    resumed = asyncio.run(
        score_headlines(
            client,
            collection_root=root,
            model_id="fake/model",
            prompt=_prompt(),
            output_path=output,
        )
    )
    assert resumed.attempted == 0
    assert resumed.already_scored == 2
    assert len(client.calls) == 2


def test_analyze_headline_value_backtests_llm_scorers(tmp_path: Path) -> None:
    root = _fixture_collection(tmp_path)
    scores_path = tmp_path / "scores.csv"
    rows = [
        {
            "headline_sha256": headline_norm_sha256("Apple beats earnings estimates"),
            "headline": "Apple beats earnings estimates",
            "model_id": "fake/model",
            "label": "positive",
            "score": "1.0",
            "status": "success",
            "error": "",
        },
        {
            "headline_sha256": headline_norm_sha256("Microsoft faces antitrust probe"),
            "headline": "Microsoft faces antitrust probe",
            "model_id": "fake/model",
            "label": "negative",
            "score": "-1.0",
            "status": "failed_rows_are_ignored",
            "error": "HTTP 503",
        },
    ]
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = analyze_headline_value(
        root,
        output_dir=tmp_path / "analysis",
        sample_size=0,
        llm_scores=(scores_path,),
    )

    with (result.output_dir / "daily_signals.csv").open(newline="", encoding="utf-8") as handle:
        signal_rows = list(csv.DictReader(handle))
    llm_rows = [row for row in signal_rows if row["scorer_id"] == "llm/fake/model"]
    assert llm_rows, "llm scorer rows missing from daily_signals.csv"
    # Both duplicate AAPL associations of the scored headline count, like the lexicon.
    scored = {(row["symbol"], row["news_date"]): row for row in llm_rows if row["valid_count"] != "0"}
    assert set(scored) == {("AAPL", "2026-06-01")}
    assert scored[("AAPL", "2026-06-01")]["valid_count"] == "2"
    assert scored[("AAPL", "2026-06-01")]["signal"] == "positive"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    llm_meta = manifest["llm_scorers"]["fake/model"]
    assert llm_meta["scored_company_associations"] == 2
    assert llm_meta["covered_company_days"] == 1
