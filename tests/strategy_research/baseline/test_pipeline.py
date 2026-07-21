from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.headline_value import headline_norm_sha256
from sentiment_benchmark.strategy_research.baseline.config import load_baseline_config
from sentiment_benchmark.strategy_research.baseline.pipeline import BaselineRunError, inspect_baseline, run_baseline


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_corpus(repo: Path) -> tuple[Path, list[dict[str, object]]]:
    root = repo / "Data/collections/test_baseline/derived/test_corpus"
    root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for session in (
        "2026-03-06",
        "2026-03-09",
        "2026-03-10",
        "2026-03-11",
        "2026-03-12",
        "2026-03-13",
    ):
        timestamp = f"{session}T13:00:00+00:00"
        for symbol, company, direction in (
            ("AAA", "Alpha Corp", "wins a contract"),
            ("BBB", "Beta Corp", "loses a contract"),
        ):
            family = f"urn:{symbol.lower()}:{session}"
            headline = f"{company} {direction}"
            records.append(
                {
                    "article_id": family,
                    "revision_id": f"{family}:revision",
                    "story_id": f"{family}:1",
                    "story_family_id": family,
                    "headline": headline,
                    "clean_text": f"{company} issued an update. {company} described the material event.",
                    "clean_text_sha256": "source-clean-hash",
                    "version_created": timestamp,
                    "matched_symbols": [symbol],
                    "scoring_eligible": True,
                    "max_scoring_chars": 1_000,
                }
            )
    articles = root / "articles.jsonl"
    articles.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    screening = root / "screening_index.csv"
    screening.write_text("article_id,revision_id\n", encoding="utf-8")
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "status": "completed",
            "config": {
                "companies": [
                    {"symbol": "AAA", "name": "Alpha Corp", "aliases": ["Alpha Corp"]},
                    {"symbol": "BBB", "name": "Beta Corp", "aliases": ["Beta Corp"]},
                ]
            },
            "files": {
                "articles_jsonl": {"path": articles.name, "sha256": sha256_file(articles)},
                "screening_index_csv": {"path": screening.name, "sha256": sha256_file(screening)},
            },
        },
    )
    return manifest, records


def _build_scores(repo: Path, records: list[dict[str, object]]) -> tuple[Path, Path]:
    root = repo / "Data/collections/test_baseline"
    score_path = root / "derived/scores.csv"
    columns = (
        "headline_sha256",
        "matched_symbols",
        "explicit_target",
        "contextual",
        "market_price_technical",
        "baseline",
        "label",
        "p_positive",
        "p_negative",
        "p_neutral",
        "score",
        "status",
    )
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        written: set[tuple[str, str]] = set()
        for record in records:
            symbol = str(record["matched_symbols"][0])  # type: ignore[index]
            label = "positive" if symbol == "AAA" else "negative"
            positive, negative, neutral = (0.8, 0.1, 0.1) if label == "positive" else (0.1, 0.8, 0.1)
            for scorer in ("finbert", "vader"):
                headline_sha = headline_norm_sha256(str(record["headline"]))
                key = (headline_sha, scorer)
                if key in written:
                    continue
                written.add(key)
                writer.writerow(
                    {
                        "headline_sha256": headline_sha,
                        "matched_symbols": symbol,
                        "explicit_target": "true",
                        "contextual": "false",
                        "market_price_technical": "false",
                        "baseline": scorer,
                        "label": label,
                        "p_positive": positive,
                        "p_negative": negative,
                        "p_neutral": neutral,
                        "score": positive - negative,
                        "status": "success",
                    }
                )
    manifest = score_path.with_suffix(".csv.manifest.json")
    _write_json(
        manifest,
        {
            "status": "completed",
            "input": {
                "collection_root": "Data/collections/test_baseline",
                "unique_headlines": len({str(record["headline"]) for record in records}),
            },
            "models": {
                "finbert": {
                    "model_id": "ProsusAI/finbert",
                    "revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
                    "revision_enforced": True,
                    "local_model_path": None,
                },
                "vader": {
                    "implementation": "nltk.sentiment.vader",
                    "lexicon_sha256": "a" * 64,
                },
            },
            "inference": {"local_only": True, "local_files_only_enforced": True},
            "counts": {"remaining": 0},
            "output": {
                "path": "Data/collections/test_baseline/derived/scores.csv",
                "sha256": sha256_file(score_path),
            },
            "sharing": {"contains_licensed_headline_text": True},
        },
    )
    return score_path, manifest


def _build_prices(repo: Path) -> tuple[Path, Path]:
    sessions = (
        "2026-03-05",
        "2026-03-06",
        "2026-03-09",
        "2026-03-10",
        "2026-03-11",
        "2026-03-12",
        "2026-03-13",
        "2026-03-16",
    )
    panel = repo / "Data/derived/prices/test_strategy.csv"
    panel.parent.mkdir(parents=True)
    with panel.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("symbol", "session_date", "adjusted_open"))
        for symbol, start, step in (("AAA", 100.0, 1.0), ("BBB", 100.0, -1.0)):
            for index, session in enumerate(sessions):
                writer.writerow((symbol, session, start + step * index))
    manifest = panel.with_suffix(".manifest.json")
    _write_json(
        manifest,
        {
            "status": "completed",
            "calendar": "XNYS",
            "execution_field": "adjusted_open",
            "return_convention": "open_to_open",
            "adjustment_supported": True,
            "split_adjusted": True,
            "dividend_adjusted": False,
            "symbols": ["AAA", "BBB"],
            "symbol_count": 2,
            "session_count": len(sessions),
            "row_count": 2 * len(sessions),
            "files": {"price_panel": {"path": panel.name, "sha256": sha256_file(panel)}},
        },
    )
    return panel, manifest


def _config(repo: Path) -> Path:
    corpus_manifest, records = _build_corpus(repo)
    scores, score_manifest = _build_scores(repo, records)
    prices, price_manifest = _build_prices(repo)
    path = repo / "baseline.toml"
    path.write_text(
        f"""
[run]
id = "synthetic-baseline-v1"
evaluation_start = "2026-03-09"
sample_status = "previously_explored"

[replication]
paper_id = "Synthetic test"
doi = "10.0000/test"
reference_version = "arXiv:2304.07619v6"
reference_url = "https://arxiv.org/pdf/2304.07619v6"
fidelity = "conceptual_adaptation"
deviations = ["Synthetic fixture"]

[data]
collection_root = "Data/collections/test_baseline"
corpus_manifest = "{corpus_manifest.relative_to(repo).as_posix()}"
score_path = "{scores.relative_to(repo).as_posix()}"
score_manifest = "{score_manifest.relative_to(repo).as_posix()}"
processing_buffer_minutes = 15
calendar = "XNYS"
timezone = "America/New_York"

[prices]
panel_path = "{prices.relative_to(repo).as_posix()}"
manifest_path = "{price_manifest.relative_to(repo).as_posix()}"
execution_field = "adjusted_open"
return_convention = "open_to_open"

[[scorers]]
id = "finbert"
role = "primary"
model_id = "ProsusAI/finbert"
revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
score_definition = "p_positive_minus_p_negative"

[[scorers]]
id = "vader"
role = "comparator"
model_id = "nltk.sentiment.vader"
score_definition = "vader_pos_minus_neg"

[screening]
require_explicit_target = true
require_single_target = true
exclude_market_price_technical = true

[signal]
aggregation = "sign_mean_hard_label"
minimum_events = 1
holding_sessions = 1
carry = "none"

[portfolio]
construction = "equal_weight_dollar_neutral"
gross_exposure = 1.0
minimum_names_per_side = 1
initial_nav_usd = 1000000.0
cost_bps_per_side = 10.0
force_final_liquidation = true

[inference]
block_length_sessions = 5
replications = 2000
seed = 20260718

[outputs]
derived_root = "Data/derived/strategy_research/baselines"
results_root = "results/strategy_research/baselines"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_end_to_end_run_is_deterministic_immutable_and_hash_verified(tmp_path: Path) -> None:
    config = load_baseline_config(_config(tmp_path))

    inspection = inspect_baseline(config, repo_root=tmp_path)
    first = run_baseline(config, repo_root=tmp_path)
    second = run_baseline(config, repo_root=tmp_path)

    assert inspection.resolved_run_id == first.resolved_run_id
    assert inspection.decision_sessions == 6
    assert inspection.development_sessions == 1
    assert inspection.evaluation_sessions == 5
    assert first.reused is False
    assert second.reused is True
    assert first.report_path.is_file()
    assert (first.results_dir / "manifest.json").is_file()
    assert "Alpha Corp" not in (first.derived_dir / "event_labels.jsonl").read_text(encoding="utf-8")
    metrics = json.loads(first.metrics_path.read_text(encoding="utf-8"))
    assert metrics["finbert"]["evaluation"]["active_day_count_excluding_liquidation"] == 5
    assert metrics["finbert"]["evaluation"]["observations"] == 5
    assert metrics["finbert"]["evaluation"]["cumulative_gross_return"] > 0
    daily_rows = [json.loads(line) for line in (first.results_dir / "daily_pnl.jsonl").read_text().splitlines()]
    assert sum(row["final_liquidation"] for row in daily_rows if row["scorer"] == "finbert") == 1

    with (tmp_path / config.data.score_path).open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(BaselineRunError, match="manifest hash does not match"):
        inspect_baseline(config, repo_root=tmp_path)


def test_inspection_rejects_score_manifest_without_cache_only_proof(tmp_path: Path) -> None:
    config = load_baseline_config(_config(tmp_path))
    manifest_path = tmp_path / config.data.score_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["inference"]["local_files_only_enforced"]
    _write_json(manifest_path, manifest)

    with pytest.raises(BaselineRunError, match="does not prove cache-only inference"):
        inspect_baseline(config, repo_root=tmp_path)
