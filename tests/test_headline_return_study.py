from __future__ import annotations

import pandas as pd
import pytest

import sentiment_benchmark.headline_return_study as headline_return_study
from sentiment_benchmark.baselines import SoftSentiment, VaderSentiment
from sentiment_benchmark.headline_return_study import (
    align_next_open_returns,
    audit_headline_scores,
    calculate_return_metrics,
    run_headline_return_study,
    score_collection_baselines,
    score_collection_vader_compound,
)
from sentiment_benchmark.headline_value import ScorableHeadline


def _signal(timestamp: str, score: float, sha: str) -> dict[str, object]:
    return {
        "headline_sha256": sha,
        "symbol": "TEST",
        "first_timestamp": pd.Timestamp(timestamp),
        "score": score,
        "explicit_target": True,
        "contextual": False,
        "market_price_technical": False,
        "label": "positive" if score > 0 else "negative",
        "score_100": 100 * score,
        "scorer": "test",
    }


def test_alignment_uses_same_open_only_when_headline_is_available_before_open() -> None:
    signals = pd.DataFrame(
        [
            _signal("2026-07-06T13:00:00Z", 0.5, "pre-open"),
            _signal("2026-07-06T13:30:00Z", 0.5, "at-open"),
            _signal("2026-07-06T14:00:00Z", -0.5, "after-open"),
        ]
    )
    prices = pd.DataFrame(
        {
            "symbol": ["TEST", "TEST", "TEST"],
            "session_date": ["2026-07-06", "2026-07-07", "2026-07-08"],
            "open": [100.0, 110.0, 121.0],
        }
    )

    aligned = align_next_open_returns(signals, prices).set_index("headline_sha256")

    assert str(aligned.loc["pre-open", "entry_date"]) == "2026-07-06"
    assert str(aligned.loc["pre-open", "exit_date"]) == "2026-07-07"
    assert aligned.loc["pre-open", "forward_return"] == pytest.approx(0.10)
    assert str(aligned.loc["at-open", "entry_date"]) == "2026-07-07"
    assert str(aligned.loc["after-open", "entry_date"]) == "2026-07-07"
    assert str(aligned.loc["after-open", "exit_date"]) == "2026-07-08"
    assert aligned.loc["after-open", "forward_return"] == pytest.approx(0.10)


def test_return_metrics_aggregate_headlines_before_forming_positions() -> None:
    frame = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2026-07-06").date()] * 2 + [pd.Timestamp("2026-07-07").date()],
            "symbol": ["TEST", "TEST", "TEST"],
            "headline_sha256": ["a", "b", "c"],
            "score": [0.8, 0.2, -0.5],
            "forward_return": [0.02, 0.02, -0.01],
        }
    )

    metrics = calculate_return_metrics(frame, bootstrap_samples=100, seed=42)

    assert metrics["company_day_observations"] == 2
    assert metrics["headline_observations"] == 3
    assert metrics["mean_strategy_return"] == pytest.approx(0.015)
    assert metrics["hit_rate"] == 1.0
    assert metrics["mean_strategy_return_ci_low"] <= metrics["mean_strategy_return"]
    assert metrics["mean_strategy_return_ci_high"] >= metrics["mean_strategy_return"]


def test_score_audit_checks_probability_arithmetic_without_copying_text(tmp_path) -> None:
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "headline_sha256": "abc",
                "headline": "licensed text stays out of the audit",
                "model_id": "model",
                "model_digest": "digest",
                "prompt_hash": "prompt",
                "label": "positive",
                "p_positive": 0.7,
                "p_negative": 0.1,
                "p_neutral": 0.2,
                "score": 0.6,
                "score_100": 60.0,
                "status": "success",
                "error": "",
                "latency_ms": 10,
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            }
        ]
    ).to_csv(scores, index=False, lineterminator="\n")
    audit_path = tmp_path / "audit.json"

    summary = audit_headline_scores(scores, audit_path, expected_rows=1)

    assert summary.valid is True
    assert "licensed text" not in audit_path.read_text(encoding="utf-8")


def test_complete_collection_baselines_are_resumable(tmp_path, monkeypatch) -> None:
    records = [
        ScorableHeadline("a", "positive headline", ("AAA",), "2026-01-01T00:00:00Z", True, False, False),
        ScorableHeadline("b", "negative headline", ("BBB",), "2026-01-02T00:00:00Z", False, True, False),
    ]
    monkeypatch.setattr(headline_return_study, "collect_scorable_headline_records", lambda _: records)
    local_only_calls: list[str] = []

    def fake_vader_digest(*, local_files_only: bool) -> str:
        assert local_files_only is True
        return "vader-lexicon-digest"

    def fake_vader_score(text: str, *, local_files_only: bool) -> SoftSentiment:
        assert local_files_only is True
        local_only_calls.append(text)
        return SoftSentiment("positive", 0.7, 0.1, 0.2)

    monkeypatch.setattr(headline_return_study, "vader_lexicon_sha256", fake_vader_digest)
    monkeypatch.setattr(headline_return_study, "score_vader_text", fake_vader_score)

    def fake_finbert_batches(texts, *, batch_size, inference_batch_size, revision, local_files_only):
        del batch_size, inference_batch_size
        assert revision == "4556d13015211d73dccd3fdd39d39232506f3e43"
        assert local_files_only is True
        for _text in texts:
            yield [SoftSentiment("negative", 0.1, 0.7, 0.2)]

    monkeypatch.setattr(headline_return_study, "iter_finbert_text_batches", fake_finbert_batches)
    output = tmp_path / "complete_baselines.csv"

    revision = "4556d13015211d73dccd3fdd39d39232506f3e43"
    first = score_collection_baselines(
        tmp_path / "collection",
        output,
        finbert_batch_size=2,
        vader_flush_size=1,
        finbert_revision=revision,
    )
    initial_bytes = output.read_bytes()
    second = score_collection_baselines(
        tmp_path / "collection",
        output,
        finbert_batch_size=2,
        vader_flush_size=1,
        finbert_revision=revision,
    )

    scores = pd.read_csv(output)
    assert first.input_rows == second.input_rows == 2
    assert first.succeeded == second.succeeded == 4
    assert output.read_bytes() == initial_bytes
    assert not scores.duplicated(["headline_sha256", "baseline"]).any()
    assert set(scores["baseline"]) == {"finbert", "vader"}
    manifest = pd.read_json(first.manifest_path, typ="series")
    assert manifest["status"] == "completed"
    assert manifest["counts"]["remaining"] == 0
    assert manifest["models"]["finbert"]["revision_enforced"] is True
    assert manifest["inference"]["local_files_only_enforced"] is True
    assert local_only_calls == ["positive headline", "negative headline"]
    with pytest.raises(ValueError, match="different or unenforced FinBERT provenance"):
        score_collection_baselines(
            tmp_path / "collection",
            output,
            finbert_batch_size=2,
            vader_flush_size=1,
            finbert_revision="a" * 40,
        )


def test_vader_compound_scoring_writes_canonical_labels_and_manifest(tmp_path, monkeypatch) -> None:
    records = [
        ScorableHeadline("a", "great earnings beat", ("AAA",), "2026-01-01T00:00:00Z", True, False, False),
        ScorableHeadline("b", "plain routine filing", ("BBB",), "2026-01-02T00:00:00Z", True, False, False),
    ]
    monkeypatch.setattr(headline_return_study, "collect_scorable_headline_records", lambda _: records)
    monkeypatch.setattr(
        headline_return_study,
        "vader_lexicon_sha256",
        lambda *, local_files_only: "b" * 64,
    )

    def fake_classify(text: str, *, local_files_only: bool) -> VaderSentiment:
        assert local_files_only is True
        return VaderSentiment("positive", 0.62) if "great" in text else VaderSentiment("neutral", 0.0)

    monkeypatch.setattr(headline_return_study, "classify_vader_text", fake_classify)
    output = tmp_path / "compound.csv"

    summary = score_collection_vader_compound(tmp_path / "collection", output)

    scores = pd.read_csv(output)
    assert summary.succeeded == 2 and summary.failed == 0
    assert set(scores["baseline"]) == {"vader_compound"}
    assert list(scores["label"]) == ["positive", "neutral"]
    assert list(scores["compound"]) == [0.62, 0.0]
    manifest = pd.read_json(summary.manifest_path, typ="series")
    assert manifest["status"] == "completed"
    assert manifest["counts"]["remaining"] == 0
    assert manifest["models"]["vader_compound"]["classification"] == "compound_threshold"
    assert manifest["models"]["vader_compound"]["threshold"] == 0.05
    assert manifest["inference"]["local_files_only_enforced"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        score_collection_vader_compound(tmp_path / "collection", output)


def test_return_study_writes_aggregate_outputs_for_all_scorers(tmp_path) -> None:
    common = [
        {
            "headline_sha256": "a",
            "headline": "first synthetic headline",
            "matched_symbols": "TEST",
            "first_timestamp": "2026-07-06T13:00:00Z",
            "explicit_target": True,
            "contextual": False,
            "market_price_technical": False,
            "label": "positive",
            "score": 0.6,
            "score_100": 60,
            "status": "success",
        },
        {
            "headline_sha256": "b",
            "headline": "second synthetic headline",
            "matched_symbols": "TEST",
            "first_timestamp": "2026-07-07T14:00:00.123000+00:00",
            "explicit_target": True,
            "contextual": False,
            "market_price_technical": False,
            "label": "negative",
            "score": -0.4,
            "score_100": -40,
            "status": "success",
        },
    ]
    gemma = tmp_path / "gemma.csv"
    pd.DataFrame(common).to_csv(gemma, index=False, lineterminator="\n")
    baselines = tmp_path / "baselines.csv"
    pd.DataFrame([{**row, "baseline": baseline} for baseline in ("vader", "finbert") for row in common]).to_csv(
        baselines,
        index=False,
        lineterminator="\n",
    )
    prices = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "symbol": ["TEST"] * 4,
            "session_date": ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"],
            "open": [100.0, 101.0, 99.0, 100.0],
        }
    ).to_csv(prices, index=False, lineterminator="\n")
    events = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "normalized_headline_sha256": ["a", "b"],
            "event_type": ["earnings_guidance", "analyst_rating"],
        }
    ).to_csv(events, index=False, lineterminator="\n")

    result = run_headline_return_study(
        gemma,
        baselines,
        prices,
        tmp_path / "study",
        bootstrap_samples=10,
        event_records_path=events,
    )

    metrics = pd.read_csv(result.output_dir / "return_metrics.csv")
    assert set(metrics["scorer"]) == {"finbert", "gemma4:e4b-it-qat", "vader"}
    assert (result.output_dir / "event_type_coverage.csv").exists()
    assert "synthetic headline" not in result.report_path.read_text(encoding="utf-8")
