from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.headline_value import headline_norm_sha256
from sentiment_benchmark.strategy_research.baseline.v2.config import load_baseline_v2_config
from sentiment_benchmark.strategy_research.baseline.v2.pipeline import (
    BaselineV2RunError,
    inspect_baseline_v2,
    run_baseline_v2,
)

# finbert: (label, p_positive, p_negative, p_neutral); vader_compound: (label, compound).
COMPANIES = {
    "AAA": ("Alpha Corp", "wins a contract", ("positive", 0.8, 0.1, 0.1), ("positive", 0.6)),
    "CCC": ("Gamma Corp", "wins an award", ("positive", 0.6, 0.2, 0.2), ("positive", 0.3)),
    "EEE": ("Epsilon Corp", "expands its factory", ("positive", 0.65, 0.15, 0.2), ("negative", -0.4)),
    "BBB": ("Beta Corp", "loses a contract", ("negative", 0.1, 0.8, 0.1), ("negative", -0.6)),
    "DDD": ("Delta Corp", "loses a customer", ("negative", 0.2, 0.7, 0.1), ("negative", -0.3)),
}
SESSIONS = ("2026-03-06", "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13")
PRICE_SESSIONS = ("2026-03-05", *SESSIONS, "2026-03-16")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_corpus(repo: Path) -> tuple[Path, list[dict[str, object]]]:
    root = repo / "Data/collections/test_baseline_v2/derived/test_corpus"
    root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for session in SESSIONS:
        timestamp = f"{session}T13:00:00+00:00"
        for symbol, (company, direction, _finbert, _vader) in COMPANIES.items():
            family = f"urn:{symbol.lower()}:{session}"
            records.append(
                {
                    "article_id": family,
                    "revision_id": f"{family}:revision",
                    "story_id": f"{family}:1",
                    "story_family_id": family,
                    "headline": f"{company} {direction}",
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
                    {"symbol": symbol, "name": company, "aliases": [company]}
                    for symbol, (company, _direction, _finbert, _vader) in COMPANIES.items()
                ]
            },
            "files": {
                "articles_jsonl": {"path": articles.name, "sha256": sha256_file(articles)},
                "screening_index_csv": {"path": screening.name, "sha256": sha256_file(screening)},
            },
        },
    )
    return manifest, records


def _build_scores(repo: Path) -> tuple[Path, Path]:
    root = repo / "Data/collections/test_baseline_v2"
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
        for symbol, (company, direction, (label, positive, negative, neutral), _vader) in COMPANIES.items():
            writer.writerow(
                {
                    "headline_sha256": headline_norm_sha256(f"{company} {direction}"),
                    "matched_symbols": symbol,
                    "explicit_target": "true",
                    "contextual": "false",
                    "market_price_technical": "false",
                    "baseline": "finbert",
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
                "path": "Data/collections/test_baseline_v2/derived/scores.csv",
                "sha256": sha256_file(score_path),
            },
            "sharing": {"contains_licensed_headline_text": True},
        },
    )
    return score_path, manifest


def _build_compound_scores(repo: Path) -> tuple[Path, Path]:
    root = repo / "Data/collections/test_baseline_v2"
    path = root / "derived/compound.csv"
    columns = ("headline_sha256", "baseline", "label", "compound", "status", "error")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for _symbol, (company, direction, _finbert, (label, compound)) in COMPANIES.items():
            writer.writerow(
                {
                    "headline_sha256": headline_norm_sha256(f"{company} {direction}"),
                    "baseline": "vader_compound",
                    "label": label,
                    "compound": compound,
                    "status": "success",
                    "error": "",
                }
            )
    manifest = path.with_suffix(".csv.manifest.json")
    _write_json(
        manifest,
        {
            "status": "completed",
            "models": {
                "vader_compound": {
                    "implementation": "nltk.sentiment.vader",
                    "lexicon_sha256": "a" * 64,
                    "classification": "compound_threshold",
                    "threshold": 0.05,
                },
            },
            "inference": {"local_only": True, "local_files_only_enforced": True},
            "counts": {"expected": len(COMPANIES), "succeeded": len(COMPANIES), "failed": 0, "remaining": 0},
            "output": {
                "path": "Data/collections/test_baseline_v2/derived/compound.csv",
                "sha256": sha256_file(path),
            },
            "sharing": {"contains_licensed_headline_text": True},
        },
    )
    return path, manifest


def _build_prices(repo: Path) -> tuple[Path, Path]:
    panel = repo / "Data/derived/prices/test_strategy_v2.csv"
    panel.parent.mkdir(parents=True)
    trends = {"AAA": 1.0, "CCC": 1.0, "EEE": 1.0, "BBB": -1.0, "DDD": -1.0}
    with panel.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("symbol", "session_date", "adjusted_open"))
        for symbol in sorted(COMPANIES):
            for index, session in enumerate(PRICE_SESSIONS):
                writer.writerow((symbol, session, 100.0 + trends[symbol] * index))
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
            "symbols": sorted(COMPANIES),
            "symbol_count": len(COMPANIES),
            "session_count": len(PRICE_SESSIONS),
            "row_count": len(COMPANIES) * len(PRICE_SESSIONS),
            "files": {"price_panel": {"path": panel.name, "sha256": sha256_file(panel)}},
        },
    )
    return panel, manifest


def _config(repo: Path) -> Path:
    corpus_manifest, _records = _build_corpus(repo)
    scores, score_manifest = _build_scores(repo)
    compound, compound_manifest = _build_compound_scores(repo)
    prices, price_manifest = _build_prices(repo)
    path = repo / "baseline_v2.toml"
    path.write_text(
        f"""
[run]
id = "synthetic-baseline-v2"
evaluation_start = "2026-03-09"
sample_status = "previously_explored"
specified_after_v1_results = true

[replication]
paper_id = "Synthetic test"
doi = "10.0000/test"
reference_version = "arXiv:2304.07619v6"
reference_url = "https://arxiv.org/pdf/2304.07619v6"
fidelity = "conceptual_adaptation"
deviations = ["Synthetic fixture"]

[data]
collection_root = "Data/collections/test_baseline_v2"
corpus_manifest = "{corpus_manifest.relative_to(repo).as_posix()}"
score_path = "{scores.relative_to(repo).as_posix()}"
score_manifest = "{score_manifest.relative_to(repo).as_posix()}"
vader_compound_path = "{compound.relative_to(repo).as_posix()}"
vader_compound_manifest = "{compound_manifest.relative_to(repo).as_posix()}"
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
id = "vader_compound"
role = "comparator"
model_id = "nltk.sentiment.vader"
score_definition = "vader_compound_threshold_0p05"

[screening]
require_explicit_target = true
require_single_target = true
exclude_market_price_technical = true

[signal]
arms = ["finbert", "vader_compound", "finbert_magnitude", "agreement"]
minimum_events = 1
holding_sessions = 1
carry = "none"

[portfolio]
construction = "dollar_neutral_two_name_minimum"
gross_exposure = 1.0
minimum_names_per_side = 2
initial_nav_usd = 1000000.0
cost_bps_per_side = 10.0
cost_grid_bps = [5.0, 10.0, 20.0]
force_final_liquidation = true

[inference]
block_length_sessions = 5
replications = 2000
seed = 20260721
event_level_primary = true

[outputs]
derived_root = "Data/derived/strategy_research/baselines"
results_root = "results/strategy_research/baselines"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_end_to_end_run_is_deterministic_immutable_and_arm_complete(tmp_path: Path) -> None:
    config = load_baseline_v2_config(_config(tmp_path))

    inspection = inspect_baseline_v2(config, repo_root=tmp_path)
    first = run_baseline_v2(config, repo_root=tmp_path)
    second = run_baseline_v2(config, repo_root=tmp_path)

    assert inspection.resolved_run_id == first.resolved_run_id
    assert inspection.decision_sessions == 6
    assert inspection.evaluation_sessions == 5
    # Every arm keeps two names on both sides in this fixture, so all six
    # decision sessions trade: finbert 3L/2S, vader 2L/3S, agreement 2L/2S.
    assert inspection.active_sessions_by_arm == {
        "finbert": 6,
        "vader_compound": 6,
        "finbert_magnitude": 6,
        "agreement": 6,
    }
    assert first.reused is False
    assert second.reused is True

    metrics = json.loads(first.metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"finbert", "vader_compound", "finbert_magnitude", "agreement"}
    # Long winners, short losers: the finbert arm must be profitable gross.
    assert metrics["finbert"]["evaluation"]["cumulative_gross_return"] > 0
    assert metrics["finbert"]["evaluation"]["maximum_name_concentration"] <= 0.25 + 1e-9

    bootstrap = json.loads((first.results_dir / "bootstrap.json").read_text(encoding="utf-8"))
    event_level = bootstrap["event_level_primary"]
    assert event_level["finbert"]["status"] == "ok"
    assert event_level["finbert"]["firm_session_count"] == 25  # 5 stocks x 5 evaluation sessions
    assert event_level["agreement"]["firm_session_count"] == 20  # EEE excluded by disagreement
    assert event_level["finbert"]["session_mean_bootstrap"]["confidence_direction"] == "positive"
    assert set(bootstrap["daily_portfolio"]) == {
        "finbert_vs_cash",
        "finbert_vs_vader_compound",
        "finbert_magnitude_vs_finbert",
        "agreement_vs_finbert",
    }

    cost_grid = json.loads((first.results_dir / "cost_grid.json").read_text(encoding="utf-8"))
    for arm_payload in cost_grid.values():
        assert set(arm_payload) == {"5", "10", "20"}
        # Higher costs can only lower net returns.
        assert (
            arm_payload["5"]["cumulative_net_return"]
            >= arm_payload["10"]["cumulative_net_return"]
            >= arm_payload["20"]["cumulative_net_return"]
        )

    assert "Alpha Corp" not in (first.derived_dir / "event_labels.jsonl").read_text(encoding="utf-8")
    report = first.report_path.read_text(encoding="utf-8")
    assert "specified after the v1 result" in report
    assert "## Primary inference: firm-session event-level bootstrap" in report
    assert "canonical VADER" in report


def test_v2_inspection_fails_closed_on_tampered_compound_artifact(tmp_path: Path) -> None:
    config = load_baseline_v2_config(_config(tmp_path))
    with (tmp_path / config.data.vader_compound_path).open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(BaselineV2RunError, match="hash does not match"):
        inspect_baseline_v2(config, repo_root=tmp_path)


def test_v2_inspection_rejects_non_canonical_compound_convention(tmp_path: Path) -> None:
    config = load_baseline_v2_config(_config(tmp_path))
    manifest_path = tmp_path / config.data.vader_compound_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"]["vader_compound"]["threshold"] = 0.1
    _write_json(manifest_path, manifest)
    with pytest.raises(BaselineV2RunError, match="canonical compound convention"):
        inspect_baseline_v2(config, repo_root=tmp_path)
