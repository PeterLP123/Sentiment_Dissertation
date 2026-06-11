import json
from pathlib import Path

import pytest

from sentiment_benchmark.news_batch import (
    NewsBatchError,
    build_news_batch_plan,
    build_weekly_date_windows,
    config_parameters,
    find_fetched_corpus,
    load_fetched_corpora,
    parse_date_window,
)


def _write_matrix(path: Path) -> Path:
    path.write_text(
        """
version = 1
description = "Reusable Tavily query matrix for tests."

[[queries]]
id = "bank_earnings"
family = "Bank earnings"
query = "bank earnings sentiment"
topic = "news"
time_range = "month"
max_results = 20
search_depth = "basic"
coverage_target = "Company performance and earnings commentary."
include_domains = []
exclude_domains = []
notes = "Use for source collection; labels are intentionally absent."

[[queries]]
id = "market_volatility"
family = "Market volatility"
query = "market volatility stocks bonds currency"
topic = "news"
time_range = "month"
max_results = 10
search_depth = "basic"
extract_depth = "advanced"
coverage_target = "Market movement and event-driven reporting."
include_domains = ["reuters.com"]
exclude_domains = ["example.com"]
notes = "Use for source collection; labels are intentionally absent."
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_parse_date_window_validates_format_and_order() -> None:
    window = parse_date_window("2026-05-01:2026-05-07")
    assert window.start_date == "2026-05-01"
    assert window.end_date == "2026-05-07"

    with pytest.raises(NewsBatchError, match="START:END"):
        parse_date_window("2026-05-01")
    with pytest.raises(NewsBatchError, match="before or equal"):
        parse_date_window("2026-05-07:2026-05-01")


def test_build_weekly_date_windows_generates_contiguous_weeks() -> None:
    windows = build_weekly_date_windows(3, end_date="2026-06-11")
    assert windows == [
        "2026-05-22:2026-05-28",
        "2026-05-29:2026-06-04",
        "2026-06-05:2026-06-11",
    ]

    # Defaults to ending today and stays parseable.
    for window in build_weekly_date_windows(2):
        parse_date_window(window)

    with pytest.raises(NewsBatchError, match="weeks must be 1 or greater"):
        build_weekly_date_windows(0)
    with pytest.raises(NewsBatchError, match="at most"):
        build_weekly_date_windows(53)
    with pytest.raises(NewsBatchError, match="YYYY-MM-DD"):
        build_weekly_date_windows(2, end_date="June 11")


def test_build_news_batch_plan_uses_all_queries_and_date_windows(tmp_path: Path) -> None:
    matrix = _write_matrix(tmp_path / "matrix.toml")

    plans = build_news_batch_plan(
        query_matrix_path=matrix,
        date_windows=["2026-05-01:2026-05-07", "2026-05-08:2026-05-14"],
    )

    assert len(plans) == 4
    assert plans[0].query_entry.id == "bank_earnings"
    assert plans[0].config.query == "bank earnings sentiment"
    assert plans[0].config.time_range is None
    assert plans[0].config.start_date == "2026-05-01"
    assert plans[1].date_window.start_date == "2026-05-08"
    assert plans[0].config.extract_depth == "basic"
    assert plans[2].query_entry.id == "market_volatility"
    assert plans[2].config.include_domains == ("reuters.com",)
    assert plans[2].config.exclude_domains == ("example.com",)
    assert plans[2].config.extract_depth == "advanced"


def test_build_news_batch_plan_extract_depth_override(tmp_path: Path) -> None:
    matrix = _write_matrix(tmp_path / "matrix.toml")

    plans = build_news_batch_plan(
        query_matrix_path=matrix,
        date_windows=["2026-05-01:2026-05-07"],
        extract_depth="advanced",
    )

    assert all(plan.config.extract_depth == "advanced" for plan in plans)


def test_find_fetched_corpus_matches_exact_parameters_only(tmp_path: Path) -> None:
    matrix = _write_matrix(tmp_path / "matrix.toml")
    plans = build_news_batch_plan(
        query_matrix_path=matrix,
        date_windows=["2026-05-01:2026-05-07"],
        query_ids=["bank_earnings"],
    )
    plan = plans[0]

    corpus_dir = tmp_path / "news" / "tavily_news_existing"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"query": plan.config.query, "record_count": 20, "parameters": config_parameters(plan.config)}),
        encoding="utf-8",
    )
    # Manifests without parameters and broken JSON are ignored, not fatal.
    other_dir = tmp_path / "news" / "tavily_news_other"
    other_dir.mkdir()
    (other_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    fetched = load_fetched_corpora(tmp_path / "news")
    assert [corpus.path for corpus in fetched] == [corpus_dir]
    assert find_fetched_corpus(plan, fetched) == corpus_dir

    # Zero-record corpora are re-fetched rather than treated as done.
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"query": plan.config.query, "record_count": 0, "parameters": config_parameters(plan.config)}),
        encoding="utf-8",
    )
    assert find_fetched_corpus(plan, load_fetched_corpora(tmp_path / "news")) is None
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"query": plan.config.query, "record_count": 20, "parameters": config_parameters(plan.config)}),
        encoding="utf-8",
    )

    different_window = build_news_batch_plan(
        query_matrix_path=matrix,
        date_windows=["2026-05-08:2026-05-14"],
        query_ids=["bank_earnings"],
    )[0]
    assert find_fetched_corpus(different_window, fetched) is None

    different_depth = build_news_batch_plan(
        query_matrix_path=matrix,
        date_windows=["2026-05-01:2026-05-07"],
        query_ids=["bank_earnings"],
        extract_depth="advanced",
    )[0]
    assert find_fetched_corpus(different_depth, fetched) is None

    assert load_fetched_corpora(tmp_path / "missing") == []


def test_build_news_batch_plan_can_select_query_ids(tmp_path: Path) -> None:
    matrix = _write_matrix(tmp_path / "matrix.toml")

    plans = build_news_batch_plan(
        query_matrix_path=matrix,
        date_windows=["2026-05-01:2026-05-07"],
        query_ids=["market_volatility"],
    )

    assert len(plans) == 1
    assert plans[0].query_entry.id == "market_volatility"
    assert plans[0].config.max_results == 10

    with pytest.raises(NewsBatchError, match="unknown query id"):
        build_news_batch_plan(
            query_matrix_path=matrix,
            date_windows=["2026-05-01:2026-05-07"],
            query_ids=["missing"],
        )
