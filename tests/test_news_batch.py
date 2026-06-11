from pathlib import Path

import pytest

from sentiment_benchmark.news_batch import NewsBatchError, build_news_batch_plan, parse_date_window


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
    assert plans[2].query_entry.id == "market_volatility"
    assert plans[2].config.include_domains == ("reuters.com",)
    assert plans[2].config.exclude_domains == ("example.com",)


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
