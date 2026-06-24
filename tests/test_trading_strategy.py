import asyncio
import json
from pathlib import Path

import pytest

from sentiment_benchmark.models import LLMResponseRecord
from sentiment_benchmark.news_source import NewsArticleRecord, article_record_id, normalize_url
from sentiment_benchmark.newsapi_source import NewsApiFetchResult
from sentiment_benchmark.prompts import load_prompts
from sentiment_benchmark.trading_strategy import (
    ArticleCandidate,
    DailySignal,
    IndexFallback,
    PriceRow,
    SentimentScore,
    TradingCompany,
    TradingStrategyError,
    apply_screening_overrides,
    article_consensus,
    calculate_returns,
    canonicalize_url,
    load_trading_config,
    merge_article_candidates,
    run_trading_strategy,
    score_articles,
)

COMPANY = TradingCompany(
    symbol="AAPL",
    name="Apple Inc.",
    query="Apple AAPL stock news",
    tavily_query_id="us_aapl",
    family="AAPL — Apple",
    aliases=("Apple", "AAPL"),
)


def _record(url: str, title: str, published: str, snippet: str = "Business update") -> NewsArticleRecord:
    normalized = normalize_url(url)
    return NewsArticleRecord(
        record_id=article_record_id(normalized),
        url=url,
        normalized_url=normalized,
        title=title,
        snippet=snippet,
        published_date=published,
        published_date_source="search",
        text_quality="snippet_only",
    )


def _candidate(provider: str, record: NewsArticleRecord) -> ArticleCandidate:
    return ArticleCandidate(COMPANY, provider, COMPANY.query, f"/{provider}", record)


def _score(article_id: str, model: str, label: str) -> SentimentScore:
    values = {"positive": 1, "neutral": 0, "negative": -1}
    return SentimentScore(
        article_id=article_id,
        symbol="AAPL",
        news_date="2026-06-10",
        scorer_id=model,
        scorer_kind="llm",
        label=label,
        label_value=values[label],
        status="success",
        parse_status="valid",
        raw_output=label,
        compound=None,
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        generation_id=None,
        total_cost_usd=0.0,
        error=None,
    )


def test_canonicalize_url_removes_tracking_and_www() -> None:
    assert canonicalize_url("https://www.Example.com/story/?utm_source=x&id=2#top") == "https://example.com/story?id=2"


def test_merge_deduplicates_url_and_title_and_assigns_new_york_day() -> None:
    candidates = [
        _candidate(
            "tavily",
            _record("https://example.com/apple?utm_source=x", "Apple launches product", "2026-06-10T01:00:00Z"),
        ),
        _candidate(
            "newsapi",
            _record("https://www.example.com/apple", "Apple launches product", "Tue, 09 Jun 2026 21:00:00 GMT"),
        ),
    ]
    articles = merge_article_candidates(
        candidates,
        timezone="America/New_York",
        selected_dates=("2026-06-09", "2026-06-10"),
    )
    assert len(articles) == 1
    assert articles[0].published_date_local == "2026-06-09"
    assert articles[0].providers == ["newsapi", "tavily"]
    assert articles[0].screening_decision == "include"
    assert "Headline: Apple launches product" in articles[0].scoring_text


def test_screen_excludes_promotions_and_incidental_mentions() -> None:
    articles = merge_article_candidates(
        [
            _candidate("newsapi", _record("https://x.test/1", "Apple Prime Day discount reaches record low", "2026-06-10")),
            _candidate("newsapi", _record("https://x.test/2", "Supplier expands production", "2026-06-10", "Apple mentioned")),
        ],
        timezone="America/New_York",
        selected_dates=("2026-06-10",),
    )
    assert {article.screening_reason for article in articles} == {"consumer_promotion", "target_not_in_title"}


def test_manual_screening_override_is_traceable(tmp_path: Path) -> None:
    articles = merge_article_candidates(
        [_candidate("newsapi", _record("https://x.test/apple", "Apple supplier wins contract", "2026-06-10"))],
        timezone="America/New_York",
        selected_dates=("2026-06-10",),
    )
    override = tmp_path / "overrides.toml"
    override.write_text(
        f'[[overrides]]\narticle_id = "{articles[0].article_id}"\ndecision = "exclude"\nreason = "other_company"\n'
    )
    reviewed = apply_screening_overrides(articles, override)
    assert reviewed[0].screening_decision == "exclude"
    assert reviewed[0].screening_reason == "other_company"
    assert reviewed[0].screening_method == "manual_review_override_v1"


def test_consensus_requires_complete_roster_and_three_way_tie_is_neutral() -> None:
    models = ("a", "b", "c")
    scores = [_score("one", "a", "positive"), _score("one", "b", "negative"), _score("one", "c", "neutral")]
    scores += [_score("two", "a", "positive"), _score("two", "b", "positive")]
    result = article_consensus(scores, models)
    assert len(result) == 1
    assert result[0].article_id == "one"
    assert result[0].label == "neutral"


def _prices() -> list[PriceRow]:
    dates = ["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16"]
    return [
        PriceRow("AAPL", day, 100 + index, 105, 95, 101 + index, 1000, False)
        for index, day in enumerate(dates)
    ]


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [("positive", 1, 1 / 101), ("negative", -1, -1 / 101), ("neutral", 0, 0.0)],
)
def test_return_formula_and_trading_session_horizon(label: str, value: int, expected: float) -> None:
    signal = DailySignal("AAPL", "2026-06-10", "model", 1, 1, float(value), label, value)
    rows = calculate_returns([signal], _prices(), horizons=(1, 3), notional_usd=10_000)
    assert rows[0].entry_date == "2026-06-11"
    assert rows[1].exit_date == "2026-06-15"
    assert rows[0].strategy_return == pytest.approx(expected)
    assert rows[0].pnl_usd == pytest.approx(expected * 10_000)


def test_return_rejects_incomplete_horizon() -> None:
    signal = DailySignal("AAPL", "2026-06-10", "model", 1, 1, 1.0, "positive", 1)
    with pytest.raises(TradingStrategyError, match="incomplete price horizon"):
        calculate_returns([signal], _prices(), horizons=(5,), notional_usd=10_000)


def _index_prices() -> list[PriceRow]:
    dates = ["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16"]
    return [
        PriceRow("^GSPC", day, 5000 + index, 5100, 4900, 5000 + index * 10, 0, False)
        for index, day in enumerate(dates)
    ]


def test_index_fallback_routes_thin_company_day_to_index_only() -> None:
    prices = _prices() + _index_prices()
    fallback = IndexFallback(symbol="^GSPC", min_texts=3)
    thin = DailySignal("AAPL", "2026-06-10", "model", 1, 1, 1.0, "positive", 1)
    covered = DailySignal("AAPL", "2026-06-10", "model", 5, 5, 1.0, "positive", 1)
    rows = calculate_returns([thin, covered], prices, horizons=(1,), notional_usd=10_000, index_fallback=fallback)

    thin_row = next(row for row in rows if row.index_fallback)
    covered_row = next(row for row in rows if not row.index_fallback)
    # The thin company-day keeps its company attribution but trades the index series.
    assert thin_row.symbol == "AAPL"
    assert thin_row.traded_symbol == "^GSPC"
    assert thin_row.entry_adjusted_open == 5001
    # A well-covered company-day still trades its own stock.
    assert covered_row.traded_symbol == "AAPL"
    assert covered_row.entry_adjusted_open == 101


class FakeNewsApi:
    async def fetch(self, config):
        record = _record("https://news.test/apple", "Apple reports strong growth", "2026-06-10T12:00:00Z")
        return NewsApiFetchResult(config, "2026-06-24T00:00:00+00:00", [record], 1, 1)


class FakeTavily:
    async def fetch(self, config):  # pragma: no cover - gaps should not trigger
        raise AssertionError(config)


class FakeLlm:
    async def classify(self, model_id, prompt, example, **_kwargs):
        return LLMResponseRecord(
            row_number=example.row_number,
            model_id=model_id,
            prompt_hash=prompt.prompt_hash,
            raw_content="positive",
            normalized_label="positive",
            parse_status="valid",
            status="success",
            latency_ms=1,
            total_tokens=2,
        )


def test_score_articles_checkpoints_llm_results_before_vader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles = merge_article_candidates(
        [_candidate("newsapi", _record("https://x.test/apple", "Apple wins large contract", "2026-06-10"))],
        timezone="America/New_York",
        selected_dates=("2026-06-10",),
    )
    output = tmp_path / "scores.jsonl"

    def fail_vader(_text: str):
        raise RuntimeError("vader unavailable")

    monkeypatch.setattr("sentiment_benchmark.trading_strategy.classify_vader_text", fail_vader)
    prompt = load_prompts(Path("configs/default_prompts.toml"))["target_company_news_label_only"]

    with pytest.raises(RuntimeError, match="vader unavailable"):
        asyncio.run(
            score_articles(
                articles,
                models=("a",),
                prompt=prompt,
                client=FakeLlm(),
                output_path=output,
                resume_path=None,
                temperature=0.0,
                max_completion_tokens=64,
                concurrency=1,
                retries=0,
            )
        )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["scorer_id"] == "a"
    assert rows[0]["label"] == "positive"


def test_end_to_end_runner_writes_meeting_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "tavily"
    corpus.mkdir()
    tavily_record = _record("https://tavily.test/apple", "Apple wins large contract", "2026-06-10T13:00:00Z")
    (corpus / "articles.jsonl").write_text(json.dumps(tavily_record.__dict__) + "\n")
    package = tmp_path / "package.json"
    package.write_text(
        json.dumps(
            {
                "source_corpora": [
                    {"path": str(corpus), "query_id": "us_aapl", "query": COMPANY.query}
                ]
            }
        )
    )
    config_path = tmp_path / "trading.toml"
    config_path.write_text(
        f"""
[run]
id = "test_run"
title = "Test run"
timezone = "America/New_York"
dates = ["2026-06-10"]
horizons = [1, 2]
notional_usd = 10000
[scoring]
models = ["a", "b", "c"]
prompt_id = "target_company_news_label_only"
prompts_path = "{Path('configs/default_prompts.toml').resolve()}"
temperature = 0.0
max_completion_tokens = 64
concurrency = 2
retries = 0
[sources]
tavily_package_manifest = "{package}"
newsapi_max_pages = 1
[outputs]
news_output_root = "{tmp_path / 'news'}"
derived_output_root = "{tmp_path / 'derived'}"
results_output_root = "{tmp_path / 'results'}"
experiment_registry = "{tmp_path / 'experiments.toml'}"
[[companies]]
symbol = "AAPL"
name = "Apple Inc."
query = "Apple AAPL stock news"
tavily_query_id = "us_aapl"
family = "AAPL — Apple"
aliases = ["Apple", "AAPL"]
"""
    )

    def price_loader(_config):
        return _prices()[:3]

    result = asyncio.run(
        run_trading_strategy(
            config_path,
            newsapi_client=FakeNewsApi(),
            tavily_client=FakeTavily(),
            llm_client=FakeLlm(),
            price_loader=price_loader,
        )
    )
    assert result.accepted_article_count == 2
    assert (result.results_dir / "summary.md").exists()
    assert "1 company-day observations" in (result.results_dir / "summary.md").read_text()
    manifest = json.loads((result.results_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["counts"]["articles_accepted"] == 2
    assert (result.derived_dir / "screening.csv").exists()
    registry = (tmp_path / "experiments.toml").read_text()
    assert 'id = "test_run"' in registry
    assert 'status = "completed"' in registry
    resumed = asyncio.run(
        run_trading_strategy(
            config_path,
            newsapi_client=object(),
            tavily_client=object(),
            llm_client=object(),
            price_loader=lambda _config: [],
        )
    )
    assert resumed.return_count == result.return_count


def test_fixed_config_loads() -> None:
    config = load_trading_config("configs/week3_trading_pilot.toml")
    assert [company.symbol for company in config.companies] == ["AAPL", "AMZN", "TSLA"]
    assert config.horizons == (1, 2, 3, 4, 5, 6, 7)


def test_broadened_reviewed_config_loads_fixed_panel() -> None:
    config = load_trading_config("configs/week3_trading_broad_reviewed.toml")

    assert [company.symbol for company in config.companies] == ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "JPM", "XOM", "BA"]
    assert config.screening_overrides_path == Path("configs/week3_broad_screening_overrides.toml")


def test_index_fallback_disabled_by_default() -> None:
    assert load_trading_config("configs/week3_trading_pilot.toml").index_fallback is None


def test_index_fallback_config_parses(tmp_path: Path) -> None:
    config_path = tmp_path / "trade.toml"
    base = Path("configs/week3_trading_pilot.toml").read_text()
    config_path.write_text(base + '\n[index_fallback]\nenabled = true\nsymbol = "^GSPC"\nmin_texts = 4\n')
    config = load_trading_config(config_path)
    assert config.index_fallback is not None
    assert config.index_fallback.symbol == "^GSPC"
    assert config.index_fallback.min_texts == 4


def test_index_fallback_enabled_requires_symbol(tmp_path: Path) -> None:
    config_path = tmp_path / "trade.toml"
    base = Path("configs/week3_trading_pilot.toml").read_text()
    config_path.write_text(base + "\n[index_fallback]\nenabled = true\nmin_texts = 2\n")
    with pytest.raises(TradingStrategyError, match="index_fallback.symbol is required"):
        load_trading_config(config_path)
