import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.models import LLMResponseRecord
from sentiment_benchmark.news_source import NewsArticleRecord, article_record_id, normalize_url
from sentiment_benchmark.newsapi_source import NewsApiFetchResult
from sentiment_benchmark.prompts import load_prompts
from sentiment_benchmark.trading_strategy import (
    ArticleCandidate,
    DailySignal,
    DecisionPolicyConfig,
    IndexFallback,
    PriceRow,
    SentimentScore,
    TradingCompany,
    TradingDecision,
    TradingStrategyError,
    _mask_articles,
    apply_screening_overrides,
    article_consensus,
    calculate_returns,
    canonicalize_url,
    load_trading_config,
    make_trading_decisions,
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


def test_decision_policy_boundaries_and_insufficient_evidence() -> None:
    policy = DecisionPolicyConfig(min_valid_stories=3, threshold=0.5, transaction_cost_bps_per_side=10)
    signals = [
        DailySignal("AAPL", "2026-06-10", "model", 3, 3, 0.5, "positive", 1),
        DailySignal("AAPL", "2026-06-10", "model2", 3, 3, -0.5, "negative", -1),
        DailySignal("AAPL", "2026-06-10", "model3", 3, 3, 0.49, "positive", 1),
        DailySignal("AAPL", "2026-06-10", "model4", 2, 2, 1.0, "positive", 1),
        DailySignal("AAPL", "2026-06-10", "model5", 0, 0, None, None, None),
    ]

    decisions = make_trading_decisions(signals, policy)

    assert [decision.action for decision in decisions] == ["buy", "sell", "hold", "hold", "hold"]
    assert decisions[2].reason == "inside_no_trade_band"
    assert decisions[3].reason == "insufficient_valid_stories"
    assert decisions[4].reason == "no_valid_sentiment"


def test_decision_returns_skip_holds_and_report_net_costs() -> None:
    decisions = [
        TradingDecision("AAPL", "2026-06-10", "buy", 3, 3, 1.0, "buy", 1, 0.5, 3, "v1", "threshold"),
        TradingDecision("AAPL", "2026-06-10", "hold", 3, 3, 0.0, "hold", 0, 0.5, 3, "v1", "band"),
    ]

    rows = calculate_returns(
        decisions,
        _prices(),
        horizons=(1,),
        notional_usd=10_000,
        transaction_cost_bps_per_side=10,
    )

    assert len(rows) == 1
    assert rows[0].action == "buy"
    assert rows[0].transaction_cost == pytest.approx(0.002)
    assert rows[0].net_strategy_return == pytest.approx(rows[0].strategy_return - 0.002)
    assert rows[0].net_pnl_usd == pytest.approx(rows[0].net_strategy_return * 10_000)


@pytest.mark.parametrize(
    ("available_at", "expected_entry"),
    [
        ("2026-06-10T12:00:00+00:00", "2026-06-10"),  # 08:00 New York, before the open
        ("2026-06-10T14:00:00+00:00", "2026-06-11"),  # 10:00 New York, after the open
        ("2026-06-13T12:00:00+00:00", "2026-06-15"),  # weekend
    ],
)
def test_lseg_availability_enters_at_next_observed_session_open(available_at: str, expected_entry: str) -> None:
    signal = DailySignal("AAPL", "2026-06-10", "model", 3, 3, 1.0, "positive", 1, available_at)

    row = calculate_returns([signal], _prices(), horizons=(1,), notional_usd=10_000)[0]

    assert row.entry_date == expected_entry
    assert row.availability_timestamp == available_at


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
    def __init__(self):
        self.calls = 0

    async def classify(self, model_id, prompt, example, **_kwargs):
        self.calls += 1
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


def test_strict_resume_identity_invalidates_changed_content_and_model_digest(tmp_path: Path) -> None:
    articles = merge_article_candidates(
        [_candidate("newsapi", _record("https://x.test/apple", "Apple wins contract", "2026-06-10"))],
        timezone="America/New_York",
        selected_dates=("2026-06-10",),
    )
    article = articles[0]
    article.source_revision_id = "revision-1"
    article.scoring_text_sha256 = "content-1"
    output = tmp_path / "scores.jsonl"
    prompt = load_prompts(Path("configs/default_prompts.toml"))["target_company_news_label_only"]
    first_client = FakeLlm()
    asyncio.run(
        score_articles(
            articles,
            models=("model",),
            prompt=prompt,
            client=first_client,
            output_path=output,
            resume_path=None,
            temperature=0,
            max_completion_tokens=64,
            concurrency=1,
            retries=0,
            baselines=(),
            model_digests={"model": "digest-1"},
            request_settings={"structured_output": True},
            strict_resume_identity=True,
        )
    )
    assert first_client.calls == 1

    resumed_client = FakeLlm()
    asyncio.run(
        score_articles(
            articles,
            models=("model",),
            prompt=prompt,
            client=resumed_client,
            output_path=output,
            resume_path=None,
            temperature=0,
            max_completion_tokens=64,
            concurrency=1,
            retries=0,
            baselines=(),
            model_digests={"model": "digest-1"},
            request_settings={"structured_output": True},
            strict_resume_identity=True,
        )
    )
    assert resumed_client.calls == 0

    settings_client = FakeLlm()
    asyncio.run(
        score_articles(
            articles,
            models=("model",),
            prompt=prompt,
            client=settings_client,
            output_path=output,
            resume_path=None,
            temperature=0,
            max_completion_tokens=64,
            concurrency=1,
            retries=0,
            baselines=(),
            model_digests={"model": "digest-1"},
            request_settings={"structured_output": True, "keep_alive": "30m"},
            strict_resume_identity=True,
        )
    )
    assert settings_client.calls == 1

    prompt_client = FakeLlm()
    changed_prompt = replace(prompt, prompt_hash="changed-prompt-hash")
    asyncio.run(
        score_articles(
            articles,
            models=("model",),
            prompt=changed_prompt,
            client=prompt_client,
            output_path=output,
            resume_path=None,
            temperature=0,
            max_completion_tokens=64,
            concurrency=1,
            retries=0,
            baselines=(),
            model_digests={"model": "digest-1"},
            request_settings={"structured_output": True, "keep_alive": "30m"},
            strict_resume_identity=True,
        )
    )
    assert prompt_client.calls == 1

    changed = replace(article, scoring_text=article.scoring_text + " changed", scoring_text_sha256="content-2")
    changed_client = FakeLlm()
    asyncio.run(
        score_articles(
            [changed],
            models=("model",),
            prompt=prompt,
            client=changed_client,
            output_path=output,
            resume_path=None,
            temperature=0,
            max_completion_tokens=64,
            concurrency=1,
            retries=0,
            baselines=(),
            model_digests={"model": "digest-2"},
            request_settings={"structured_output": True},
            strict_resume_identity=True,
        )
    )
    assert changed_client.calls == 1


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


def test_pure_lseg_ollama_run_needs_no_web_news_clients_and_writes_decisions(tmp_path: Path) -> None:
    corpus = tmp_path / "lseg_corpus"
    corpus.mkdir()
    article_rows = []
    for index in range(3):
        text = f"Apple contract story {index} with favorable revenue implications."
        article_rows.append(
            {
                "article_id": f"source-{index}",
                "revision_id": f"revision-{index}",
                "story_family_id": "urn:test:shared" if index < 2 else f"urn:test:{index}",
                "story_id": f"urn:test:{index}:1",
                "headline": f"Apple wins contract {index}",
                "version_created": f"2026-06-10T1{index}:00:00+00:00",
                "matched_symbols": ["AAPL"],
                "matched_queries": ["R:AAPL.O"],
                "clean_text": text,
                "clean_text_sha256": f"clean-{index}",
                "max_scoring_chars": 8000,
                "scoring_eligible": True,
            }
        )
    articles_path = corpus / "articles.jsonl"
    articles_path.write_text("".join(json.dumps(row) + "\n" for row in article_rows))
    lseg_manifest = corpus / "manifest.json"
    lseg_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "files": {
                    "articles_jsonl": {"path": "articles.jsonl", "sha256": sha256_file(articles_path)}
                },
            }
        )
    )
    config_path = tmp_path / "lseg_trading.toml"
    config_path.write_text(
        f"""
[run]
id = "lseg_test"
title = "LSEG test"
timezone = "America/New_York"
dates = ["2026-06-10"]
horizons = [1]
notional_usd = 10000
[scoring]
provider = "ollama"
models = ["primary:exact", "secondary:exact"]
primary_model = "primary:exact"
baselines = []
consensus_enabled = false
prompt_id = "target_company_news_label_only"
prompts_path = "{Path('configs/default_prompts.toml').resolve()}"
temperature = 0.0
max_completion_tokens = 64
concurrency = 1
retries = 0
ollama_keep_alive = "30m"
ollama_think = false
structured_output = true
[sources]
lseg_corpus_manifest = "{lseg_manifest}"
newsapi_enabled = false
tavily_gap_fetch = false
[signal_policy]
min_valid_stories = 3
threshold = 0.5
transaction_cost_bps_per_side = 10
short_borrow_bps_per_day = 0
[outputs]
derived_output_root = "{tmp_path / 'derived'}"
results_output_root = "{tmp_path / 'results'}"
experiment_registry = "{tmp_path / 'experiments.toml'}"
[[companies]]
symbol = "AAPL"
name = "Apple Inc."
aliases = ["Apple", "AAPL"]
"""
    )

    result = asyncio.run(
        run_trading_strategy(
            config_path,
            newsapi_client=None,
            tavily_client=None,
            llm_client=FakeLlm(),
            price_loader=lambda _config: _prices()[:2],
        )
    )

    assert result.accepted_article_count == 3
    assert result.return_count == 2
    decisions = (result.results_dir / "trading_decisions.csv").read_text()
    assert decisions.count("positive_mean_meets_threshold") == 2
    returns = (result.results_dir / "returns.csv").read_text()
    assert "net_strategy_return" in returns
    manifest = json.loads((result.results_dir / "run_manifest.json").read_text())
    assert manifest["settings"]["provider"] == "ollama"
    assert manifest["settings"]["decision_policy"]["threshold"] == 0.5
    assert manifest["counts"]["traded_decisions"] == 2


def test_fixed_config_loads() -> None:
    config = load_trading_config("configs/trading_pilot_3co.toml")
    assert [company.symbol for company in config.companies] == ["AAPL", "AMZN", "TSLA"]
    assert config.horizons == (1, 2, 3, 4, 5, 6, 7)
    assert config.provider == "openrouter"
    assert config.lseg_corpus_manifest is None
    assert config.decision_policy_enabled is False


def test_broadened_reviewed_config_loads_fixed_panel() -> None:
    config = load_trading_config("configs/trading_pilot_8co_reviewed.toml")

    assert [company.symbol for company in config.companies] == ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "JPM", "XOM", "BA"]
    assert config.screening_overrides_path == Path("configs/trading_pilot_8co_screening_overrides.toml")


def test_index_fallback_disabled_by_default() -> None:
    assert load_trading_config("configs/trading_pilot_3co.toml").index_fallback is None


def test_index_fallback_config_parses(tmp_path: Path) -> None:
    config_path = tmp_path / "trade.toml"
    base = Path("configs/trading_pilot_3co.toml").read_text()
    config_path.write_text(base + '\n[index_fallback]\nenabled = true\nsymbol = "^GSPC"\nmin_texts = 4\n')
    config = load_trading_config(config_path)
    assert config.index_fallback is not None
    assert config.index_fallback.symbol == "^GSPC"
    assert config.index_fallback.min_texts == 4


def test_index_fallback_enabled_requires_symbol(tmp_path: Path) -> None:
    config_path = tmp_path / "trade.toml"
    base = Path("configs/trading_pilot_3co.toml").read_text()
    config_path.write_text(base + "\n[index_fallback]\nenabled = true\nmin_texts = 2\n")
    with pytest.raises(TradingStrategyError, match="index_fallback.symbol is required"):
        load_trading_config(config_path)


class CapturingLlm:
    """Fake LLM that records the text it was asked to score."""

    def __init__(self) -> None:
        self.sentences: list[str] = []

    async def classify(self, model_id, prompt, example, **_kwargs):
        self.sentences.append(example.sentence)
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


def _apple_article():
    return merge_article_candidates(
        [_candidate("newsapi", _record("https://x.test/apple", "Apple wins large contract", "2026-06-10"))],
        timezone="America/New_York",
        selected_dates=("2026-06-10",),
    )


def test_mask_articles_anonymises_scoring_text_and_counts() -> None:
    articles = _apple_article()
    masked, counts = _mask_articles(articles, {"AAPL": COMPANY})

    assert "Apple" not in masked[0].scoring_text
    assert "[COMPANY]" in masked[0].scoring_text
    assert counts[articles[0].article_id] >= 1
    # The masked copy carries a refreshed content hash.
    assert masked[0].scoring_text_sha256 != articles[0].scoring_text_sha256


def test_score_articles_masked_arm_tags_scores_and_sends_masked_text(tmp_path: Path) -> None:
    articles = _apple_article()
    masked, counts = _mask_articles(articles, {"AAPL": COMPANY})
    client = CapturingLlm()
    prompt = load_prompts(Path("configs/default_prompts.toml"))["target_company_news_label_only"]

    scores = asyncio.run(
        score_articles(
            masked,
            models=("a",),
            prompt=prompt,
            client=client,
            output_path=tmp_path / "masked.jsonl",
            resume_path=None,
            temperature=0.0,
            max_completion_tokens=64,
            concurrency=1,
            retries=0,
            baselines=(),
            scorer_suffix="#masked",
            masking_mode="both",
            entity_masked=True,
            masked_token_counts=counts,
        )
    )

    # The masked arm actually sent anonymised text to the model.
    assert "[COMPANY]" in client.sentences[0]
    assert "Apple" not in client.sentences[0]
    # Scores are tagged for the ablation under a distinct scorer id.
    assert len(scores) == 1
    assert scores[0].scorer_id == "a#masked"
    assert scores[0].entity_masked is True
    assert scores[0].masking_mode == "both"
    assert scores[0].n_masked_tokens == counts[articles[0].article_id]
