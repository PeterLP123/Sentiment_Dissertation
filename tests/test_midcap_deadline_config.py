from pathlib import Path

from sentiment_benchmark.lseg_source import load_lseg_collection_config

CONFIG = Path("configs/lseg_us_midcap_22_1y.toml")


def test_midcap_deadline_config_is_bounded_and_headline_only() -> None:
    config = load_lseg_collection_config(CONFIG)

    assert config.collection_id == "us_midcap_22_1y"
    assert len(config.companies) == 22
    assert config.window_days == 7
    assert config.fetch_story_bodies is False
    assert config.max_requests_per_run == 8000
    assert config.start == "2025-06-26T00:00:00Z"
    assert config.end == "2026-06-26T00:00:00Z"
    assert len({company.symbol for company in config.companies}) == 22
    assert all("Source:RTRS" in company.news_query for company in config.companies)
    assert all("Language:LEN" in company.news_query for company in config.companies)


def test_midcap_deadline_config_avoids_ambiguous_ticker_only_aliases() -> None:
    config = load_lseg_collection_config(CONFIG)
    ambiguous = {"BIO", "NOV", "POST", "SF", "SON"}

    for company in config.companies:
        if company.symbol in ambiguous:
            assert company.symbol not in company.aliases
