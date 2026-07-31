from pathlib import Path

from sentiment_benchmark.lseg_source import load_lseg_collection_config

ROOT = Path(__file__).resolve().parents[1]


def test_sector_33_backward_extension_matches_frozen_base_universe() -> None:
    base = load_lseg_collection_config(ROOT / "configs/lseg_us_sector_33_6m.toml")
    extension = load_lseg_collection_config(ROOT / "configs/lseg_us_sector_33_back2m_headlines.toml")

    assert {(company.symbol, company.ric) for company in extension.companies} == {
        (company.symbol, company.ric) for company in base.companies
    }
    assert extension.start == "2025-10-26T00:00:00Z"
    assert extension.end == base.start
    assert extension.fetch_story_bodies is False
    assert extension.max_requests_per_run == 8000


def test_additional_sector_panel_is_disjoint_and_frozen() -> None:
    base = load_lseg_collection_config(ROOT / "configs/lseg_us_sector_33_6m.toml")
    additions = load_lseg_collection_config(ROOT / "configs/lseg_us_sector_add11_8m_headlines.toml")

    assert len(additions.companies) == 11
    assert {company.symbol for company in additions.companies}.isdisjoint(
        {company.symbol for company in base.companies}
    )
    assert additions.start == "2025-10-26T00:00:00Z"
    assert additions.end == base.end
    assert additions.fetch_story_bodies is False
    assert additions.max_requests_per_run == 8000
    assert all("Source:" not in company.news_query for company in additions.companies)
