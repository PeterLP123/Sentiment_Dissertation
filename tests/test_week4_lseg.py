from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _load_week4_lseg(monkeypatch, fake_news=None):
    fake_lseg = types.ModuleType("lseg")
    fake_data = types.ModuleType("lseg.data")
    fake_data.news = fake_news
    fake_data.open_session = lambda: None
    fake_data.close_session = lambda: None
    fake_lseg.data = fake_data
    monkeypatch.setitem(sys.modules, "lseg", fake_lseg)
    monkeypatch.setitem(sys.modules, "lseg.data", fake_data)

    path = Path(__file__).resolve().parents[1] / "week_4_tasks" / "LSEG.py"
    spec = importlib.util.spec_from_file_location("week4_lseg_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daily_windows_are_inclusive_calendar_days(monkeypatch) -> None:
    module = _load_week4_lseg(monkeypatch)

    windows = module.daily_windows("2026-06-11T00:00:00", "2026-06-13T23:59:59")

    assert windows == [
        (module.date(2026, 6, 11), "2026-06-11T00:00:00", "2026-06-11T23:59:59"),
        (module.date(2026, 6, 12), "2026-06-12T00:00:00", "2026-06-12T23:59:59"),
        (module.date(2026, 6, 13), "2026-06-13T00:00:00", "2026-06-13T23:59:59"),
    ]


def test_story_type_from_id(monkeypatch) -> None:
    module = _load_week4_lseg(monkeypatch)

    assert module.story_type_from_id("urn:link:webnews:20260624:nNRA104jkw:0") == "webnews"
    assert module.story_type_from_id("urn:newsml:social:20260624:nTWT5symmw:1") == "social"
    assert module.story_type_from_id("urn:newsml:reuters.com:20260624:nFWN42W0G3:1") == "reuters"
    assert module.story_type_from_id("urn:newsml:newsroom:20260624:nNRA0zzlhh:0") == "newsroom"
    assert module.story_type_from_id(pd.NA) == "unknown"


def test_load_companies_uses_built_ins_when_default_csv_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_week4_lseg(monkeypatch)
    missing_default = tmp_path / "week4_companies.csv"
    monkeypatch.setattr(module, "DEFAULT_COMPANIES_CSV", missing_default)

    companies = module.load_companies()

    assert ("MSFT", "MSFT.O", "R:MSFT.O and Language:LEN") in companies
    assert ("AAPL", "AAPL.O", "R:AAPL.O and Language:LEN") in companies


def test_fetch_daily_news_requests_each_day(monkeypatch) -> None:
    class FakeNews:
        def __init__(self) -> None:
            self.headline_calls: list[tuple[str, str, str, int]] = []

        def get_headlines(self, *, query, start, end, count):
            self.headline_calls.append((query, start, end, count))
            day = start[:10]
            return pd.DataFrame(
                [
                    {
                        "storyId": f"urn:newsml:newsroom:{day}:0",
                        "sourceCode": "NS:TEST",
                        "headline": f"Headline for {day}",
                    }
                ],
                index=pd.to_datetime([f"{day}T12:00:00"]),
            )

        def get_story(self, story_id):
            return f"<p>{story_id} body.</p>"

    fake_news = FakeNews()
    module = _load_week4_lseg(monkeypatch, fake_news=fake_news)

    dataframe = module.fetch_daily_news(
        "R:MSFT.O and Language:LEN",
        5,
        "2026-06-11T00:00:00",
        "2026-06-13T23:59:59",
        company_symbol="MSFT",
        ric="MSFT.O",
    )

    assert fake_news.headline_calls == [
        ("R:MSFT.O and Language:LEN", "2026-06-11T00:00:00", "2026-06-11T23:59:59", 5),
        ("R:MSFT.O and Language:LEN", "2026-06-12T00:00:00", "2026-06-12T23:59:59", 5),
        ("R:MSFT.O and Language:LEN", "2026-06-13T00:00:00", "2026-06-13T23:59:59", 5),
    ]
    assert dataframe["request_date"].tolist() == ["2026-06-11", "2026-06-12", "2026-06-13"]
    assert dataframe["story_type"].tolist() == ["newsroom", "newsroom", "newsroom"]
    assert dataframe["story_html"].tolist() == [
        "<p>urn:newsml:newsroom:2026-06-11:0 body.</p>",
        "<p>urn:newsml:newsroom:2026-06-12:0 body.</p>",
        "<p>urn:newsml:newsroom:2026-06-13:0 body.</p>",
    ]


def test_dedupe_story_ids_keeps_same_story_for_different_companies(monkeypatch) -> None:
    module = _load_week4_lseg(monkeypatch)
    dataframe = pd.DataFrame(
        [
            {"company_symbol": "AAPL", "storyId": "shared"},
            {"company_symbol": "AAPL", "storyId": "shared"},
            {"company_symbol": "AAPL", "storyId": None},
            {"company_symbol": "AAPL", "storyId": None},
            {"company_symbol": "MSFT", "storyId": "shared"},
        ]
    )

    deduped, removed = module.dedupe_story_ids(dataframe)

    assert removed == 1
    assert deduped.to_dict("records") == [
        {"company_symbol": "AAPL", "storyId": "shared"},
        {"company_symbol": "AAPL", "storyId": None},
        {"company_symbol": "AAPL", "storyId": None},
        {"company_symbol": "MSFT", "storyId": "shared"},
    ]
