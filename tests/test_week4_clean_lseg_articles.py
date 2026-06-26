from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_cleaner():
    path = Path(__file__).resolve().parents[1] / "week_4_tasks" / "clean_lseg_articles.py"
    spec = importlib.util.spec_from_file_location("week4_clean_lseg_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_clean_file_handles_legacy_story_column_and_drops_raw_html(tmp_path: Path) -> None:
    module = _load_cleaner()
    source = tmp_path / "MSFT.csv"
    destination = tmp_path / "MSFT_cleaned.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-24T12:00:00",
                "company_symbol": "MSFT",
                "storyId": "urn:newsml:newsroom:20260624:nTEST:0",
                "headline": "Microsoft expands testing",
                "story": (
                    "<article><p>June 24 (Reuters) - Microsoft reported a larger cloud contract "
                    "with long-term revenue implications.</p><p>(c) Copyright Thomson Reuters 2026.</p></article>"
                ),
                "story_error": "",
            },
            {
                "timestamp": "2026-06-24T13:00:00",
                "company_symbol": "MSFT",
                "storyId": "urn:link:webnews:20260624:nTEST:0",
                "headline": "External link only",
                "story": "",
                "story_error": "",
            },
        ]
    ).to_csv(source, index=False)

    summary = module.clean_file(source, destination, min_chars=40)

    cleaned = pd.read_csv(destination)
    assert summary["rows_in"] == 2
    assert summary["quality_ok"] == 1
    assert summary["quality_missing"] == 1
    assert "story" not in cleaned.columns
    assert cleaned["story_type"].tolist() == ["newsroom", "webnews"]
    assert cleaned["cleaning_quality"].tolist() == ["ok", "missing"]
    assert cleaned["article_story_type"].tolist() == [True, False]
    assert cleaned["usable_article"].tolist() == [True, False]
    assert cleaned.loc[0, "removed_trailing_lines"] == 1
    assert len(cleaned.loc[0, "clean_text_sha256"]) == 64


def test_standalone_cleaner_removes_provider_boilerplate() -> None:
    module = _load_cleaner()
    raw = """
    <article>
      <p>June 24 (Reuters) - The company reported higher revenue and a stronger outlook.</p>
      <p>Created by www.buysellsignals.com</p>
      <p>Click here https://www.buysellsignals.com/bst/disclaimer</p>
      <p>Disclaimer: While this document is based on information sources which are considered reliable, it is generic.</p>
      <p>Management said demand remained resilient.</p>
    </article>
    """

    result = module.clean_news_html(raw, min_chars=20)

    assert result.quality == module.QUALITY_OK
    assert "higher revenue" in result.text
    assert "demand remained resilient" in result.text
    assert "buysellsignals" not in result.text.lower()
    assert "generic" not in result.text.lower()


def test_standalone_cleaner_marks_provider_link_only_as_boilerplate() -> None:
    module = _load_cleaner()

    result = module.clean_news_html(
        "<p>https://filings.ica.int.thomsonreuters.com/filings.viewer/example</p>",
        min_chars=10,
    )

    assert result.quality == module.QUALITY_BOILERPLATE_ONLY
    assert result.text == ""


def test_clean_dataframe_ok_only_keeps_all_valid_story_types() -> None:
    module = _load_cleaner()
    dataframe = pd.DataFrame(
        [
            {
                "storyId": "urn:newsml:reuters.com:20260624:nTEST:1",
                "story_html": "<p>June 24 (Reuters) - Nvidia signed a material supply agreement.</p>",
            },
            {
                "storyId": "urn:newsml:social:20260624:nTEST:1",
                "story_html": "<p>Short.</p>",
            },
        ]
    )

    cleaned, story_column = module.clean_dataframe(dataframe, min_chars=30, ok_only=True)

    assert story_column == "story_html"
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["story_type"] == "reuters"
    assert cleaned.iloc[0]["cleaning_quality"] == "ok"


def test_clean_dataframe_articles_only_keeps_valid_article_story_types() -> None:
    module = _load_cleaner()
    dataframe = pd.DataFrame(
        [
            {
                "storyId": "urn:newsml:reuters.com:20260624:nTEST:1",
                "story_html": "<p>June 24 (Reuters) - Nvidia signed a material supply agreement.</p>",
            },
            {
                "storyId": "urn:newsml:social:20260624:nTEST:1",
                "story_html": "<p>Social update has enough words to pass the minimum character threshold.</p>",
            },
            {
                "storyId": "urn:link:webnews:20260624:nTEST:1",
                "story_html": "",
            },
        ]
    )

    cleaned, _story_column = module.clean_dataframe(dataframe, min_chars=30, articles_only=True)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["story_type"] == "reuters"
    assert bool(cleaned.iloc[0]["usable_article"])
