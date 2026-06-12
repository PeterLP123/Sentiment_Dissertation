import json
from pathlib import Path

import pytest

from sentiment_benchmark.news_quality import NewsQualityError, discover_news_corpora, summarize_news_quality
from sentiment_benchmark.news_source import article_record_id, normalize_url

GOOD_TEXT = "Bank earnings rose sharply this quarter on stronger lending margins. " * 20
ERROR_PAGE_TEXT = "Oops, something went wrong\nSkip to navigation\n" + ("Menu item\n" * 100)


def _record(url: str, *, text: str | None, text_quality: str | None = None, published_date: str | None = None) -> dict:
    normalized = normalize_url(url)
    record = {
        "record_id": article_record_id(normalized),
        "url": url,
        "normalized_url": normalized,
        "title": "Article",
        "snippet": "Snippet",
        "article_text": text,
        "published_date": published_date,
        "extraction_status": "success" if text else "failed",
    }
    if text_quality is not None:
        record["text_quality"] = text_quality
    return record


def _write_corpus(path: Path, *, query: str, records: list[dict]) -> Path:
    path.mkdir(parents=True)
    with (path / "articles.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    (path / "manifest.json").write_text(json.dumps({"query": query}), encoding="utf-8")
    return path


def _write_matrix(path: Path) -> Path:
    path.write_text(
        """
version = 1
description = "Quality test matrix."

[[queries]]
id = "bank_earnings"
family = "Bank earnings"
query = "bank earnings sentiment"
topic = "news"
max_results = 5
search_depth = "basic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_summarize_news_quality_counts_quality_domains_and_dates(tmp_path: Path) -> None:
    matrix = _write_matrix(tmp_path / "matrix.toml")
    corpus = _write_corpus(
        tmp_path / "news" / "tavily_news_one",
        query="bank earnings sentiment",
        records=[
            _record("https://example.com/good", text=GOOD_TEXT, text_quality="ok", published_date="2026-06-01"),
            _record("https://example.com/junk", text=ERROR_PAGE_TEXT),  # legacy record, assessed on the fly
            _record("https://other.com/missing", text=None),
        ],
    )
    empty = _write_corpus(tmp_path / "news" / "tavily_news_two", query="merger news", records=[])

    report = summarize_news_quality([corpus, empty], query_matrix_path=matrix)

    assert report.corpus_count == 2
    assert report.record_count == 3
    assert report.unique_url_count == 3
    assert report.usable_unique_url_count == 1
    assert report.quality_counts == {"error_page": 1, "missing": 1, "ok": 1}
    assert report.published_date_counts == {"none": 2, "search": 1}

    families = {item.family: item for item in report.families}
    assert families["Bank earnings"].record_count == 3
    assert families["Bank earnings"].usable_count == 1
    # Zero-record corpora stay visible as coverage gaps.
    assert families["merger news"].record_count == 0

    domains = {item.domain: item for item in report.domains}
    assert domains["example.com"].record_count == 2
    assert domains["example.com"].usable_count == 1
    assert domains["other.com"].usable_count == 0


def test_summarize_news_quality_screens_ticker_families_for_entity_relevance(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.toml"
    matrix.write_text(
        """
version = 1
description = "Ticker panel test matrix."

[[queries]]
id = "us_aapl"
family = "AAPL — Apple"
query = "apple aapl stock news"
topic = "news"
max_results = 5
search_depth = "basic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    on_topic = _record(
        "https://example.com/apple",
        text="Apple reported record revenue. Apple shares rose after the iPhone maker raised guidance.",
        text_quality="ok",
    )
    on_topic["title"] = "Apple beats expectations"
    off_topic = _record(
        "https://example.com/samsung",
        text="Samsung unveiled a new chip plant. The Korean group expects semiconductor demand to recover.",
        text_quality="ok",
    )
    off_topic["title"] = "Samsung expands chip output"
    unusable = _record("https://example.com/missing", text=None)
    corpus = _write_corpus(
        tmp_path / "news" / "tavily_news_aapl",
        query="apple aapl stock news",
        records=[on_topic, off_topic, unusable],
    )

    report = summarize_news_quality([corpus], query_matrix_path=matrix)

    assert report.entity_relevance_counts == {"off_topic": 1, "on_topic": 1}
    families = {item.family: item for item in report.families}
    family = families["AAPL — Apple"]
    assert family.usable_count == 2
    assert family.screened_count == 2  # the unusable record is never scored
    assert family.on_topic_count == 1


def test_summarize_news_quality_leaves_thematic_families_unscreened(tmp_path: Path) -> None:
    matrix = _write_matrix(tmp_path / "matrix.toml")
    corpus = _write_corpus(
        tmp_path / "news" / "tavily_news_one",
        query="bank earnings sentiment",
        records=[_record("https://example.com/good", text=GOOD_TEXT, text_quality="ok")],
    )

    report = summarize_news_quality([corpus], query_matrix_path=matrix)

    assert report.entity_relevance_counts == {}
    families = {item.family: item for item in report.families}
    assert families["Bank earnings"].screened_count == 0
    assert families["Bank earnings"].on_topic_count == 0


def test_summarize_news_quality_detects_shared_boilerplate_prefix(tmp_path: Path) -> None:
    chrome = "Site Chrome Navigation Menu Markets Watchlist Subscribe " * 10
    corpus = _write_corpus(
        tmp_path / "news" / "tavily_news_one",
        query="bank earnings sentiment",
        records=[
            _record("https://templated.com/a", text=chrome + GOOD_TEXT, text_quality="ok"),
            _record("https://templated.com/b", text=chrome + "Totally different article body. " * 30, text_quality="ok"),
            _record("https://clean.com/a", text="Alpha " + GOOD_TEXT, text_quality="ok"),
            _record("https://clean.com/b", text="Beta " + GOOD_TEXT, text_quality="ok"),
            # The same URL refetched with identical text must not register as a template.
            _record("https://single.com/a", text=GOOD_TEXT, text_quality="ok"),
            _record("https://single.com/a", text=GOOD_TEXT, text_quality="ok"),
        ],
    )

    report = summarize_news_quality([corpus])

    domains = {item.domain: item for item in report.domains}
    assert domains["templated.com"].shared_prefix_chars >= len(chrome)
    assert domains["clean.com"].shared_prefix_chars < 10
    assert domains["single.com"].shared_prefix_chars == 0


def test_discover_news_corpora_skips_non_corpus_entries(tmp_path: Path) -> None:
    root = tmp_path / "news"
    corpus = _write_corpus(root / "tavily_news_one", query="bank earnings sentiment", records=[])
    (root / "not_a_corpus").mkdir()
    (root / ".gitkeep").touch()

    assert discover_news_corpora(root) == [corpus]
    assert discover_news_corpora(tmp_path / "missing") == []


def test_summarize_news_quality_rejects_empty_and_invalid_sources(tmp_path: Path) -> None:
    with pytest.raises(NewsQualityError, match="no Tavily corpus directories"):
        summarize_news_quality([])

    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(NewsQualityError, match="not a Tavily corpus directory"):
        summarize_news_quality([bad])
