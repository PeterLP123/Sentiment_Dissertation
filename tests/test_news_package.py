import csv
import json
from pathlib import Path

import pytest

from sentiment_benchmark.news_package import NewsPackageError, load_query_matrix, package_news_sources
from sentiment_benchmark.news_source import article_record_id, normalize_url


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
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _record(url: str, *, title: str = "Article", text: str = "Full article text", text_quality: str | None = None) -> dict:
    normalized = normalize_url(url)
    record = {
        "record_id": article_record_id(normalized),
        "url": url,
        "normalized_url": normalized,
        "title": title,
        "snippet": "Snippet",
        "article_text": text,
        "published_date": "2026-06-01",
        "score": 0.9,
        "extraction_status": "success" if text else "skipped",
        "search_rank": 1,
        "search_request_id": "search-record-1",
        "extract_request_id": "extract-record-1" if text else None,
    }
    if text_quality is None:
        text_quality = "ok" if text else "missing"
    record["text_quality"] = text_quality
    return record


def _write_corpus(path: Path, *, query: str = "bank earnings sentiment", records: list[dict] | None = None) -> Path:
    path.mkdir(parents=True)
    records = records or [_record("https://example.com/article")]
    with (path / "articles.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "tavily",
                "fetched_at": "2026-06-07T12:00:00+00:00",
                "query": query,
                "record_count": len(records),
                "search_request_id": "search-manifest-1",
                "extract_request_id": "extract-manifest-1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_package_news_sources_writes_metadata_first_package(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tavily_news_one")
    matrix = _write_matrix(tmp_path / "matrix.toml")

    result = package_news_sources(
        [corpus],
        package_id="tavily_shared_v1",
        output_root=tmp_path / "derived",
        query_matrix_path=matrix,
    )

    assert result.unique_source_count == 1
    assert result.input_record_count == 1
    assert result.duplicate_url_count == 0
    assert result.paths.sources_csv.exists()
    assert result.paths.screening_index_csv.exists()
    assert result.paths.package_manifest_json.exists()
    assert result.paths.readme_md.exists()
    assert result.paths.extracts_jsonl is None

    sources = _csv_rows(result.paths.sources_csv)
    assert "article_text" not in sources[0]
    assert sources[0]["query_ids"] == "bank_earnings"
    assert sources[0]["query_families"] == "Bank earnings"
    assert sources[0]["text_quality"] == "ok"
    assert sources[0]["extract_text_available"] == "true"
    assert sources[0]["text_share_scope"] == "metadata_only"

    screening = _csv_rows(result.paths.screening_index_csv)
    assert screening[0]["screening_decision"] == "pending"
    assert screening[0]["optional_sentiment_label"] == ""

    manifest = json.loads(result.paths.package_manifest_json.read_text(encoding="utf-8"))
    assert manifest["package_id"] == "tavily_shared_v1"
    assert manifest["text_policy"] == "metadata"
    assert manifest["unique_source_count"] == 1
    assert manifest["files"]["sources_csv"]["sha256"]
    assert manifest["files"]["package_manifest_json"]["sha256"] is None


def test_package_news_sources_dedupes_urls_and_aggregates_provenance(tmp_path: Path) -> None:
    first = _write_corpus(tmp_path / "tavily_news_one", records=[_record("https://example.com/article#section")])
    second = _write_corpus(
        tmp_path / "tavily_news_two",
        query="market volatility",
        records=[_record("https://example.com/article", title="Duplicate")],
    )

    result = package_news_sources([first, second], package_id="shared", output_root=tmp_path / "derived", query_matrix_path=None)

    rows = _csv_rows(result.paths.sources_csv)
    assert result.input_record_count == 2
    assert result.unique_source_count == 1
    assert result.duplicate_url_count == 1
    assert "tavily_news_one" in rows[0]["source_corpora"]
    assert "tavily_news_two" in rows[0]["source_corpora"]
    assert rows[0]["queries"] == "bank earnings sentiment; market volatility"
    manifest = json.loads(result.paths.package_manifest_json.read_text(encoding="utf-8"))
    assert manifest["duplicate_url_count"] == 1
    assert manifest["unmatched_queries"] == ["bank earnings sentiment", "market volatility"]


def test_package_news_sources_internal_extracts_policy_writes_extracts_jsonl(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tavily_news_one")

    result = package_news_sources(
        [corpus],
        package_id="shared",
        output_root=tmp_path / "derived",
        query_matrix_path=None,
        text_policy="internal-extracts",
    )

    assert result.paths.extracts_jsonl is not None
    extract_rows = result.paths.extracts_jsonl.read_text(encoding="utf-8").splitlines()
    payload = json.loads(extract_rows[0])
    assert payload["article_text"] == "Full article text"
    assert payload["text_share_scope"] == "internal_review_only"
    manifest = json.loads(result.paths.package_manifest_json.read_text(encoding="utf-8"))
    assert manifest["text_policy"] == "internal-extracts"
    assert manifest["files"]["extracts_jsonl"]["sha256"]


def test_package_news_sources_flags_legacy_junk_text_as_unusable(tmp_path: Path) -> None:
    error_page = "Oops, something went wrong\nSkip to navigation\n" + ("Menu item\n" * 200)
    stub = "Short stub body."
    good = "Bank earnings rose sharply this quarter. " * 30
    corpus = _write_corpus(
        tmp_path / "tavily_news_legacy",
        records=[
            _record("https://example.com/error", text=error_page, text_quality=""),
            _record("https://example.com/stub", text=stub, text_quality=""),
            _record("https://example.com/good", text=good, text_quality=""),
        ],
    )

    result = package_news_sources(
        [corpus],
        package_id="shared",
        output_root=tmp_path / "derived",
        query_matrix_path=None,
        text_policy="internal-extracts",
    )

    rows = {row["url"]: row for row in _csv_rows(result.paths.sources_csv)}
    assert rows["https://example.com/error"]["text_quality"] == "error_page"
    assert rows["https://example.com/error"]["extract_text_available"] == "false"
    assert rows["https://example.com/stub"]["text_quality"] == "too_short"
    assert rows["https://example.com/stub"]["extract_text_available"] == "false"
    assert rows["https://example.com/good"]["text_quality"] == "ok"
    assert rows["https://example.com/good"]["extract_text_available"] == "true"

    screening = {row["url"]: row for row in _csv_rows(result.paths.screening_index_csv)}
    assert screening["https://example.com/error"]["text_quality"] == "error_page"

    assert result.paths.extracts_jsonl is not None
    extract_urls = [json.loads(line)["url"] for line in result.paths.extracts_jsonl.read_text(encoding="utf-8").splitlines()]
    assert extract_urls == ["https://example.com/good"]

    manifest = json.loads(result.paths.package_manifest_json.read_text(encoding="utf-8"))
    assert manifest["usable_source_count"] == 1
    assert manifest["text_quality_counts"] == {"error_page": 1, "ok": 1, "too_short": 1}


def test_package_news_sources_reflags_listing_pages_despite_stored_ok(tmp_path: Path) -> None:
    good = "Bank earnings rose sharply this quarter. " * 30
    corpus = _write_corpus(
        tmp_path / "tavily_news_one",
        records=[
            _record("https://example.com/category/markets/page/2", text=good, text_quality="ok"),
            _record("https://example.com/real-article", text=good, text_quality="ok"),
        ],
    )

    result = package_news_sources(
        [corpus],
        package_id="shared",
        output_root=tmp_path / "derived",
        query_matrix_path=None,
        text_policy="internal-extracts",
    )

    rows = {row["url"]: row for row in _csv_rows(result.paths.sources_csv)}
    assert rows["https://example.com/category/markets/page/2"]["text_quality"] == "non_article"
    assert rows["https://example.com/category/markets/page/2"]["extract_text_available"] == "false"
    assert rows["https://example.com/real-article"]["text_quality"] == "ok"
    assert result.paths.extracts_jsonl is not None
    extract_urls = [json.loads(line)["url"] for line in result.paths.extracts_jsonl.read_text(encoding="utf-8").splitlines()]
    assert extract_urls == ["https://example.com/real-article"]


def test_package_news_sources_rejects_missing_and_malformed_inputs(tmp_path: Path) -> None:
    missing_articles = tmp_path / "missing_articles"
    missing_articles.mkdir()
    (missing_articles / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NewsPackageError, match="articles.jsonl"):
        package_news_sources([missing_articles], package_id="shared", output_root=tmp_path / "derived")

    missing_manifest = tmp_path / "missing_manifest"
    missing_manifest.mkdir()
    (missing_manifest / "articles.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(NewsPackageError, match="manifest.json"):
        package_news_sources([missing_manifest], package_id="shared", output_root=tmp_path / "derived2")

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("{}", encoding="utf-8")
    (malformed / "articles.jsonl").write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(NewsPackageError, match="invalid JSONL"):
        package_news_sources([malformed], package_id="shared", output_root=tmp_path / "derived3")


def test_package_news_sources_rejects_invalid_package_id_and_existing_output(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tavily_news_one")

    with pytest.raises(NewsPackageError, match="package_id"):
        package_news_sources([corpus], package_id="../bad", output_root=tmp_path / "derived")

    existing = tmp_path / "derived" / "shared"
    existing.mkdir(parents=True)
    with pytest.raises(NewsPackageError, match="already exists"):
        package_news_sources([corpus], package_id="shared", output_root=tmp_path / "derived")


def test_package_news_sources_overwrite_rebuilds_existing_package(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tavily_news_one")

    first = package_news_sources([corpus], package_id="shared", output_root=tmp_path / "derived", query_matrix_path=None)
    stale = first.paths.output_dir / "stale.txt"
    stale.write_text("leftover from previous run", encoding="utf-8")

    with pytest.raises(NewsPackageError, match="already exists"):
        package_news_sources([corpus], package_id="shared", output_root=tmp_path / "derived", query_matrix_path=None)

    result = package_news_sources(
        [corpus],
        package_id="shared",
        output_root=tmp_path / "derived",
        query_matrix_path=None,
        overwrite=True,
    )

    assert result.paths.output_dir == first.paths.output_dir
    assert result.unique_source_count == 1
    assert result.paths.sources_csv.exists()
    assert result.paths.package_manifest_json.exists()
    assert not stale.exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
version = 1
description = "bad"
[[queries]]
id = "same"
family = "A"
query = "first"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
[[queries]]
id = "same"
family = "B"
query = "second"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
""",
            "duplicate query matrix id",
        ),
        (
            """
version = 1
description = "bad"
[[queries]]
id = "one"
family = "A"
query = "same query"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
[[queries]]
id = "two"
family = "B"
query = "same query"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
""",
            "duplicate query matrix query",
        ),
        (
            """
version = 1
description = "bad"
[[queries]]
id = "one"
family = "A"
query = "first"
topic = "bogus"
time_range = "month"
max_results = 1
search_depth = "basic"
""",
            "invalid topic",
        ),
        (
            """
version = 1
description = "bad"
[[queries]]
id = "one"
family = "A"
query = "first"
topic = "news"
time_range = "decade"
max_results = 1
search_depth = "basic"
""",
            "invalid time_range",
        ),
        (
            """
version = 1
description = "bad"
[[queries]]
id = "one"
family = "A"
query = "first"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "deep"
""",
            "invalid search_depth",
        ),
        (
            """
version = 1
description = "bad"
[[queries]]
id = "one"
family = "A"
query = "first"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
extract_depth = "deepest"
""",
            "invalid extract_depth",
        ),
    ],
)
def test_load_query_matrix_validates_entries(tmp_path: Path, body: str, message: str) -> None:
    matrix = tmp_path / "matrix.toml"
    matrix.write_text(body.strip() + "\n", encoding="utf-8")

    with pytest.raises(NewsPackageError, match=message):
        load_query_matrix(matrix)
