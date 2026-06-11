from __future__ import annotations

import csv
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from .news_source import (
    MAX_TAVILY_RESULTS,
    NEWS_SEARCH_DEPTHS,
    NEWS_TIME_RANGES,
    NEWS_TOPICS,
    article_record_id,
    normalize_url,
)

NEWS_PACKAGE_SCHEMA_VERSION = 1
DEFAULT_NEWS_PACKAGE_OUTPUT_DIR = Path("Data/derived")
DEFAULT_NEWS_PACKAGE_ID = "tavily_shared_v1"
DEFAULT_QUERY_MATRIX_PATH = Path("configs/tavily_query_matrix.toml")

NewsTextPolicy = Literal["metadata", "internal-extracts"]
NEWS_TEXT_POLICIES: tuple[NewsTextPolicy, ...] = ("metadata", "internal-extracts")

SOURCES_FIELDNAMES = [
    "source_id",
    "record_id",
    "url",
    "normalized_url",
    "source_domain",
    "title",
    "snippet",
    "published_date",
    "query_ids",
    "query_families",
    "queries",
    "source_corpora",
    "fetched_at_values",
    "tavily_search_request_ids",
    "tavily_extract_request_ids",
    "search_rank_first",
    "score_first",
    "extraction_status",
    "extract_text_available",
    "article_text_chars",
    "article_text_sha256",
    "text_share_scope",
]

SCREENING_FIELDNAMES = [
    "source_id",
    "title",
    "url",
    "source_domain",
    "query_families",
    "published_date",
    "extraction_status",
    "extract_text_available",
    "screening_decision",
    "screening_reason",
    "reviewer_notes",
    "optional_sentiment_label",
    "optional_labeler",
    "optional_label_notes",
]


class NewsPackageError(RuntimeError):
    """Raised when Tavily news package generation cannot complete."""


@dataclass(frozen=True)
class QueryMatrixEntry:
    id: str
    family: str
    query: str
    topic: str
    time_range: str | None
    max_results: int
    search_depth: str
    coverage_target: str
    include_domains: tuple[str, ...]
    exclude_domains: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class QueryMatrix:
    path: Path
    version: int
    description: str
    entries: tuple[QueryMatrixEntry, ...]
    entries_by_id: dict[str, QueryMatrixEntry]
    entries_by_query: dict[str, QueryMatrixEntry]


@dataclass(frozen=True)
class NewsPackagePaths:
    output_dir: Path
    sources_csv: Path
    screening_index_csv: Path
    package_manifest_json: Path
    readme_md: Path
    extracts_jsonl: Path | None = None


@dataclass(frozen=True)
class NewsPackageResult:
    paths: NewsPackagePaths
    package_id: str
    text_policy: NewsTextPolicy
    unique_source_count: int
    input_record_count: int
    duplicate_url_count: int


@dataclass
class _Corpus:
    path: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    query_entry: QueryMatrixEntry | None


@dataclass
class _RecordContext:
    record: dict[str, Any]
    corpus: _Corpus
    normalized_url: str
    record_id: str
    source_corpus: str
    query: str
    fetched_at: str
    search_request_id: str
    extract_request_id: str


@dataclass
class _Aggregate:
    canonical: _RecordContext
    records: list[_RecordContext]
    text_context: _RecordContext | None


def sanitize_package_id(value: str) -> str:
    package_id = re.sub(r"\s+", "-", value.strip().lower())
    package_id = package_id.strip("-_")
    if not package_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", package_id):
        raise NewsPackageError("package_id must contain only lowercase letters, numbers, hyphens, and underscores")
    return package_id


def _clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise NewsPackageError(f"{name} must be a list of strings")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise NewsPackageError(f"{name} must be a list of strings")
        item = item.strip().lower()
        if item:
            cleaned.append(item)
    return tuple(dict.fromkeys(cleaned))


def _query_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _validate_query_id(value: str) -> str:
    query_id = value.strip()
    if not query_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", query_id):
        raise NewsPackageError("query matrix id must contain only lowercase letters, numbers, hyphens, and underscores")
    return query_id


def load_query_matrix(path: str | Path | None = DEFAULT_QUERY_MATRIX_PATH) -> QueryMatrix | None:
    if path is None:
        return None
    matrix_path = Path(path)
    if not matrix_path.exists():
        return None
    try:
        with matrix_path.open("rb") as file:
            payload = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise NewsPackageError(f"Invalid query matrix TOML at {matrix_path}: {exc}") from exc

    version = payload.get("version")
    if version != 1:
        raise NewsPackageError("query matrix version must be 1")
    description = _clean_string(payload.get("description"))
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise NewsPackageError("query matrix must contain at least one [[queries]] entry")

    ids: set[str] = set()
    queries: set[str] = set()
    entries: list[QueryMatrixEntry] = []
    entries_by_id: dict[str, QueryMatrixEntry] = {}
    entries_by_query: dict[str, QueryMatrixEntry] = {}
    for index, raw_entry in enumerate(raw_queries, start=1):
        if not isinstance(raw_entry, dict):
            raise NewsPackageError(f"query matrix entry {index} must be a table")
        query_id = _validate_query_id(_clean_string(raw_entry.get("id")))
        if query_id in ids:
            raise NewsPackageError(f"duplicate query matrix id: {query_id}")
        ids.add(query_id)

        query = _clean_string(raw_entry.get("query"))
        if not query:
            raise NewsPackageError(f"query matrix entry {query_id} is missing query")
        query_key = _query_key(query)
        if query_key in queries:
            raise NewsPackageError(f"duplicate query matrix query: {query}")
        queries.add(query_key)

        family = _clean_string(raw_entry.get("family"))
        if not family:
            raise NewsPackageError(f"query matrix entry {query_id} is missing family")
        topic = _clean_string(raw_entry.get("topic") or "news").lower()
        if topic not in NEWS_TOPICS:
            raise NewsPackageError(f"query matrix entry {query_id} has invalid topic: {topic}")
        time_range = _optional_string(raw_entry.get("time_range")).lower() or None
        if time_range is not None and time_range not in NEWS_TIME_RANGES:
            raise NewsPackageError(f"query matrix entry {query_id} has invalid time_range: {time_range}")
        search_depth = _clean_string(raw_entry.get("search_depth") or "basic").lower()
        if search_depth not in NEWS_SEARCH_DEPTHS:
            raise NewsPackageError(f"query matrix entry {query_id} has invalid search_depth: {search_depth}")
        max_results = _optional_int(raw_entry.get("max_results"))
        if max_results is None or max_results < 1 or max_results > MAX_TAVILY_RESULTS:
            raise NewsPackageError(f"query matrix entry {query_id} max_results must be between 1 and {MAX_TAVILY_RESULTS}")

        entry = QueryMatrixEntry(
            id=query_id,
            family=family,
            query=query,
            topic=topic,
            time_range=time_range,
            max_results=max_results,
            search_depth=search_depth,
            coverage_target=_optional_string(raw_entry.get("coverage_target")),
            include_domains=_optional_string_list(raw_entry.get("include_domains", []), name=f"{query_id}.include_domains"),
            exclude_domains=_optional_string_list(raw_entry.get("exclude_domains", []), name=f"{query_id}.exclude_domains"),
            notes=_optional_string(raw_entry.get("notes")),
        )
        entries.append(entry)
        entries_by_id[query_id] = entry
        entries_by_query[query_key] = entry
    return QueryMatrix(
        path=matrix_path,
        version=version,
        description=description,
        entries=tuple(entries),
        entries_by_id=entries_by_id,
        entries_by_query=entries_by_query,
    )


def package_news_sources(
    sources: list[str | Path],
    *,
    package_id: str = DEFAULT_NEWS_PACKAGE_ID,
    output_root: str | Path = DEFAULT_NEWS_PACKAGE_OUTPUT_DIR,
    query_matrix_path: str | Path | None = DEFAULT_QUERY_MATRIX_PATH,
    text_policy: str = "metadata",
) -> NewsPackageResult:
    resolved_policy = _validate_text_policy(text_policy)
    resolved_package_id = sanitize_package_id(package_id)
    output_dir = Path(output_root) / resolved_package_id
    if output_dir.exists():
        raise NewsPackageError(f"output directory already exists: {output_dir}")
    if not sources:
        raise NewsPackageError("at least one source corpus directory is required")

    query_matrix = load_query_matrix(query_matrix_path)
    corpora = [_load_corpus(Path(source), query_matrix) for source in sources]
    contexts = _record_contexts(corpora)
    aggregates = _aggregate_records(contexts)

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        sources_csv = output_dir / "sources.csv"
        screening_csv = output_dir / "screening_index.csv"
        manifest_json = output_dir / "package_manifest.json"
        readme_md = output_dir / "README.md"
        extracts_jsonl = output_dir / "extracts.jsonl" if resolved_policy == "internal-extracts" else None

        source_rows = [_source_row(aggregate, resolved_policy) for aggregate in aggregates]
        screening_rows = [_screening_row(row) for row in source_rows]
        _write_csv(sources_csv, SOURCES_FIELDNAMES, source_rows)
        _write_csv(screening_csv, SCREENING_FIELDNAMES, screening_rows)
        if extracts_jsonl is not None:
            _write_extracts_jsonl(extracts_jsonl, aggregates)

        readme_md.write_text(_readme_text(resolved_package_id, resolved_policy, len(aggregates), len(contexts)), encoding="utf-8")
        manifest = _manifest(
            package_id=resolved_package_id,
            text_policy=resolved_policy,
            corpora=corpora,
            contexts=contexts,
            aggregates=aggregates,
            query_matrix=query_matrix,
            query_matrix_path=query_matrix_path,
            output_dir=output_dir,
            file_paths=[path for path in (sources_csv, screening_csv, manifest_json, readme_md, extracts_jsonl) if path is not None],
        )
        manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    except Exception:
        for path in sorted(output_dir.glob("*")):
            if path.is_file():
                path.unlink()
        output_dir.rmdir()
        raise

    duplicate_count = len(contexts) - len(aggregates)
    return NewsPackageResult(
        paths=NewsPackagePaths(
            output_dir=output_dir,
            sources_csv=sources_csv,
            screening_index_csv=screening_csv,
            package_manifest_json=manifest_json,
            readme_md=readme_md,
            extracts_jsonl=extracts_jsonl,
        ),
        package_id=resolved_package_id,
        text_policy=resolved_policy,
        unique_source_count=len(aggregates),
        input_record_count=len(contexts),
        duplicate_url_count=duplicate_count,
    )


def _validate_text_policy(value: str) -> NewsTextPolicy:
    policy = value.strip().lower()
    if policy not in NEWS_TEXT_POLICIES:
        raise NewsPackageError(f"text_policy must be one of: {', '.join(NEWS_TEXT_POLICIES)}")
    return policy


def _load_corpus(path: Path, query_matrix: QueryMatrix | None) -> _Corpus:
    if not path.exists() or not path.is_dir():
        raise NewsPackageError(f"source corpus directory does not exist: {path}")
    articles_path = path / "articles.jsonl"
    manifest_path = path / "manifest.json"
    if not articles_path.exists():
        raise NewsPackageError(f"source corpus is missing articles.jsonl: {path}")
    if not manifest_path.exists():
        raise NewsPackageError(f"source corpus is missing manifest.json: {path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NewsPackageError(f"invalid manifest JSON at {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise NewsPackageError(f"manifest must be a JSON object: {manifest_path}")

    records: list[dict[str, Any]] = []
    with articles_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise NewsPackageError(f"invalid JSONL record at {articles_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise NewsPackageError(f"JSONL record must be an object at {articles_path}:{line_number}")
            records.append(record)

    query = _clean_string(manifest.get("query"))
    query_entry = query_matrix.entries_by_query.get(_query_key(query)) if query_matrix and query else None
    return _Corpus(path=path, manifest=manifest, records=records, query_entry=query_entry)


def _record_contexts(corpora: list[_Corpus]) -> list[_RecordContext]:
    contexts: list[_RecordContext] = []
    for corpus in corpora:
        query = _clean_string(corpus.manifest.get("query"))
        fetched_at = _optional_string(corpus.manifest.get("fetched_at"))
        manifest_search_request = _optional_string(corpus.manifest.get("search_request_id"))
        manifest_extract_request = _optional_string(corpus.manifest.get("extract_request_id"))
        for record in corpus.records:
            raw_url = _clean_string(record.get("normalized_url")) or _clean_string(record.get("url"))
            if not raw_url:
                continue
            normalized = normalize_url(raw_url)
            record_id = _clean_string(record.get("record_id")) or article_record_id(normalized)
            contexts.append(
                _RecordContext(
                    record=record,
                    corpus=corpus,
                    normalized_url=normalized,
                    record_id=record_id,
                    source_corpus=str(corpus.path),
                    query=query,
                    fetched_at=fetched_at,
                    search_request_id=_optional_string(record.get("search_request_id")) or manifest_search_request,
                    extract_request_id=_optional_string(record.get("extract_request_id")) or manifest_extract_request,
                )
            )
    return contexts


def _aggregate_records(contexts: list[_RecordContext]) -> list[_Aggregate]:
    aggregates_by_url: dict[str, _Aggregate] = {}
    for context in contexts:
        aggregate = aggregates_by_url.get(context.normalized_url)
        has_text = bool(_article_text(context))
        if aggregate is None:
            aggregates_by_url[context.normalized_url] = _Aggregate(
                canonical=context,
                records=[context],
                text_context=context if has_text else None,
            )
            continue
        aggregate.records.append(context)
        if aggregate.text_context is None and has_text:
            aggregate.text_context = context
    return list(aggregates_by_url.values())


def _source_row(aggregate: _Aggregate, text_policy: NewsTextPolicy) -> dict[str, Any]:
    canonical = aggregate.canonical
    record = canonical.record
    text_context = aggregate.text_context
    article_text = _article_text(text_context) if text_context else ""
    text_hash = _sha256_text(article_text) if article_text else ""
    normalized_url = canonical.normalized_url
    row = {
        "source_id": _source_id(normalized_url),
        "record_id": canonical.record_id,
        "url": _clean_string(record.get("url")) or normalized_url,
        "normalized_url": normalized_url,
        "source_domain": _source_domain(normalized_url),
        "title": _optional_string(record.get("title")),
        "snippet": _optional_string(record.get("snippet")),
        "published_date": _optional_string(record.get("published_date")),
        "query_ids": _join_unique(_query_id(context) for context in aggregate.records),
        "query_families": _join_unique(_query_family(context) for context in aggregate.records),
        "queries": _join_unique(context.query for context in aggregate.records),
        "source_corpora": _join_unique(context.source_corpus for context in aggregate.records),
        "fetched_at_values": _join_unique(context.fetched_at for context in aggregate.records),
        "tavily_search_request_ids": _join_unique(context.search_request_id for context in aggregate.records),
        "tavily_extract_request_ids": _join_unique(context.extract_request_id for context in aggregate.records),
        "search_rank_first": record.get("search_rank") if record.get("search_rank") is not None else "",
        "score_first": record.get("score") if record.get("score") is not None else "",
        "extraction_status": _aggregate_extraction_status(aggregate.records),
        "extract_text_available": _bool_text(bool(article_text)),
        "article_text_chars": len(article_text) if article_text else 0,
        "article_text_sha256": text_hash,
        "text_share_scope": "metadata_only" if text_policy == "metadata" else "internal_review_only",
    }
    return row


def _screening_row(source_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source_row["source_id"],
        "title": source_row["title"],
        "url": source_row["url"],
        "source_domain": source_row["source_domain"],
        "query_families": source_row["query_families"],
        "published_date": source_row["published_date"],
        "extraction_status": source_row["extraction_status"],
        "extract_text_available": source_row["extract_text_available"],
        "screening_decision": "pending",
        "screening_reason": "",
        "reviewer_notes": "",
        "optional_sentiment_label": "",
        "optional_labeler": "",
        "optional_label_notes": "",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_extracts_jsonl(path: Path, aggregates: list[_Aggregate]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for aggregate in aggregates:
            context = aggregate.text_context
            if context is None:
                continue
            article_text = _article_text(context)
            if not article_text:
                continue
            payload = {
                "source_id": _source_id(aggregate.canonical.normalized_url),
                "record_id": aggregate.canonical.record_id,
                "url": _clean_string(context.record.get("url")) or context.normalized_url,
                "normalized_url": context.normalized_url,
                "article_text": article_text,
                "article_text_sha256": _sha256_text(article_text),
                "article_text_chars": len(article_text),
                "source_corpus": context.source_corpus,
                "text_share_scope": "internal_review_only",
            }
            file.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _manifest(
    *,
    package_id: str,
    text_policy: NewsTextPolicy,
    corpora: list[_Corpus],
    contexts: list[_RecordContext],
    aggregates: list[_Aggregate],
    query_matrix: QueryMatrix | None,
    query_matrix_path: str | Path | None,
    output_dir: Path,
    file_paths: list[Path],
) -> dict[str, Any]:
    corpus_entries = []
    for corpus in corpora:
        manifest = corpus.manifest
        query_entry = corpus.query_entry
        corpus_entries.append(
            {
                "path": str(corpus.path),
                "fetched_at": manifest.get("fetched_at"),
                "query": manifest.get("query"),
                "query_id": query_entry.id if query_entry else None,
                "query_family": query_entry.family if query_entry else None,
                "record_count": len(corpus.records),
                "manifest_record_count": manifest.get("record_count"),
                "search_request_id": manifest.get("search_request_id"),
                "extract_request_id": manifest.get("extract_request_id"),
            }
        )
    seen_queries = {
        _query_key(_clean_string(corpus.manifest.get("query")))
        for corpus in corpora
        if _clean_string(corpus.manifest.get("query"))
    }
    matched_queries = set(query_matrix.entries_by_query) if query_matrix else set()
    unmatched = sorted(
        _clean_string(corpus.manifest.get("query"))
        for corpus in corpora
        if _clean_string(corpus.manifest.get("query")) and _query_key(_clean_string(corpus.manifest.get("query"))) not in matched_queries
    )
    return {
        "schema_version": NEWS_PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "created_at": datetime.now(UTC).isoformat(),
        "text_policy": text_policy,
        "source_corpora": corpus_entries,
        "unique_source_count": len(aggregates),
        "input_record_count": len(contexts),
        "duplicate_url_count": len(contexts) - len(aggregates),
        "query_matrix_path": str(query_matrix.path) if query_matrix else (str(query_matrix_path) if query_matrix_path else None),
        "unmatched_queries": sorted(dict.fromkeys(unmatched)),
        "matched_query_count": len(seen_queries & matched_queries),
        "files": {
            _manifest_file_key(path): {
                "path": str(path.relative_to(output_dir)),
                "sha256": None if path.name == "package_manifest.json" else (_file_sha256(path) if path.exists() else None),
            }
            for path in file_paths
        },
        "notes": [
            "This is a metadata-first share package built from existing Tavily corpora.",
            "No Tavily API calls, sentiment labels, model runs, or Data/data.csv changes are performed by package-news.",
            "Full article text is included only when text_policy is internal-extracts and should be shared only with appropriate rights.",
        ],
    }


def _manifest_file_key(path: Path) -> str:
    return {
        "sources.csv": "sources_csv",
        "screening_index.csv": "screening_index_csv",
        "package_manifest.json": "package_manifest_json",
        "README.md": "readme_md",
        "extracts.jsonl": "extracts_jsonl",
    }.get(path.name, path.stem)


def _readme_text(package_id: str, text_policy: NewsTextPolicy, unique_count: int, input_count: int) -> str:
    full_text_note = (
        "This package includes `extracts.jsonl` for internal review only. Check source terms before redistributing full article text."
        if text_policy == "internal-extracts"
        else "This package is metadata-first and does not include full article bodies."
    )
    return f"""# Tavily Shared Dataset Package: {package_id}

Generated from existing Tavily news corpora for colleague review.

## Contents

- `sources.csv`: one row per unique source URL with provenance and compact review metadata.
- `screening_index.csv`: editable review worksheet with screening and optional future label columns.
- `package_manifest.json`: package metadata, source corpora, counts, file hashes, and sharing notes.
- `extracts.jsonl`: present only when full text was explicitly requested for internal review.

## Counts

- Input records: {input_count}
- Unique sources: {unique_count}
- Text policy: `{text_policy}`

## Sharing Note

{full_text_note}

This package is unlabeled source material. Do not treat it as a benchmark dataset until a separate labeling protocol is agreed
and documented.
"""


def _query_id(context: _RecordContext) -> str:
    return context.corpus.query_entry.id if context.corpus.query_entry else ""


def _query_family(context: _RecordContext) -> str:
    return context.corpus.query_entry.family if context.corpus.query_entry else ""


def _source_id(normalized_url: str) -> str:
    return f"src_{article_record_id(normalized_url)}"


def _source_domain(normalized_url: str) -> str:
    return urlparse(normalized_url).netloc.lower()


def _article_text(context: _RecordContext | None) -> str:
    if context is None:
        return ""
    text = context.record.get("article_text")
    return text if isinstance(text, str) and text else ""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_unique(values: Any) -> str:
    seen: dict[str, None] = {}
    for value in values:
        item = str(value).strip() if value is not None else ""
        if item:
            seen.setdefault(item, None)
    return "; ".join(seen)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _aggregate_extraction_status(contexts: list[_RecordContext]) -> str:
    statuses = [_optional_string(context.record.get("extraction_status")) for context in contexts]
    if any(_article_text(context) for context in contexts) or "success" in statuses:
        return "success"
    if "failed" in statuses:
        return "failed"
    if "skipped" in statuses:
        return "skipped"
    return statuses[0] if statuses else ""
