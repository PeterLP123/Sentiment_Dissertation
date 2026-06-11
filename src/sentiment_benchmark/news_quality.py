from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .news_package import QueryMatrix, _query_key, load_query_matrix
from .news_source import DEFAULT_NEWS_OUTPUT_DIR, TEXT_QUALITY_OK, assess_article_text, normalize_url


class NewsQualityError(RuntimeError):
    """Raised when a news quality report cannot be built."""


@dataclass(frozen=True)
class DomainQuality:
    domain: str
    record_count: int
    usable_count: int


@dataclass(frozen=True)
class FamilyQuality:
    family: str
    record_count: int
    usable_count: int


@dataclass(frozen=True)
class NewsQualityReport:
    corpus_count: int
    record_count: int
    unique_url_count: int
    usable_unique_url_count: int
    quality_counts: dict[str, int] = field(default_factory=dict)
    published_date_counts: dict[str, int] = field(default_factory=dict)
    domains: list[DomainQuality] = field(default_factory=list)
    families: list[FamilyQuality] = field(default_factory=list)


def discover_news_corpora(root: str | Path = DEFAULT_NEWS_OUTPUT_DIR) -> list[Path]:
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    corpora = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "articles.jsonl").exists() and (path / "manifest.json").exists()
    ]
    return corpora


def _record_quality(record: dict) -> str:
    stored = record.get("text_quality")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    text = record.get("article_text")
    quality, _ = assess_article_text(text if isinstance(text, str) else None)
    return quality


def _record_date_source(record: dict) -> str:
    stored = record.get("published_date_source")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    return "search" if record.get("published_date") else "none"


def summarize_news_quality(
    sources: list[str | Path],
    *,
    query_matrix_path: str | Path | None = None,
) -> NewsQualityReport:
    if not sources:
        raise NewsQualityError("no Tavily corpus directories found; fetch news first or pass --source")
    matrix: QueryMatrix | None = load_query_matrix(query_matrix_path) if query_matrix_path else None

    record_count = 0
    quality_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    domain_records: Counter[str] = Counter()
    domain_usable: Counter[str] = Counter()
    family_records: Counter[str] = Counter()
    family_usable: Counter[str] = Counter()
    unique_urls: set[str] = set()
    usable_urls: set[str] = set()

    for source in sources:
        path = Path(source)
        articles_path = path / "articles.jsonl"
        manifest_path = path / "manifest.json"
        if not articles_path.exists() or not manifest_path.exists():
            raise NewsQualityError(f"not a Tavily corpus directory (missing articles.jsonl or manifest.json): {path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NewsQualityError(f"invalid manifest JSON at {manifest_path}: {exc}") from exc
        query = manifest.get("query") if isinstance(manifest, dict) else None
        family = str(query or path.name)
        if matrix and isinstance(query, str):
            entry = matrix.entries_by_query.get(_query_key(query))
            if entry is not None:
                family = entry.family
        # Keep zero-record corpora visible so queries that return nothing
        # show up as a coverage gap instead of disappearing from the report.
        family_records[family] += 0

        with articles_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise NewsQualityError(f"invalid JSONL record at {articles_path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise NewsQualityError(f"JSONL record must be an object at {articles_path}:{line_number}")

                record_count += 1
                quality = _record_quality(record)
                quality_counts[quality] += 1
                date_counts[_record_date_source(record)] += 1
                family_records[family] += 1

                raw_url = record.get("normalized_url") or record.get("url")
                normalized = normalize_url(raw_url) if isinstance(raw_url, str) and raw_url.strip() else ""
                domain = urlparse(normalized).netloc if normalized else "(no url)"
                domain_records[domain] += 1
                if normalized:
                    unique_urls.add(normalized)
                if quality == TEXT_QUALITY_OK:
                    domain_usable[domain] += 1
                    family_usable[family] += 1
                    if normalized:
                        usable_urls.add(normalized)

    domains = [
        DomainQuality(domain=domain, record_count=count, usable_count=domain_usable.get(domain, 0))
        for domain, count in domain_records.most_common()
    ]
    families = [
        FamilyQuality(family=family, record_count=count, usable_count=family_usable.get(family, 0))
        for family, count in sorted(family_records.items())
    ]
    return NewsQualityReport(
        corpus_count=len(sources),
        record_count=record_count,
        unique_url_count=len(unique_urls),
        usable_unique_url_count=len(usable_urls),
        quality_counts=dict(sorted(quality_counts.items())),
        published_date_counts=dict(sorted(date_counts.items())),
        domains=domains,
        families=families,
    )
