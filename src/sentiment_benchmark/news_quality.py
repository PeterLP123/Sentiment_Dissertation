from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .news_package import QueryMatrix, _query_key, load_query_matrix
from .news_source import (
    DEFAULT_NEWS_OUTPUT_DIR,
    TEXT_QUALITY_NON_ARTICLE,
    TEXT_QUALITY_OK,
    assess_article_text,
    is_listing_url,
    normalize_url,
)

# Only the opening of each article matters for shared-prefix detection, and the
# common prefix can never exceed the shortest sample.
_PREFIX_SAMPLE_CHARS = 4000


class NewsQualityError(RuntimeError):
    """Raised when a news quality report cannot be built."""


@dataclass(frozen=True)
class DomainQuality:
    domain: str
    record_count: int
    usable_count: int
    # Length of the identical opening shared by every usable article from this
    # domain; a large value means extracted text starts with site boilerplate.
    shared_prefix_chars: int = 0


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


def _shared_prefix_chars(text_heads: list[str]) -> int:
    """Longest opening shared by at least two articles from the domain.

    After sorting, the maximal pairwise common prefix always occurs between
    neighbours, so one linear pass finds template boilerplate even when only a
    subset of the domain's articles carries it.
    """
    if len(text_heads) < 2:
        return 0
    ordered = sorted(text_heads)
    return max(len(os.path.commonprefix([first, second])) for first, second in zip(ordered, ordered[1:], strict=False))


def _record_quality(record: dict) -> str:
    # URL check first: corpora fetched before the listing-page filter existed
    # store "ok" for category/index pages.
    url = record.get("normalized_url") or record.get("url")
    if isinstance(url, str) and url.strip() and is_listing_url(url):
        return TEXT_QUALITY_NON_ARTICLE
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
    domain_text_heads: defaultdict[str, list[str]] = defaultdict(list)
    text_head_urls: set[str] = set()
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
                    text = record.get("article_text")
                    # One head per unique URL, so refetches of the same article
                    # do not masquerade as a domain-wide template.
                    if isinstance(text, str) and text and normalized not in text_head_urls:
                        text_head_urls.add(normalized)
                        domain_text_heads[domain].append(text[:_PREFIX_SAMPLE_CHARS])

    domains = [
        DomainQuality(
            domain=domain,
            record_count=count,
            usable_count=domain_usable.get(domain, 0),
            shared_prefix_chars=_shared_prefix_chars(domain_text_heads.get(domain, [])),
        )
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
