from __future__ import annotations

import csv
import importlib.metadata
import io
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .lseg_source import LsegCollectionConfig, LsegNewsError, config_from_payload, utc_now
from .news_cleaning import CLEANER_VERSION, QUALITY_OK, clean_news_html
from .runtime_metadata import collect_run_environment

LSEG_CORPUS_SCHEMA_VERSION = 1
QUALITY_NON_ENGLISH = "non_english"
QUALITY_INVALID_TIMESTAMP = "invalid_timestamp"


@dataclass(frozen=True)
class LsegCorpusResult:
    derived_dir: Path
    articles_path: Path
    screening_path: Path
    manifest_path: Path
    article_count: int
    eligible_count: int
    resumed: bool = False


def _manifest_file(raw_dir: Path, manifest: dict[str, Any], key: str) -> Path:
    files_value = manifest.get("files")
    files: dict[str, Any] = files_value if isinstance(files_value, dict) else {}
    entry = files.get(key)
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise LsegNewsError(f"raw LSEG manifest is missing files.{key}")
    path = raw_dir / str(entry["path"])
    if not path.exists():
        raise LsegNewsError(f"raw LSEG corpus file is missing: {path}")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise LsegNewsError(f"raw LSEG corpus hash mismatch for {path}: expected {entry['sha256']}, got {actual}")
    return path


def _parse_version_created(value: Any) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _is_english(value: Any, configured_language: str) -> bool:
    if value is None or not str(value).strip():
        return configured_language == "en"
    return str(value).strip().lower() in {"en", "eng", "english"}


def _story_family_id(story_id: str) -> str:
    return re.sub(r":\d+$", "", story_id)


def _build_fingerprint(raw_manifest_path: Path, config: LsegCollectionConfig) -> str:
    return sha256_text(
        canonical_json(
            {
                "raw_manifest_sha256": sha256_file(raw_manifest_path),
                "cleaner_version": CLEANER_VERSION,
                "min_text_chars": config.min_text_chars,
                "max_scoring_chars": config.max_scoring_chars,
            }
        )
    )


def _write_screening_csv(path: Path, articles: list[dict[str, Any]]) -> Path:
    fields = [
        "article_id",
        "revision_id",
        "story_id",
        "headline",
        "version_created",
        "matched_symbols",
        "source_code",
        "language",
        "text_quality",
        "cleaned_chars",
        "scoring_eligible",
        "status_detail",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for article in articles:
        row = {key: article.get(key, "") for key in fields}
        row["matched_symbols"] = "|".join(article.get("matched_symbols") or [])
        writer.writerow(row)
    return atomic_write_text(path, output.getvalue())


def build_lseg_corpus(raw_source: str | Path) -> LsegCorpusResult:
    raw_dir = Path(raw_source)
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise LsegNewsError(f"raw LSEG collection is missing manifest.json: {raw_dir}")
    raw_manifest = read_json(manifest_path)
    if raw_manifest.get("status") != "completed":
        raise LsegNewsError(f"raw LSEG collection is not completed: {raw_dir}")
    raw_config = raw_manifest.get("config")
    if not isinstance(raw_config, dict):
        raise LsegNewsError("raw LSEG manifest is missing its collection configuration")
    config = config_from_payload(raw_config)
    if raw_manifest.get("config_sha256") != config.config_sha256:
        raise LsegNewsError("raw LSEG manifest configuration hash does not match the embedded configuration")

    headlines_path = _manifest_file(raw_dir, raw_manifest, "headlines_jsonl")
    stories_path = _manifest_file(raw_dir, raw_manifest, "stories_jsonl")
    build_fingerprint = _build_fingerprint(manifest_path, config)
    derived_dir = config.derived_dir
    derived_manifest_path = derived_dir / "manifest.json"
    if derived_manifest_path.exists():
        existing = read_json(derived_manifest_path)
        if existing.get("build_fingerprint") != build_fingerprint:
            raise LsegNewsError(
                f"derived corpus {config.collection_id} already exists for different raw data or cleaning settings; "
                "choose a new collection.id"
            )
        if existing.get("status") == "completed":
            counts_value = existing.get("counts")
            counts: dict[str, Any] = counts_value if isinstance(counts_value, dict) else {}
            return LsegCorpusResult(
                derived_dir=derived_dir,
                articles_path=derived_dir / "articles.jsonl",
                screening_path=derived_dir / "screening_index.csv",
                manifest_path=derived_manifest_path,
                article_count=int(counts.get("articles", 0)),
                eligible_count=int(counts.get("eligible", 0)),
                resumed=True,
            )
    elif derived_dir.exists() and any(derived_dir.iterdir()):
        raise LsegNewsError(f"refusing to overwrite non-empty derived corpus without a manifest: {derived_dir}")

    derived_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        derived_manifest_path,
        {
            "schema_version": LSEG_CORPUS_SCHEMA_VERSION,
            "status": "in_progress",
            "build_fingerprint": build_fingerprint,
            "raw_manifest": str(manifest_path),
        },
    )
    headlines = read_jsonl(headlines_path)
    stories = {str(row.get("story_id") or ""): row for row in read_jsonl(stories_path) if row.get("story_id")}
    articles: list[dict[str, Any]] = []
    for headline in headlines:
        story_id = str(headline.get("story_id") or "").strip()
        if not story_id:
            continue
        family_id = _story_family_id(story_id)
        version_created = _parse_version_created(headline.get("version_created"))
        story = stories.get(story_id) or {}
        body = story.get("body") if isinstance(story.get("body"), str) else None
        story_status = str(story.get("status") or "missing")
        language = headline.get("language") or config.language
        detail: str | None = None
        if not version_created:
            quality = QUALITY_INVALID_TIMESTAMP
            clean_text = ""
            cleaned_chars = 0
            detail = "missing or invalid versionCreated; availability time is unknown"
            cleaning = None
        elif not _is_english(language, config.language):
            quality = QUALITY_NON_ENGLISH
            clean_text = ""
            cleaned_chars = 0
            detail = f"unsupported story language: {language}"
            cleaning = None
        elif story_status != "success":
            quality = story_status
            clean_text = ""
            cleaned_chars = 0
            detail = str(story.get("error") or f"story status is {story_status}")
            cleaning = None
        else:
            cleaned = clean_news_html(body, min_chars=config.min_text_chars)
            quality = cleaned.quality
            clean_text = cleaned.text
            cleaned_chars = cleaned.cleaned_chars
            detail = cleaned.detail
            cleaning = {
                "cleaner_version": cleaned.cleaner_version,
                "original_chars": cleaned.original_chars,
                "cleaned_chars": cleaned.cleaned_chars,
                "removed_trailing_lines": cleaned.removed_trailing_lines,
            }
        article_id = sha256_text(f"lseg|{family_id}")[:16]
        revision_id = sha256_text(f"lseg|{story_id}|{version_created or ''}")[:16]
        articles.append(
            {
                "article_id": article_id,
                "revision_id": revision_id,
                "story_family_id": family_id,
                "story_id": story_id,
                "provider": "lseg",
                "headline": str(headline.get("headline") or ""),
                "first_created": headline.get("first_created"),
                "version_created": version_created,
                "source_code": headline.get("source_code"),
                "language": language,
                "subjects": headline.get("subjects") or [],
                "entities": headline.get("entities") or [],
                "matched_symbols": headline.get("matched_symbols") or [],
                "matched_queries": headline.get("matched_queries") or [],
                "matched_rics": headline.get("matched_rics") or [],
                "body_format": story.get("body_format"),
                "raw_body_sha256": sha256_text(body) if body else None,
                "clean_text": clean_text,
                "clean_text_sha256": sha256_text(clean_text) if clean_text else None,
                "cleaned_chars": cleaned_chars,
                "max_scoring_chars": config.max_scoring_chars,
                "text_quality": quality,
                "status_detail": detail,
                "scoring_eligible": quality == QUALITY_OK,
                "cleaning": cleaning,
                "raw_story_path": f"stories/{sha256_text(story_id)}.json",
            }
        )
    articles.sort(key=lambda row: (str(row.get("version_created") or ""), row["story_id"]))
    articles_path = atomic_write_jsonl(derived_dir / "articles.jsonl", articles)
    screening_path = _write_screening_csv(derived_dir / "screening_index.csv", articles)
    quality_counts = Counter(str(article["text_quality"]) for article in articles)
    eligible_count = sum(bool(article["scoring_eligible"]) for article in articles)
    derived_manifest = {
        "schema_version": LSEG_CORPUS_SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "build_fingerprint": build_fingerprint,
        "raw_manifest": str(manifest_path),
        "raw_manifest_sha256": sha256_file(manifest_path),
        "config": config.to_payload(),
        "cleaner_version": CLEANER_VERSION,
        "package_versions": {
            "lxml": importlib.metadata.version("lxml"),
        },
        "environment": collect_run_environment(),
        "counts": {
            "articles": len(articles),
            "eligible": eligible_count,
            "ineligible": len(articles) - eligible_count,
            "text_quality": dict(sorted(quality_counts.items())),
        },
        "files": {
            "articles_jsonl": {"path": articles_path.name, "sha256": sha256_file(articles_path)},
            "screening_index_csv": {"path": screening_path.name, "sha256": sha256_file(screening_path)},
        },
        "sharing": {
            "licensed_full_text": True,
            "redistribute": False,
            "note": "The JSONL contains licensed LSEG story text and is local-only.",
        },
    }
    atomic_write_json(derived_manifest_path, derived_manifest)
    return LsegCorpusResult(
        derived_dir=derived_dir,
        articles_path=articles_path,
        screening_path=screening_path,
        manifest_path=derived_manifest_path,
        article_count=len(articles),
        eligible_count=eligible_count,
    )


def load_verified_lseg_corpus(manifest_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(manifest_path)
    manifest = read_json(path)
    if manifest.get("status") != "completed" or manifest.get("schema_version") != LSEG_CORPUS_SCHEMA_VERSION:
        raise LsegNewsError(f"LSEG corpus manifest is not a completed schema-v{LSEG_CORPUS_SCHEMA_VERSION} corpus: {path}")
    files_value = manifest.get("files")
    files: dict[str, Any] = files_value if isinstance(files_value, dict) else {}
    articles_entry = files.get("articles_jsonl")
    if not isinstance(articles_entry, dict):
        raise LsegNewsError(f"LSEG corpus manifest is missing files.articles_jsonl: {path}")
    articles_path = path.parent / str(articles_entry.get("path") or "")
    expected = str(articles_entry.get("sha256") or "")
    if not articles_path.exists() or not expected:
        raise LsegNewsError(f"LSEG corpus articles file is missing or unhashed: {articles_path}")
    actual = sha256_file(articles_path)
    if actual != expected:
        raise LsegNewsError(f"LSEG corpus hash mismatch for {articles_path}: expected {expected}, got {actual}")
    return manifest, read_jsonl(articles_path)
