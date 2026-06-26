"""Self-contained Week 4 LSEG article cleaner.

Dependencies: pandas and lxml. Reads raw CSVs from LSEG.py, converts story HTML
to plain text, and writes derived CSVs without modifying the raw downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cleaned"
RAW_STORY_COLUMNS = ("story_html", "story")
SUMMARY_NAME = "cleaning_summary.csv"
DEFAULT_ARTICLE_TYPES = ("newsroom", "reuters")
CLEANER_VERSION = "week4_lseg_html_v1"

QUALITY_OK = "ok"
QUALITY_MISSING = "missing"
QUALITY_PARSE_ERROR = "parse_error"
QUALITY_BOILERPLATE_ONLY = "boilerplate_only"
QUALITY_TOO_SHORT = "too_short"

DROP_ELEMENTS = ("script", "style", "noscript", "svg", "canvas", "form", "nav", "footer")
BLOCK_ELEMENTS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
}
TRAILING_BOILERPLATE = (
    re.compile(r"^\(?c\)?\s*copyright\s+(?:thomson\s+)?reuters\b.*$", re.IGNORECASE),
    re.compile(r"^all rights reserved\.?$", re.IGNORECASE),
    re.compile(r"^click\s+for\s+restrictions\b.*$", re.IGNORECASE),
    re.compile(r"^(?:created|produced)\s+by\s+(?:www\.)?buysellsignals\.com(?:\b.*)?$", re.IGNORECASE),
    re.compile(r"^source:\s*(?:www\.)?buysellsignals\.com\.?$", re.IGNORECASE),
    re.compile(r"^the views expressed in any and all content distributed by\b.*$", re.IGNORECASE),
    re.compile(r"^\(?for (?:republication|redistribution)\b.*(?:prohibited|restrictions)\b.*$", re.IGNORECASE),
    re.compile(r"^https://filings\.ica\.int\.thomsonreuters\.com/filings\.viewer/\S+.*$", re.IGNORECASE),
    re.compile(r"^\(?\w+ reports deliver fact-based news of research and discoveries\b.*$", re.IGNORECASE),
    re.compile(r"^(?:copyright\s+)?©\s*\d{4}(?:-\d{4})?\s+.*(?:pivotalsources\.com|powered by)\b.*$", re.IGNORECASE),
    re.compile(r"^\(\(.*(?:@|thomsonreuters|reuters).*(?:\)\))$", re.IGNORECASE),
)
INLINE_BOILERPLATE = (
    re.compile(r"^(?:created|produced)\s+by\s+(?:www\.)?buysellsignals\.com(?:\b.*)?$", re.IGNORECASE),
    re.compile(r"^(?:www\.)?buysellsignals\.com/?$", re.IGNORECASE),
    re.compile(r'^"?click here"?\s+https://www\.buysellsignals\.com/bst/(?:glossary|disclaimer)\b.*$', re.IGNORECASE),
    re.compile(r"^link to disclaimer:\s*https://www\.buysellsignals\.com/bst/disclaimer\b.*$", re.IGNORECASE),
    re.compile(r"^disclaimer:\s+while this document is based on information sources\b.*$", re.IGNORECASE),
    re.compile(r"^data for the buysellsignals algorithms is drawn from\b.*$", re.IGNORECASE),
    re.compile(r"^the views expressed in any and all content distributed by\b.*$", re.IGNORECASE),
    re.compile(r"^\(\(.*(?:@|thomsonreuters|reuters).*(?:\)\))$", re.IGNORECASE),
)


class CleaningDependencyError(RuntimeError):
    """Raised when the optional HTML parser is unavailable."""


@dataclass(frozen=True)
class CleanedNewsText:
    text: str
    quality: str
    detail: str | None
    original_chars: int
    cleaned_chars: int
    removed_trailing_lines: int
    cleaner_version: str = CLEANER_VERSION


def import_lxml_html():
    try:
        from lxml import etree, html
    except ImportError as exc:
        raise CleaningDependencyError("Install lxml to clean article HTML: python -m pip install lxml") from exc
    return etree, html


def add_boundary(element: Any) -> None:
    tail = getattr(element, "tail", None) or ""
    if not tail.startswith("\n"):
        element.tail = f"\n{tail}"


def normalize_lines(value: str) -> list[str]:
    normalized: list[str] = []
    blank = False
    for raw_line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
        if not line:
            if normalized and not blank:
                normalized.append("")
            blank = True
            continue
        normalized.append(line)
        blank = False
    while normalized and not normalized[-1]:
        normalized.pop()
    return normalized


def remove_inline_boilerplate(lines: list[str]) -> list[str]:
    return [line for line in lines if line and not any(pattern.fullmatch(line) for pattern in INLINE_BOILERPLATE)]


def remove_trailing_boilerplate(lines: list[str]) -> tuple[list[str], int]:
    remaining = list(lines)
    removed = 0
    while remaining:
        while remaining and not remaining[-1]:
            remaining.pop()
        if not remaining or not any(pattern.fullmatch(remaining[-1]) for pattern in TRAILING_BOILERPLATE):
            break
        remaining.pop()
        removed += 1
    while remaining and not remaining[-1]:
        remaining.pop()
    return remaining, removed


def clean_news_html(value: str | None, *, min_chars: int = 100) -> CleanedNewsText:
    if min_chars < 0:
        raise ValueError("min_chars must be zero or greater")
    if not value or not value.strip():
        return CleanedNewsText("", QUALITY_MISSING, "story body is empty", len(value or ""), 0, 0)

    etree, html = import_lxml_html()
    try:
        parser = html.HTMLParser(encoding="utf-8", recover=True, remove_comments=True)
        document = html.fromstring(value, parser=parser)
        for element_name in DROP_ELEMENTS:
            for element in document.xpath(f"//{element_name}"):
                element.drop_tree()
        for comment in document.xpath("//comment()"):
            parent = comment.getparent()
            if parent is not None:
                parent.remove(comment)
        for element in document.iter():
            tag = element.tag.lower() if isinstance(element.tag, str) else ""
            if tag == "br" or tag in BLOCK_ELEMENTS:
                add_boundary(element)
        raw_text = document.text_content()
    except (etree.ParserError, UnicodeError, ValueError, TypeError) as exc:
        return CleanedNewsText("", QUALITY_PARSE_ERROR, f"cannot parse story HTML: {exc}", len(value), 0, 0)

    lines, removed = remove_trailing_boilerplate(remove_inline_boilerplate(normalize_lines(raw_text)))
    text = "\n\n".join(line for line in lines if line).strip()
    if not text:
        return CleanedNewsText(
            "",
            QUALITY_BOILERPLATE_ONLY,
            "story contains no substantive text after boilerplate removal",
            len(value),
            0,
            removed,
        )
    if min_chars and len(text) < min_chars:
        return CleanedNewsText(
            text,
            QUALITY_TOO_SHORT,
            f"cleaned story has {len(text)} characters (minimum {min_chars})",
            len(value),
            len(text),
            removed,
        )
    return CleanedNewsText(text, QUALITY_OK, None, len(value), len(text), removed)


def story_id_is_present(story_id: object) -> bool:
    return bool(pd.notna(story_id) and str(story_id).strip())


def story_type_from_id(story_id: object) -> str:
    if not story_id_is_present(story_id):
        return "unknown"
    value = str(story_id)
    if value.startswith("urn:link:webnews:"):
        return "webnews"
    if value.startswith("urn:newsml:social:"):
        return "social"
    if value.startswith("urn:newsml:reuters.com:"):
        return "reuters"
    if value.startswith("urn:newsml:newsroom:"):
        return "newsroom"
    if value.startswith("urn:"):
        parts = value.split(":")
        return ":".join(parts[1:3]) if len(parts) >= 3 else parts[1]
    return "unknown"


def find_story_column(dataframe: pd.DataFrame) -> str:
    for column in RAW_STORY_COLUMNS:
        if column in dataframe.columns:
            return column
    raise ValueError(f"Input CSV must contain one of these columns: {', '.join(RAW_STORY_COLUMNS)}")


def text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text.strip() else None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_story_types(value: str) -> tuple[str, ...]:
    story_types = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not story_types:
        raise ValueError("--article-types must contain at least one story type")
    return story_types


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_cleaned.csv"


def clean_dataframe(
    dataframe: pd.DataFrame,
    *,
    min_chars: int = 100,
    keep_raw_html: bool = False,
    ok_only: bool = False,
    articles_only: bool = False,
    article_types: tuple[str, ...] = DEFAULT_ARTICLE_TYPES,
) -> tuple[pd.DataFrame, str]:
    story_column = find_story_column(dataframe)
    cleaned_rows: list[dict[str, object]] = []
    article_type_set = set(article_types)

    for _, row in dataframe.iterrows():
        record = row.to_dict()
        story_type = str(record.get("story_type") or "").strip()
        if not story_type:
            story_type = story_type_from_id(record.get("storyId"))
            record["story_type"] = story_type
        story_type = story_type.lower()

        cleaned = clean_news_html(text_or_none(row.get(story_column)), min_chars=min_chars)
        article_story_type = story_type in article_type_set
        usable_article = article_story_type and cleaned.quality == "ok"
        if not keep_raw_html:
            for column in RAW_STORY_COLUMNS:
                record.pop(column, None)

        record.update(
            {
                "clean_text": cleaned.text,
                "clean_text_sha256": sha256_text(cleaned.text) if cleaned.text else "",
                "cleaning_quality": cleaned.quality,
                "cleaning_detail": cleaned.detail or "",
                "original_chars": cleaned.original_chars,
                "cleaned_chars": cleaned.cleaned_chars,
                "removed_trailing_lines": cleaned.removed_trailing_lines,
                "cleaner_version": cleaned.cleaner_version,
                "article_story_type": article_story_type,
                "usable_article": usable_article,
            }
        )
        cleaned_rows.append(record)

    cleaned_dataframe = pd.DataFrame(cleaned_rows)
    if ok_only:
        cleaned_dataframe = cleaned_dataframe.loc[cleaned_dataframe["cleaning_quality"] == "ok"].copy()
    if articles_only:
        cleaned_dataframe = cleaned_dataframe.loc[cleaned_dataframe["usable_article"]].copy()
    return cleaned_dataframe, story_column


def clean_file(
    input_path: Path,
    output_path: Path,
    *,
    min_chars: int = 100,
    keep_raw_html: bool = False,
    ok_only: bool = False,
    articles_only: bool = False,
    article_types: tuple[str, ...] = DEFAULT_ARTICLE_TYPES,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    dataframe = pd.read_csv(input_path)
    cleaned_dataframe, story_column = clean_dataframe(
        dataframe,
        min_chars=min_chars,
        keep_raw_html=keep_raw_html,
        ok_only=ok_only,
        articles_only=articles_only,
        article_types=article_types,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataframe.to_csv(output_path, index=False)

    quality_counts = Counter(cleaned_dataframe.get("cleaning_quality", pd.Series(dtype=str)))
    return {
        "source_file": str(input_path),
        "output_file": str(output_path),
        "story_column": story_column,
        "rows_in": len(dataframe),
        "rows_out": len(cleaned_dataframe),
        "min_chars": min_chars,
        "ok_only": ok_only,
        "articles_only": articles_only,
        "article_types": ",".join(article_types),
        "keep_raw_html": keep_raw_html,
        "cleaner_version": CLEANER_VERSION,
        "article_story_type_rows": int(cleaned_dataframe.get("article_story_type", pd.Series(dtype=bool)).sum()),
        "usable_article_rows": int(cleaned_dataframe.get("usable_article", pd.Series(dtype=bool)).sum()),
        "quality_ok": quality_counts.get("ok", 0),
        "quality_missing": quality_counts.get("missing", 0),
        "quality_too_short": quality_counts.get("too_short", 0),
        "quality_boilerplate_only": quality_counts.get("boilerplate_only", 0),
        "quality_parse_error": quality_counts.get("parse_error", 0),
    }


def input_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    files = sorted(path for path in input_path.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No files matching {pattern!r} found in {input_path}")
    return files


def write_summary(summary_rows: list[dict[str, object]], output_dir: Path, *, overwrite: bool = False) -> Path:
    summary_path = output_dir / SUMMARY_NAME
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"{summary_path} already exists; pass --overwrite to replace it")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Week 4 LSEG story HTML into plain-text article CSVs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV file or directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for derived cleaned CSVs.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern used when --input is a directory.")
    parser.add_argument("--min-chars", type=int, default=100, help="Minimum cleaned text length for quality=ok.")
    parser.add_argument("--ok-only", action="store_true", help="Write only rows with cleaning_quality=ok.")
    parser.add_argument(
        "--articles-only",
        action="store_true",
        help="Write only rows whose story_type is in --article-types and cleaning_quality=ok.",
    )
    parser.add_argument(
        "--article-types",
        default=",".join(DEFAULT_ARTICLE_TYPES),
        help="Comma-separated story_type values considered articles. Default: newsroom,reuters.",
    )
    parser.add_argument("--keep-raw-html", action="store_true", help="Keep the original story/story_html column in the cleaned CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cleaned outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_chars < 0:
        raise ValueError("--min-chars must be zero or greater")
    article_types = parse_story_types(args.article_types)

    summaries = []
    for source_path in input_files(args.input, args.pattern):
        destination = output_path_for(source_path, args.output_dir)
        summary = clean_file(
            source_path,
            destination,
            min_chars=args.min_chars,
            keep_raw_html=args.keep_raw_html,
            ok_only=args.ok_only,
            articles_only=args.articles_only,
            article_types=article_types,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(
            f"Wrote {summary['rows_out']} of {summary['rows_in']} rows to {destination} "
            f"(ok={summary['quality_ok']}, missing={summary['quality_missing']}, "
            f"too_short={summary['quality_too_short']}, boilerplate_only={summary['quality_boilerplate_only']}, "
            f"usable_articles={summary['usable_article_rows']})"
        )

    summary_path = write_summary(summaries, args.output_dir, overwrite=args.overwrite)
    print(f"Wrote cleaning summary to {summary_path}")


if __name__ == "__main__":
    main()
