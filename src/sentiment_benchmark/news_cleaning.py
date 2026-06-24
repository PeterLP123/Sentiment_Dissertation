from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CLEANER_VERSION = "lseg_html_v1"

QUALITY_OK = "ok"
QUALITY_MISSING = "missing"
QUALITY_PARSE_ERROR = "parse_error"
QUALITY_BOILERPLATE_ONLY = "boilerplate_only"
QUALITY_TOO_SHORT = "too_short"

_DROP_ELEMENTS = ("script", "style", "noscript", "svg", "canvas", "form", "nav", "footer")
_BLOCK_ELEMENTS = {
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
_TRAILING_BOILERPLATE = (
    re.compile(r"^\(?c\)?\s*copyright\s+(?:thomson\s+)?reuters\b.*$", re.IGNORECASE),
    re.compile(r"^click\s+for\s+restrictions\b.*$", re.IGNORECASE),
    re.compile(r"^\(\(.*(?:@|thomsonreuters|reuters).*(?:\)\))$", re.IGNORECASE),
)


class NewsCleaningDependencyError(RuntimeError):
    """Raised when the optional LSEG HTML parser is unavailable."""


@dataclass(frozen=True)
class CleanedNewsText:
    text: str
    quality: str
    detail: str | None
    original_chars: int
    cleaned_chars: int
    removed_trailing_lines: int
    cleaner_version: str = CLEANER_VERSION


def _import_lxml_html():
    try:
        from lxml import etree, html
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise NewsCleaningDependencyError(
            "LSEG story cleaning requires lxml. Install with: python -m pip install -e '.[lseg]'"
        ) from exc
    return etree, html


def _add_boundary(element: Any) -> None:
    tail = getattr(element, "tail", None) or ""
    if not tail.startswith("\n"):
        element.tail = f"\n{tail}"


def _normalize_lines(value: str) -> list[str]:
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


def _remove_trailing_boilerplate(lines: list[str]) -> tuple[list[str], int]:
    remaining = list(lines)
    removed = 0
    while remaining:
        while remaining and not remaining[-1]:
            remaining.pop()
        if not remaining or not any(pattern.fullmatch(remaining[-1]) for pattern in _TRAILING_BOILERPLATE):
            break
        remaining.pop()
        removed += 1
    while remaining and not remaining[-1]:
        remaining.pop()
    return remaining, removed


def clean_news_html(value: str | None, *, min_chars: int = 100) -> CleanedNewsText:
    """Convert a story body to deterministic paragraph-preserving plain text."""
    if min_chars < 0:
        raise ValueError("min_chars must be zero or greater")
    if not value or not value.strip():
        return CleanedNewsText("", QUALITY_MISSING, "story body is empty", len(value or ""), 0, 0)

    etree, html = _import_lxml_html()
    try:
        parser = html.HTMLParser(encoding="utf-8", recover=True, remove_comments=True)
        document = html.fromstring(value, parser=parser)
        for element_name in _DROP_ELEMENTS:
            for element in document.xpath(f"//{element_name}"):
                element.drop_tree()
        for comment in document.xpath("//comment()"):
            parent = comment.getparent()
            if parent is not None:
                parent.remove(comment)
        for element in document.iter():
            tag = element.tag.lower() if isinstance(element.tag, str) else ""
            if tag == "br" or tag in _BLOCK_ELEMENTS:
                _add_boundary(element)
        raw_text = document.text_content()
    except (etree.ParserError, UnicodeError, ValueError, TypeError) as exc:
        return CleanedNewsText(
            "",
            QUALITY_PARSE_ERROR,
            f"cannot parse story HTML: {exc}",
            len(value),
            0,
            0,
        )

    lines, removed = _remove_trailing_boilerplate(_normalize_lines(raw_text))
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
