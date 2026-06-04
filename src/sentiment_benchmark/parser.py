from __future__ import annotations

import re

from .constants import ALLOWED_LABELS
from .models import OutputMode, ParsedResponse

_ANSWER_MARKER = re.compile(
    r"^\s*(?:final\s+answer|answer|label|sentiment|classification)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-zA-Z]+")


def _clean_candidate(value: str) -> str:
    candidate = value.strip().lower()
    while len(candidate) >= 2 and (
        (candidate[0] == candidate[-1] and candidate[0] in {"'", '"', "`"})
        or (candidate[0] == "`" and candidate[-1] == "`")
    ):
        candidate = candidate[1:-1].strip()
    if candidate.endswith("."):
        candidate = candidate[:-1].strip()
    return candidate


def parse_label_candidate(value: str) -> str | None:
    candidate = _clean_candidate(value)
    if candidate in ALLOWED_LABELS:
        return candidate
    return None


def _parse_cot_label(lines: list[str]) -> str | None:
    """Extract the conclusion label from a chain-of-thought response.

    The answer in CoT output comes last, so we prefer an explicit answer marker
    ("Answer: positive"), then a trailing line that is itself a label, and finally
    any label word appearing in the closing lines (scanned from the end).
    """
    for line in reversed(lines):
        match = _ANSWER_MARKER.match(line)
        if match:
            label = parse_label_candidate(match.group(1))
            if label:
                return label
    for line in reversed(lines):
        label = parse_label_candidate(line)
        if label:
            return label
    last_line = lines[-1]
    for word in reversed(_WORD.findall(last_line.lower())):
        if word in ALLOWED_LABELS:
            return word
    return None


def parse_model_response(raw_content: str | None, output_mode: OutputMode) -> ParsedResponse:
    if raw_content is None:
        return ParsedResponse(normalized_label=None, parse_status="invalid")

    if output_mode == "label_only":
        label = parse_label_candidate(raw_content)
        return ParsedResponse(normalized_label=label, parse_status="valid" if label else "invalid")

    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    if not lines:
        return ParsedResponse(normalized_label=None, parse_status="invalid")

    if output_mode == "cot":
        label = _parse_cot_label(lines)
        explanation = "\n".join(lines).strip() or None
        return ParsedResponse(normalized_label=label, parse_status="valid" if label else "invalid", explanation=explanation)

    label = parse_label_candidate(lines[0])
    explanation = "\n".join(lines[1:]).strip() or None
    return ParsedResponse(normalized_label=label, parse_status="valid" if label else "invalid", explanation=explanation)

