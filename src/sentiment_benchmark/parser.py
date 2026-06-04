from __future__ import annotations

from .constants import ALLOWED_LABELS
from .models import OutputMode, ParsedResponse


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


def parse_model_response(raw_content: str | None, output_mode: OutputMode) -> ParsedResponse:
    if raw_content is None:
        return ParsedResponse(normalized_label=None, parse_status="invalid")

    if output_mode == "label_only":
        label = parse_label_candidate(raw_content)
        return ParsedResponse(normalized_label=label, parse_status="valid" if label else "invalid")

    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    if not lines:
        return ParsedResponse(normalized_label=None, parse_status="invalid")
    label = parse_label_candidate(lines[0])
    explanation = "\n".join(lines[1:]).strip() or None
    return ParsedResponse(normalized_label=label, parse_status="valid" if label else "invalid", explanation=explanation)

