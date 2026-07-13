from __future__ import annotations

import json
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
    while len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"', "`"}:
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


_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_soft_label_probabilities(raw_content: str) -> dict[str, float] | None:
    """Extract a strict label-probability distribution from model output.

    Accepts the JSON object anywhere in the response (models often wrap it in
    code fences or prose). The object must contain exactly the three allowed
    labels. Values must be finite numbers in [0, 1], and their sum must equal
    1 within a small floating-point tolerance. Invalid distributions are
    rejected rather than silently repaired.
    """
    for match in _JSON_OBJECT.finditer(raw_content):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict) or {str(key).strip().lower() for key in candidate} != set(ALLOWED_LABELS):
            continue
        cleaned: dict[str, float] = {}
        valid = True
        for key, value in candidate.items():
            label = str(key).strip().lower()
            if label not in ALLOWED_LABELS:
                valid = False
                break
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                valid = False
                break
            number = float(value)
            if number < 0 or number > 1 or number != number or number in {float("inf"), float("-inf")}:
                valid = False
                break
            cleaned[label] = number
        if not valid:
            continue
        total = sum(cleaned.values())
        if abs(total - 1.0) > 1e-6:
            continue
        return {label: cleaned[label] for label in ALLOWED_LABELS}
    return None


def _soft_label_argmax(probabilities: dict[str, float]) -> str:
    # Ties break in ALLOWED_LABELS order so the derived hard label is deterministic.
    return max(ALLOWED_LABELS, key=lambda label: probabilities[label])


def parse_model_response(raw_content: str | None, output_mode: OutputMode) -> ParsedResponse:
    if raw_content is None:
        return ParsedResponse(normalized_label=None, parse_status="invalid")

    if output_mode == "soft_label":
        probabilities = parse_soft_label_probabilities(raw_content)
        if probabilities is None:
            return ParsedResponse(normalized_label=None, parse_status="invalid")
        return ParsedResponse(
            normalized_label=_soft_label_argmax(probabilities),
            parse_status="valid",
            label_probabilities=probabilities,
        )

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
