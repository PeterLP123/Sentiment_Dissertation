"""Small cross-cutting helpers shared across modules.

Kept dependency-free (stdlib only, no intra-package imports) so any module can
import from here without risking a circular import.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def slugify(value: str, limit: int = 48) -> str:
    """Lowercase, hyphenated slug suitable for filenames; falls back to ``news``."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit].strip("-") or "news"


def preview(value: str | None, limit: int = 500) -> str:
    """Whitespace-collapsed preview of ``value``, truncated with an ellipsis."""
    collapsed = re.sub(r"\s+", " ", value or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def normalize_domains(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Strip/lowercase domains, drop blanks, and de-duplicate preserving order."""
    cleaned = []
    for value in values or []:
        item = value.strip().lower()
        if item:
            cleaned.append(item)
    return tuple(dict.fromkeys(cleaned))


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serialisable data.

    Note: ``lseg_source`` deliberately keeps its own stricter variant that also
    normalises NaN/inf floats, datetimes, and numpy scalars.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return to_jsonable(as_dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return repr(value)
