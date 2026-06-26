"""Model knowledge-cutoff registry for contamination control.

An LLM that scores a news item dated *before* its training knowledge cutoff may
have memorised the event and its market outcome, so its "sentiment" can encode
hindsight and inflate backtest performance. Events dated *after* the cutoff are
contamination-free. This module records each scorer's cutoff so the backtest can
stratify results by ``is_post_cutoff`` (see the project's pre-registration: the
stratification is a *sensitivity* layer, never a change to the frozen primary).

The dates below are BEST-EFFORT methodological inputs, not authoritative — verify
each against the provider's current model card before citing in the dissertation,
or override per run via the ``[cutoff.overrides]`` config table (keyed by model id
or its provider-suffix, value an ISO date). This module is a dependency-free leaf.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import date

# Keyed by OpenRouter-style id; the provider-suffix form (after "/") also matches.
# Sources are the providers' stated "knowledge cutoff" — verify before citing.
MODEL_KNOWLEDGE_CUTOFFS: dict[str, date] = {
    "openai/gpt-4o-mini": date(2023, 10, 1),  # OpenAI model card: Oct 2023
    "openai/gpt-4o": date(2023, 10, 1),  # OpenAI model card: Oct 2023
    "google/gemini-2.5-flash-lite": date(2025, 1, 1),  # Google: Jan 2025 (verify)
    "google/gemini-2.5-flash": date(2025, 1, 1),  # Google: Jan 2025 (verify)
    "meta-llama/llama-3.3-70b-instruct": date(2023, 12, 1),  # Meta: Dec 2023
}


def _candidates(model_id: str) -> Iterator[str]:
    mid = model_id.strip()
    yield mid
    if "/" in mid:
        yield mid.split("/", 1)[1]


def _coerce(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def knowledge_cutoff(model_id: str, overrides: Mapping[str, str] | None = None) -> date | None:
    """Training knowledge cutoff for ``model_id``; ``None`` if unknown.

    Config ``overrides`` (keyed by full id or provider-suffix) take precedence
    over the built-in table.
    """
    overrides = overrides or {}
    for key in _candidates(model_id):
        if key in overrides:
            return _coerce(overrides[key])
    for key in _candidates(model_id):
        if key in MODEL_KNOWLEDGE_CUTOFFS:
            return MODEL_KNOWLEDGE_CUTOFFS[key]
    return None


def roster_cutoff(model_ids: Iterable[str], overrides: Mapping[str, str] | None = None) -> date | None:
    """Most-conservative cutoff for an aggregate scorer (e.g. consensus): the
    latest cutoff among constituents, since contamination by *any* constituent
    can taint the aggregate. ``None`` if no constituent has a known cutoff."""
    cutoffs = [cut for cut in (knowledge_cutoff(mid, overrides) for mid in model_ids) if cut is not None]
    return max(cutoffs) if cutoffs else None


def is_post_cutoff(event_date: date | None, cutoff: date | None) -> bool | None:
    """``True`` if the event is after the cutoff (contamination-free), ``False``
    if at/before it, ``None`` if either input is unknown (not applicable)."""
    if event_date is None or cutoff is None:
        return None
    return event_date > cutoff
