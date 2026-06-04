"""Resolve per-model completion-token budgets.

Reasoning-capable models consume completion tokens on hidden reasoning before
producing the label, so the small default budget that suits standard chat models
can leave nothing for the answer (the API then returns an empty
``finish_reason=length`` response). This module bumps the budget for known
reasoning families and honours explicit per-model overrides.
"""

from __future__ import annotations

from .constants import DEFAULT_REASONING_MAX_COMPLETION_TOKENS, REASONING_MODEL_MARKERS


def is_reasoning_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(marker in lowered for marker in REASONING_MODEL_MARKERS)


def resolve_max_completion_tokens(
    model_id: str,
    default: int,
    overrides: dict[str, int] | None = None,
    reasoning_default: int = DEFAULT_REASONING_MAX_COMPLETION_TOKENS,
) -> int:
    """Return the completion-token budget to use for ``model_id``.

    Precedence: an explicit override wins; otherwise reasoning models receive at
    least ``reasoning_default``; otherwise the global ``default`` is used.
    """
    if overrides and model_id in overrides:
        return overrides[model_id]
    if is_reasoning_model(model_id):
        return max(default, reasoning_default)
    return default
