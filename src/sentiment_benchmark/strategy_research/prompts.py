from __future__ import annotations

from ..models import PromptConfig
from ..prompts import make_prompt
from .schemas import ScoringScheme


def strategy_prompt(scheme: ScoringScheme) -> PromptConfig:
    shared = (
        "Assess only the directional financial implication of the supplied news for the explicitly named target company. "
        "Use only the supplied text. Consider fundamentals, operations, cash flow, financing, litigation, regulation, "
        "material contracts, and other economically relevant developments. Do not predict a stock return, use a price "
        "response, or classify generic emotional tone. Treat incidental, immaterial, mixed, or insufficient target-specific "
        "evidence as neutral."
    )
    if scheme == "three_class":
        return make_prompt(
            "strategy_target_direction_3class_v1",
            f"{shared} Return exactly one enum value: negative, neutral, or positive.",
            "{sentence}\n\nTarget-specific financial direction enum:",
            "label_only",
        )
    if scheme == "five_level":
        return make_prompt(
            "strategy_target_direction_5level_v1",
            (
                f"{shared} Reserve very_positive and very_negative for clearly material implications. "
                "Return exactly one enum value: very_negative, negative, neutral, positive, or very_positive."
            ),
            "{sentence}\n\nTarget-specific financial direction enum:",
            "label_only",
        )
    raise ValueError(f"unknown strategy scoring scheme: {scheme!r}")
