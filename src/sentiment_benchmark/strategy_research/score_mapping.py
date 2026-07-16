from __future__ import annotations

from collections.abc import Mapping

from ..artifact_io import canonical_json, sha256_text
from .schemas import ScoringScheme

THREE_CLASS_LABELS = ("negative", "neutral", "positive")
FIVE_LEVEL_LABELS = ("very_negative", "negative", "neutral", "positive", "very_positive")

SCORE_MAPPINGS: Mapping[ScoringScheme, Mapping[str, float]] = {
    "three_class": {"negative": -1.0, "neutral": 0.0, "positive": 1.0},
    "five_level": {
        "very_negative": -1.0,
        "negative": -0.5,
        "neutral": 0.0,
        "positive": 0.5,
        "very_positive": 1.0,
    },
}


def labels_for_scheme(scheme: ScoringScheme) -> tuple[str, ...]:
    if scheme == "three_class":
        return THREE_CLASS_LABELS
    if scheme == "five_level":
        return FIVE_LEVEL_LABELS
    raise ValueError(f"unknown strategy scoring scheme: {scheme!r}")


def map_score_label(label: str, scheme: ScoringScheme) -> float:
    normalized = label.strip().lower()
    try:
        return SCORE_MAPPINGS[scheme][normalized]
    except KeyError as exc:
        raise ValueError(f"label {label!r} is invalid for {scheme}") from exc


def score_mapping_hash(scheme: ScoringScheme) -> str:
    labels_for_scheme(scheme)
    return sha256_text(canonical_json(dict(SCORE_MAPPINGS[scheme])))
