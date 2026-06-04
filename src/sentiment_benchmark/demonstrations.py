from __future__ import annotations

import random
from collections.abc import Iterable

from .constants import ALLOWED_LABELS, DEFAULT_SEED
from .models import DatasetRow


def demonstration_pool(rows: list[DatasetRow], exclude_row_numbers: Iterable[int]) -> list[DatasetRow]:
    """Rows eligible to serve as few-shot examples.

    Excludes the evaluation rows (to prevent leakage) and any row whose duplicate
    sentence carries a conflicting label, so demonstrations always show an
    unambiguous (sentence, label) mapping.
    """
    excluded = set(exclude_row_numbers)
    return [row for row in rows if row.row_number not in excluded and not row.has_conflicting_duplicate]


def select_demonstrations(
    pool: list[DatasetRow],
    k_per_class: int,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, str]]:
    """Pick ``k_per_class`` class-balanced demonstrations, sampled reproducibly.

    Demonstrations are returned interleaved by class (positive, negative, neutral,
    positive, ...) so neither the count nor the ordering of labels in the prompt
    biases the model toward a particular class.
    """
    if k_per_class <= 0:
        return []

    by_label: dict[str, list[DatasetRow]] = {label: [] for label in ALLOWED_LABELS}
    for row in pool:
        if row.hidden_label in by_label:
            by_label[row.hidden_label].append(row)

    rng = random.Random(seed)
    sampled: dict[str, list[DatasetRow]] = {}
    for label in ALLOWED_LABELS:
        candidates = by_label[label]
        if len(candidates) < k_per_class:
            raise ValueError(
                f"Cannot draw {k_per_class} {label} demonstrations; only {len(candidates)} non-conflicting rows available"
            )
        sampled[label] = rng.sample(candidates, k_per_class)

    demonstrations: list[tuple[str, str]] = []
    for index in range(k_per_class):
        for label in ALLOWED_LABELS:
            row = sampled[label][index]
            demonstrations.append((row.sentence, row.hidden_label))
    return demonstrations
