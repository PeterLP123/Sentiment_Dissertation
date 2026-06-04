from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .constants import ALLOWED_LABELS, DEFAULT_PILOT_PER_CLASS, DEFAULT_SEED
from .models import BlindExample, DatasetRow, RunMode


@dataclass(frozen=True)
class DatasetStats:
    row_count: int
    label_counts: dict[str, int]
    duplicate_sentence_groups: int
    duplicate_extra_rows: int
    conflicting_duplicate_groups: int
    conflicting_duplicate_rows: int
    primary_row_count: int
    primary_label_counts: dict[str, int]


def load_dataset(path: str | Path) -> list[DatasetRow]:
    dataset_path = Path(path)
    with dataset_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = {"Sentence", "Sentiment"} - set(columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing required column(s): {missing_text}")

        raw_rows: list[tuple[int, str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            sentence = (row.get("Sentence") or "").strip()
            label = (row.get("Sentiment") or "").strip().lower()
            if not sentence:
                raise ValueError(f"Blank Sentence at CSV line {row_number}")
            if label not in ALLOWED_LABELS:
                allowed = ", ".join(ALLOWED_LABELS)
                raise ValueError(f"Invalid Sentiment at CSV line {row_number}: expected one of {allowed}")
            raw_rows.append((row_number, sentence, label))

    by_sentence: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_number, sentence, label in raw_rows:
        by_sentence[sentence].append((row_number, label))

    duplicate_info: dict[int, tuple[bool, bool, int]] = {}
    for values in by_sentence.values():
        group_size = len(values)
        labels = {label for _, label in values}
        is_duplicate = group_size > 1
        has_conflict = len(labels) > 1
        for row_number, _ in values:
            duplicate_info[row_number] = (is_duplicate, has_conflict, group_size)

    rows: list[DatasetRow] = []
    for row_number, sentence, label in raw_rows:
        is_duplicate, has_conflict, group_size = duplicate_info[row_number]
        rows.append(
            DatasetRow(
                row_number=row_number,
                sentence=sentence,
                hidden_label=label,
                is_duplicate=is_duplicate,
                has_conflicting_duplicate=has_conflict,
                duplicate_group_size=group_size,
            )
        )
    return rows


def compute_stats(rows: list[DatasetRow]) -> DatasetStats:
    label_counts = Counter(row.hidden_label for row in rows)
    duplicate_groups: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        if row.is_duplicate:
            duplicate_groups[row.sentence].append(row)

    conflicting_groups = {
        sentence: group
        for sentence, group in duplicate_groups.items()
        if len({row.hidden_label for row in group}) > 1
    }
    primary_rows = [row for row in rows if not row.has_conflicting_duplicate]
    primary_label_counts = Counter(row.hidden_label for row in primary_rows)
    return DatasetStats(
        row_count=len(rows),
        label_counts=dict(label_counts),
        duplicate_sentence_groups=len(duplicate_groups),
        duplicate_extra_rows=sum(len(group) - 1 for group in duplicate_groups.values()),
        conflicting_duplicate_groups=len(conflicting_groups),
        conflicting_duplicate_rows=sum(len(group) for group in conflicting_groups.values()),
        primary_row_count=len(primary_rows),
        primary_label_counts=dict(primary_label_counts),
    )


def select_rows(
    rows: list[DatasetRow],
    mode: RunMode,
    sample_per_class: int = DEFAULT_PILOT_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> list[DatasetRow]:
    if mode == "full":
        return list(rows)

    primary_rows = [row for row in rows if not row.has_conflicting_duplicate]
    by_label: dict[str, list[DatasetRow]] = {label: [] for label in ALLOWED_LABELS}
    for row in primary_rows:
        by_label[row.hidden_label].append(row)

    rng = random.Random(seed)
    selected: list[DatasetRow] = []
    for label in ALLOWED_LABELS:
        candidates = by_label[label]
        if len(candidates) < sample_per_class:
            raise ValueError(
                f"Cannot sample {sample_per_class} rows for {label}; only {len(candidates)} primary rows available"
            )
        selected.extend(rng.sample(candidates, sample_per_class))

    return sorted(selected, key=lambda row: row.row_number)


def blind_examples(rows: list[DatasetRow]) -> list[BlindExample]:
    return [row.blind() for row in rows]

