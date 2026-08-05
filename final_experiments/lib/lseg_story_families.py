"""Licence-safe story-family filtering for the expanded LSEG corpus.

LSEG story identifiers end in a numeric revision suffix.  The normalized
headline score population intentionally deduplicates exact text, but it does
not collapse distinct headline revisions from the same story family.  This
module reconstructs first-release hashes and first-to-current revision
transitions without returning licensed text or raw vendor story identifiers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from final_experiments.lib.novelty import headline_norm_sha256
from sentiment_benchmark.artifact_io import sha256_file

_REVISION_SUFFIX = re.compile(r":(\d+)$")


class StoryFamilyError(RuntimeError):
    """Raised when the LSEG story-family mapping cannot be trusted."""


@dataclass(frozen=True, order=True)
class _FamilyCandidate:
    timestamp: datetime
    revision_number: int
    story_id: str
    headline_sha256: str


@dataclass(frozen=True, order=True)
class _RevisionCandidate:
    timestamp: datetime
    revision_number: int
    story_id: str
    headline_sha256: str
    matched_symbols: tuple[str, ...] = field(compare=False)


def story_family_id(story_id: str) -> str:
    """Remove the terminal numeric revision suffix from an LSEG story ID."""

    value = str(story_id or "").strip()
    match = _REVISION_SUFFIX.search(value)
    if not match:
        raise StoryFamilyError("LSEG story ID lacks a terminal numeric revision")
    family = value[: match.start()]
    if not family:
        raise StoryFamilyError("LSEG story ID has an empty family prefix")
    return family


def _revision_number(story_id: str) -> int:
    match = _REVISION_SUFFIX.search(str(story_id or "").strip())
    if not match:
        raise StoryFamilyError("LSEG story ID lacks a terminal numeric revision")
    return int(match.group(1))


def _validate_corpus(path: Path) -> dict[str, Any]:
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise StoryFamilyError(f"merged LSEG corpus or manifest is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise StoryFamilyError("merged LSEG corpus manifest is not completed")
    contract = (manifest.get("files") or {}).get("headlines_jsonl") or {}
    expected_hash = contract.get("sha256")
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise StoryFamilyError("merged LSEG headline corpus hash mismatch")
    return {"manifest_path": str(manifest_path), "corpus_sha256": actual_hash}


def _availability_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StoryFamilyError("invalid LSEG availability timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _matched_symbols(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        symbols = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        symbols = value
    else:
        symbols = ()
    return tuple(sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()}))


def load_first_release_hashes(
    corpus_path: str | Path,
    *,
    expected_hashes: set[str],
) -> tuple[set[str], dict[str, Any]]:
    """Return hashes retained by a first-release-per-story-family screen.

    A normalized headline is retained when it is the earliest timestamped
    revision of at least one family.  This preserves a headline reused as an
    independent first release elsewhere while removing hashes that occur only
    as later revisions.  Ties are resolved by revision number and story ID.
    Raw headline text is used only to reconstruct the existing normalized hash
    and is never returned.
    """

    path = Path(corpus_path)
    source_audit = _validate_corpus(path)
    expected = {str(value) for value in expected_hashes}
    if not expected:
        raise StoryFamilyError("expected score-hash population is empty")

    family_first: dict[str, _FamilyCandidate] = {}
    family_counts: dict[str, int] = {}
    observed_expected_hashes: set[str] = set()
    rows_read = 0
    rows_in_score_population = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StoryFamilyError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from exc
            rows_read += 1
            digest = headline_norm_sha256(str(row.get("headline") or ""))
            if digest not in expected:
                continue
            rows_in_score_population += 1
            observed_expected_hashes.add(digest)
            story_id = str(row.get("story_id") or "").strip()
            family = story_family_id(story_id)
            timestamp_value = row.get("version_created") or row.get("first_created")
            try:
                timestamp = _availability_timestamp(timestamp_value)
            except StoryFamilyError as exc:
                raise StoryFamilyError(
                    f"score-population row has an invalid availability timestamp on line {line_number}"
                ) from exc
            candidate = _FamilyCandidate(
                timestamp=timestamp,
                revision_number=_revision_number(story_id),
                story_id=story_id,
                headline_sha256=digest,
            )
            family_counts[family] = family_counts.get(family, 0) + 1
            current = family_first.get(family)
            if current is None or candidate < current:
                family_first[family] = candidate

    missing = expected - observed_expected_hashes
    if missing:
        raise StoryFamilyError(
            f"merged corpus misses {len(missing)} expected successful headline hashes"
        )
    retained = {candidate.headline_sha256 for candidate in family_first.values()}
    removed = expected - retained
    audit: dict[str, Any] = {
        **source_audit,
        "rows_read": rows_read,
        "rows_in_score_population": rows_in_score_population,
        "expected_successful_hashes": len(expected),
        "observed_successful_hashes": len(observed_expected_hashes),
        "story_families_in_score_population": len(family_counts),
        "families_with_multiple_rows": sum(
            count > 1 for count in family_counts.values()
        ),
        "rows_in_multirow_families": sum(
            count for count in family_counts.values() if count > 1
        ),
        "maximum_rows_per_family": max(family_counts.values()),
        "retained_first_release_hashes": len(retained),
        "removed_later_revision_only_hashes": len(removed),
        "removed_share": len(removed) / len(expected),
        "licensed_headline_text_emitted": False,
        "selection_order": "minimum version_created, then numeric revision, then story_id",
    }
    return retained, audit


def load_family_revision_transitions(
    corpus_path: str | Path,
    *,
    expected_hashes: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return licence-safe first-to-current revision transitions.

    Each row represents a later revision of a multirow LSEG story family.  It
    contains only normalized headline hashes, timestamps, revision numbers,
    and company symbols shared by the first and current revisions.  Raw
    headline text and vendor story identifiers are never returned.  Family
    identifiers are one-way SHA-256 digests so callers can deduplicate causal
    revisions without persisting licensed identifiers.
    """

    path = Path(corpus_path)
    source_audit = _validate_corpus(path)
    expected = {str(value) for value in expected_hashes}
    if not expected:
        raise StoryFamilyError("expected score-hash population is empty")

    singletons: dict[str, _RevisionCandidate] = {}
    multirow: dict[str, list[_RevisionCandidate]] = {}
    observed_expected_hashes: set[str] = set()
    rows_read = 0
    rows_in_score_population = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StoryFamilyError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from exc
            rows_read += 1
            digest = headline_norm_sha256(str(row.get("headline") or ""))
            if digest not in expected:
                continue
            rows_in_score_population += 1
            observed_expected_hashes.add(digest)
            story_id = str(row.get("story_id") or "").strip()
            family = story_family_id(story_id)
            timestamp_value = row.get("version_created") or row.get("first_created")
            try:
                timestamp = _availability_timestamp(timestamp_value)
            except StoryFamilyError as exc:
                raise StoryFamilyError(
                    f"score-population row has an invalid availability timestamp on line {line_number}"
                ) from exc
            candidate = _RevisionCandidate(
                timestamp=timestamp,
                revision_number=_revision_number(story_id),
                story_id=story_id,
                headline_sha256=digest,
                matched_symbols=_matched_symbols(row.get("matched_symbols")),
            )
            if family in multirow:
                multirow[family].append(candidate)
            elif family in singletons:
                multirow[family] = [singletons.pop(family), candidate]
            else:
                singletons[family] = candidate

    missing = expected - observed_expected_hashes
    if missing:
        raise StoryFamilyError(
            f"merged corpus misses {len(missing)} expected successful headline hashes"
        )

    records: list[dict[str, Any]] = []
    multi_distinct = 0
    maximum_distinct_revisions = 1
    for family, candidates in multirow.items():
        deduplicated: dict[
            tuple[datetime, int, str, str], set[str]
        ] = {}
        for candidate in candidates:
            key = (
                candidate.timestamp,
                candidate.revision_number,
                candidate.story_id,
                candidate.headline_sha256,
            )
            deduplicated.setdefault(key, set()).update(candidate.matched_symbols)
        ordered = [
            _RevisionCandidate(
                timestamp=key[0],
                revision_number=key[1],
                story_id=key[2],
                headline_sha256=key[3],
                matched_symbols=tuple(sorted(symbols)),
            )
            for key, symbols in deduplicated.items()
        ]
        ordered.sort()
        maximum_distinct_revisions = max(maximum_distinct_revisions, len(ordered))
        if len(ordered) < 2:
            continue
        multi_distinct += 1
        initial = ordered[0]
        family_digest = sha256(family.encode("utf-8")).hexdigest()
        for current in ordered[1:]:
            shared = tuple(
                sorted(set(initial.matched_symbols) & set(current.matched_symbols))
            )
            records.append(
                {
                    "story_family_sha256": family_digest,
                    "initial_timestamp": initial.timestamp,
                    "revision_timestamp": current.timestamp,
                    "initial_revision_number": initial.revision_number,
                    "current_revision_number": current.revision_number,
                    "initial_headline_sha256": initial.headline_sha256,
                    "current_headline_sha256": current.headline_sha256,
                    "initial_symbols": "|".join(initial.matched_symbols),
                    "current_symbols": "|".join(current.matched_symbols),
                    "shared_symbols": "|".join(shared),
                    "headline_changed": (
                        initial.headline_sha256 != current.headline_sha256
                    ),
                }
            )

    columns = [
        "story_family_sha256",
        "initial_timestamp",
        "revision_timestamp",
        "initial_revision_number",
        "current_revision_number",
        "initial_headline_sha256",
        "current_headline_sha256",
        "initial_symbols",
        "current_symbols",
        "shared_symbols",
        "headline_changed",
    ]
    transitions = pd.DataFrame.from_records(records, columns=columns)
    if not transitions.empty:
        transitions = transitions.sort_values(
            [
                "revision_timestamp",
                "story_family_sha256",
                "current_revision_number",
                "current_headline_sha256",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

    audit: dict[str, Any] = {
        **source_audit,
        "rows_read": rows_read,
        "rows_in_score_population": rows_in_score_population,
        "expected_successful_hashes": len(expected),
        "observed_successful_hashes": len(observed_expected_hashes),
        "story_families_in_score_population": len(singletons) + len(multirow),
        "families_with_multiple_rows": len(multirow),
        "rows_in_multirow_families": sum(len(values) for values in multirow.values()),
        "families_with_multiple_distinct_revisions": multi_distinct,
        "maximum_distinct_revisions_per_family": maximum_distinct_revisions,
        "transition_rows": len(transitions),
        "headline_change_transition_rows": int(
            transitions["headline_changed"].sum() if not transitions.empty else 0
        ),
        "shared_symbol_transition_rows": int(
            transitions["shared_symbols"].ne("").sum() if not transitions.empty else 0
        ),
        "licensed_headline_text_emitted": False,
        "raw_story_ids_emitted": False,
        "selection_order": "minimum version_created, then numeric revision, then story_id",
    }
    return transitions, audit


__all__ = [
    "StoryFamilyError",
    "load_family_revision_transitions",
    "load_first_release_hashes",
    "story_family_id",
]
