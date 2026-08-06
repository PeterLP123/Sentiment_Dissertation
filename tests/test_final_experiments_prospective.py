from __future__ import annotations

import json
from pathlib import Path

import pytest

from final_experiments.lib.prospective import (
    ProspectivePopulationError,
    audit_prospective_registry,
)
from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.lseg_source import load_lseg_collection_config


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_batch(
    root: Path,
    *,
    order: int,
    start: str,
    end: str,
    rows: list[dict[str, object]],
    parent_hash: str,
) -> tuple[Path, str]:
    collection_id = f"prospective_batch_{order}"
    config_path = root / "configs" / f"batch_{order}.toml"
    raw_root = root / "Data" / f"batch_{order}" / "raw"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "[collection]",
                f'id = "{collection_id}"',
                f'start = "{start}"',
                f'end = "{end}"',
                'language = "en"',
                "fetch_story_bodies = false",
                "",
                "[outputs]",
                f'raw_output_root = "{raw_root.relative_to(root)}"',
                f'derived_output_root = "Data/batch_{order}/derived"',
                "",
                "[[companies]]",
                'symbol = "AAA"',
                'name = "Alpha"',
                'ric = "AAA.N"',
                'news_query = "R:AAA.N and Language:LEN"',
                'aliases = ["Alpha"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_lseg_collection_config(config_path)
    raw_dir = root / config.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    headlines_path = raw_dir / "headlines.jsonl"
    headlines_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_json(
        raw_dir / "manifest.json",
        {
            "status": "completed",
            "config_sha256": config.config_sha256,
            "counts": {
                "fetch_story_bodies": False,
                "stories": 0,
                "failed_stories": 0,
                "headlines": len(rows),
            },
            "files": {"headlines_jsonl": {"sha256": sha256_file(headlines_path)}},
            "pagination_anomalies": [],
        },
    )
    acquisition_path = root / "specs" / f"batch_{order}.json"
    _write_json(
        acquisition_path,
        {
            "status": "frozen_before_retrieval",
            "parent_strategy_spec": {"sha256": parent_hash},
            "collection_config": {
                "path": str(config_path.relative_to(root)),
                "file_sha256": sha256_file(config_path),
                "parsed_config_sha256": config.config_sha256,
                "collection_id": collection_id,
            },
            "closed_interval": {
                "start_inclusive_utc": start,
                "end_exclusive_utc": end,
            },
            "data_separation": {
                "pool_with_fnspid": False,
                "pool_with_opened_lseg_window": False,
            },
            "scoring_authorisation": {"paid_external_gemma_scoring_permitted_by_this_spec": False},
        },
    )
    return acquisition_path, sha256_file(acquisition_path)


def _make_registry(tmp_path: Path, *, second_start: str = "2024-01-05T00:00:00Z") -> Path:
    parent_path = tmp_path / "specs" / "parent.json"
    _write_json(
        parent_path,
        {
            "status": "frozen_awaiting_genuinely_new_lseg_dates",
            "claim_boundary": {"prospective_start_utc": "2024-01-02T00:00:00Z"},
            "universe": {"sectors": {"test": ["AAA"]}, "companies": 1},
            "minimum_evaluation_population": {
                "complete_price_calendar_sessions": 3,
                "primary_active_sessions": 2,
                "active_sessions_each_chronological_half": 1,
            },
        },
    )
    parent_hash = sha256_file(parent_path)
    first_path, first_hash = _write_batch(
        tmp_path,
        order=1,
        start="2024-01-02T00:00:00Z",
        end="2024-01-05T00:00:00Z",
        rows=[
            {
                "story_id": "story-1",
                "headline": "Alpha rises",
                "first_created": "2024-01-02T12:00:00Z",
                "matched_symbols": ["AAA"],
            },
            {
                "story_id": "story-empty",
                "headline": "---",
                "first_created": "2024-01-03T12:00:00Z",
                "matched_symbols": ["AAA"],
            },
            {
                "story_id": "story-boundary",
                "headline": "Boundary headline",
                "first_created": "2024-01-02T00:00:00Z",
                "matched_symbols": ["AAA"],
            },
        ],
        parent_hash=parent_hash,
    )
    second_path, second_hash = _write_batch(
        tmp_path,
        order=2,
        start=second_start,
        end="2024-01-09T00:00:00Z",
        rows=[
            {
                "story_id": "story-2",
                "headline": "ALPHA rises!",
                "first_created": "2024-01-05T12:00:00Z",
                "matched_symbols": ["AAA"],
            },
            {
                "story_id": "story-3",
                "headline": "Beta falls",
                "first_created": "2024-01-08T12:00:00Z",
                "matched_symbols": ["AAA"],
            },
        ],
        parent_hash=parent_hash,
    )
    registry_path = tmp_path / "specs" / "registry.json"
    _write_json(
        registry_path,
        {
            "status": "append_only",
            "parent_strategy_spec": {
                "path": str(parent_path.relative_to(tmp_path)),
                "sha256": parent_hash,
            },
            "batches": [
                {
                    "order": 1,
                    "acquisition_spec_path": str(first_path.relative_to(tmp_path)),
                    "acquisition_spec_sha256": first_hash,
                },
                {
                    "order": 2,
                    "acquisition_spec_path": str(second_path.relative_to(tmp_path)),
                    "acquisition_spec_sha256": second_hash,
                },
            ],
        },
    )
    return registry_path


def test_registry_deduplicates_normalized_headlines_across_contiguous_batches(
    tmp_path: Path,
) -> None:
    registry_path = _make_registry(tmp_path)
    audit = audit_prospective_registry(tmp_path, registry_path)

    assert audit.batch_audit["headline_rows"].tolist() == [3, 2]
    assert audit.batch_audit["empty_normalized_headline_rows"].tolist() == [1, 0]
    assert audit.batch_audit["cross_batch_normalized_headline_overlap"].tolist() == [0, 1]
    assert audit.cumulative_audit.iloc[-1]["unique_story_ids_cumulative"] == 5
    assert audit.cumulative_audit.iloc[-1]["unique_normalized_headlines_cumulative"] == 3
    assert audit.cumulative_audit.iloc[-1]["eligible_unique_story_ids_cumulative"] == 4
    assert audit.cumulative_audit.iloc[-1]["eligible_unique_normalized_headlines_cumulative"] == 2
    assert audit.cumulative_audit.iloc[-1]["calendar_session_upper_bound"] == 4
    assert audit.calendar_gate_pass
    assert audit.gate_table.iloc[0]["status"] == "PASS_TO_SCORING_STAGE"
    assert not audit.batch_audit["licensed_headline_text_exported"].any()


def test_registry_rejects_a_gap_between_batches(tmp_path: Path) -> None:
    registry_path = _make_registry(tmp_path, second_start="2024-01-06T00:00:00Z")

    with pytest.raises(ProspectivePopulationError, match="contiguous"):
        audit_prospective_registry(tmp_path, registry_path)
