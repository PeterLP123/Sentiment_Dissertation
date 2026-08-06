from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from final_experiments.lib.backward_validation import (
    audit_backward_collection,
    evaluate_gemma_drift,
    select_gemma_drift_sample,
)
from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.lseg_source import load_lseg_collection_config


def _write_reference(path: Path) -> None:
    rows = [
        {"headline_sha256": "a", "headline": "A", "score": "1", "status": "success"},
        {"headline_sha256": "b", "headline": "B", "score": "-1", "status": "success"},
        {"headline_sha256": "c", "headline": "C", "score": "0.2", "status": "success"},
        {"headline_sha256": "d", "headline": "D", "score": "-0.3", "status": "success"},
        {"headline_sha256": "e", "headline": "E", "score": "0", "status": "success"},
        {"headline_sha256": "z", "headline": "failed", "score": "", "status": "failed"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_backward_collection_audit_reports_only_checkpoint_progress(tmp_path: Path) -> None:
    companies = "\n".join(
        f'''[[companies]]
symbol = "S{index:02d}"
name = "Company {index:02d}"
ric = "S{index:02d}.N"
news_query = "R:S{index:02d}.N and Language:LEN"
aliases = ["S{index:02d}"]'''
        for index in range(33)
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'''[collection]
id = "backward_test"
start = "2024-01-01T00:00:00Z"
end = "2024-01-02T00:00:00Z"
window_days = 1
max_requests_per_run = 9500
fetch_story_bodies = false

[outputs]
raw_output_root = "raw"
derived_output_root = "derived"

{companies}
''',
        encoding="utf-8",
    )
    config = load_lseg_collection_config(config_path)
    parent_path = tmp_path / "parent.json"
    parent_path.write_text("{}\n", encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "status": "frozen_awaiting_lseg_collection",
                "parent_strategy_spec": {"path": "parent.json", "sha256": sha256_file(parent_path)},
                "collection": {
                    "config_path": "config.toml",
                    "config_file_sha256": sha256_file(config_path),
                    "parsed_config_sha256": config.config_sha256,
                    "collection_id": config.collection_id,
                },
            }
        ),
        encoding="utf-8",
    )
    raw_dir = tmp_path / config.raw_dir
    pages = raw_dir / "headline_pages"
    pages.mkdir(parents=True)
    (pages / "001-S00-window-0001-page-0001.json").write_text(
        json.dumps({"symbol": "S00", "window_index": 1, "cursor_out": None}),
        encoding="utf-8",
    )
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "in_progress",
                "config_sha256": config.config_sha256,
                "config": config.to_payload(),
                "started_at": "2026-08-06T13:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    audit = audit_backward_collection(tmp_path, spec_path)

    row = audit.summary.iloc[0]
    assert row["completed_windows"] == 1
    assert row["expected_windows"] == 33
    assert not audit.collection_gate_pass


def test_drift_sample_keeps_all_endpoints_and_deterministic_controls(tmp_path: Path) -> None:
    source = tmp_path / "scores.csv"
    _write_reference(source)

    first = select_gemma_drift_sample(
        source,
        expected_successes=5,
        expected_endpoints=2,
        nonendpoint_count=2,
    )
    second = select_gemma_drift_sample(
        source,
        expected_successes=5,
        expected_endpoints=2,
        nonendpoint_count=2,
    )

    assert first.equals(second)
    assert set(first.loc[first["stratum"].eq("endpoint"), "headline_sha256"]) == {"a", "b"}
    assert len(first) == 4


def test_gemma_drift_gate_passes_identical_scores() -> None:
    reference = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c", "d"],
            "reference_score": [1.0, -1.0, 0.2, -0.3],
            "stratum": ["endpoint", "endpoint", "nonendpoint_control", "nonendpoint_control"],
        }
    )
    repeated = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c", "d"],
            "score": [1.0, -1.0, 0.2, -0.3],
            "status": ["success"] * 4,
        }
    )

    audit = evaluate_gemma_drift(reference, repeated)

    assert audit.drift_gate_pass
    assert audit.gate_table["passed"].all()


def test_gemma_drift_gate_fails_endpoint_sign_reversal() -> None:
    reference = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c", "d"],
            "reference_score": [1.0, -1.0, 0.2, -0.3],
            "stratum": ["endpoint", "endpoint", "nonendpoint_control", "nonendpoint_control"],
        }
    )
    repeated = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c", "d"],
            "score": [-1.0, -1.0, 0.2, -0.3],
            "status": ["success"] * 4,
        }
    )

    audit = evaluate_gemma_drift(reference, repeated)

    assert not audit.drift_gate_pass
    opposite = audit.gate_table.set_index("gate").loc["endpoint_opposite_direction_count"]
    assert not bool(opposite["passed"])
