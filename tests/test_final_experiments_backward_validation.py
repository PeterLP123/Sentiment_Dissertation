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
        f"""[[companies]]
symbol = "S{index:02d}"
name = "Company {index:02d}"
ric = "S{index:02d}.N"
news_query = "R:S{index:02d}.N and Language:LEN"
aliases = ["S{index:02d}"]"""
        for index in range(33)
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[collection]
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
""",
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
    (pages / "002-S01-window-0001-page-0001.json").write_text(
        json.dumps({"symbol": "S01", "window_index": 1, "cursor_out": "next-page"}),
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
    assert row["complete_dates"] == 0
    assert row["complete_date_prefix"] == 0
    assert row["frontier_date"] == "2024-01-01T00:00:00Z"
    assert row["frontier_companies_completed"] == 1
    assert audit.date_progress["completed_companies"].tolist() == [1]
    assert audit.date_progress["page_requests"].tolist() == [2]
    completion = audit.company_date_completion.set_index(["symbol", "window_index"])["complete"]
    assert audit.company_date_completion.shape == (33, 5)
    assert int(completion.sum()) == 1
    assert bool(completion.loc[("S00", 1)])
    assert not bool(completion.loc[("S01", 1)])
    projection = audit.projection.iloc[0]
    assert projection["estimated_remaining_requests"] == 64
    assert projection["estimated_remaining_quota_days"] == 1
    assert "company-biased" in projection["projection_basis"]
    assert not audit.collection_gate_pass


def test_backward_collection_audit_reports_date_frontier_requests_and_projection(tmp_path: Path) -> None:
    companies = "\n".join(
        f"""[[companies]]
symbol = "S{index:02d}"
name = "Company {index:02d}"
ric = "S{index:02d}.N"
news_query = "R:S{index:02d}.N and Language:LEN"
aliases = ["S{index:02d}"]"""
        for index in range(33)
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[collection]
id = "backward_test"
start = "2024-01-01T00:00:00Z"
end = "2024-01-04T00:00:00Z"
window_days = 1
max_requests_per_run = 9500
fetch_story_bodies = false

[outputs]
raw_output_root = "raw"
derived_output_root = "derived"

{companies}
""",
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
    completed_by_window = {1: 33, 2: 5, 3: 1}
    for window_index, company_count in completed_by_window.items():
        for company_index in range(company_count):
            symbol = f"S{company_index:02d}"
            (pages / f"{company_index + 1:03d}-{symbol}-window-{window_index:04d}-page-0001.json").write_text(
                json.dumps({"symbol": symbol, "window_index": window_index, "cursor_out": None}),
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

    audit = audit_backward_collection(tmp_path, spec_path, daily_request_budget=10)

    summary = audit.summary.iloc[0]
    assert summary["complete_dates"] == 1
    assert summary["complete_date_prefix"] == 1
    assert summary["partial_dates"] == 2
    assert summary["untouched_dates"] == 0
    assert summary["frontier_date"] == "2024-01-02T00:00:00Z"
    assert summary["frontier_companies_completed"] == 5
    assert audit.date_progress["completed_companies"].tolist() == [33, 5, 1]
    assert audit.date_progress["page_requests"].tolist() == [33, 5, 1]
    completion_matrix = audit.company_date_completion.pivot(
        index="symbol",
        columns="window_index",
        values="complete",
    )
    assert completion_matrix.shape == (33, 3)
    assert completion_matrix.sum(axis=0).astype(int).to_dict() == completed_by_window
    assert int(completion_matrix.to_numpy().sum()) == int(summary["completed_windows"])
    projection = audit.projection.iloc[0]
    assert projection["projection_basis"] == "mean page requests across completed 33-company dates"
    assert projection["observed_requests_per_complete_date"] == 33
    assert projection["minimum_remaining_requests"] == 60
    assert projection["estimated_remaining_requests"] == 60
    assert projection["estimated_remaining_quota_days"] == 6

    for window_index, first_missing_company in ((2, 5), (3, 1)):
        for company_index in range(first_missing_company, 33):
            symbol = f"S{company_index:02d}"
            (pages / f"{company_index + 1:03d}-{symbol}-window-{window_index:04d}-page-0001.json").write_text(
                json.dumps({"symbol": symbol, "window_index": window_index, "cursor_out": None}),
                encoding="utf-8",
            )

    complete_audit = audit_backward_collection(tmp_path, spec_path, daily_request_budget=10)

    complete_summary = complete_audit.summary.iloc[0]
    assert complete_summary["complete_dates"] == 3
    assert complete_summary["complete_date_prefix"] == 3
    assert complete_summary["frontier_date"] is None
    assert complete_summary["frontier_companies_completed"] is None
    assert complete_audit.projection.iloc[0]["estimated_remaining_requests"] == 0
    assert complete_audit.projection.iloc[0]["estimated_remaining_quota_days"] == 0


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
