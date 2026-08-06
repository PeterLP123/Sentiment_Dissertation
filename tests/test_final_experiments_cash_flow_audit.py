from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from final_experiments.lib.cash_flow_audit import (
    AUDIT_STRATA,
    CODER_COLUMNS,
    build_blinded_audit_sample,
    evaluate_cash_flow_reliability,
    load_cash_flow_audit_state,
    load_full_text_candidates,
    validate_coder_sheet,
    worksheet_identity_sha256,
    write_audit_pack,
)
from final_experiments.lib.novelty import headline_norm_sha256
from sentiment_benchmark.artifact_io import sha256_file


def _candidate_frame(rows_per_stratum: int = 15) -> pd.DataFrame:
    rows = []
    for stratum in AUDIT_STRATA:
        for number in range(rows_per_stratum):
            rows.append(
                {
                    "headline_sha256": f"{stratum}-{number}",
                    "headline": f"Headline {stratum} {number}",
                    "clean_text": f"Body {stratum} {number}",
                    "proxy_stratum": stratum,
                    "version_created": "2026-01-01T00:00:00Z",
                    "article_id": f"a-{stratum}-{number}",
                    "cleaned_chars": 20,
                }
            )
    return pd.DataFrame(rows)


def _complete_sheet(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    completed = frame.copy()
    completed["human_distance"] = labels
    completed["unresolved_prerequisites"] = ["none" if label == "0" else "named prerequisite" for label in labels]
    completed["evidence_span"] = "supporting passage"
    completed["confidence"] = "high"
    return completed


def test_blinded_sample_is_deterministic_and_leak_free() -> None:
    allocation = {stratum: 12 for stratum in AUDIT_STRATA}
    first = build_blinded_audit_sample(_candidate_frame(), allocation=allocation, seed=7)
    second = build_blinded_audit_sample(_candidate_frame(), allocation=allocation, seed=7)
    coder_a, coder_b, key, summary = first

    pd.testing.assert_frame_equal(coder_a, second[0])
    pd.testing.assert_frame_equal(coder_b, second[1])
    assert tuple(coder_a.columns) == CODER_COLUMNS
    assert len(coder_a) == 60
    assert len(coder_b) == 60
    assert set(coder_b["audit_id"]) == set(key.loc[key["double_code"], "audit_id"])
    assert summary["sample_count"].sum() == 60
    assert "proxy_stratum" not in coder_a
    assert "headline_sha256" not in coder_a


def test_sample_fails_closed_when_a_stratum_is_too_small() -> None:
    allocation = {stratum: 12 for stratum in AUDIT_STRATA}
    allocation["d3"] = 16
    with pytest.raises(ValueError, match="stratum d3 has 15 candidates"):
        build_blinded_audit_sample(_candidate_frame(), allocation=allocation)


def test_audit_pack_manifest_records_exact_written_artifacts(tmp_path: Path) -> None:
    allocation = {stratum: 12 for stratum in AUDIT_STRATA}
    coder_a, coder_b, key, summary = build_blinded_audit_sample(_candidate_frame(), allocation=allocation, seed=7)

    manifest = write_audit_pack(
        tmp_path,
        coder_a=coder_a,
        coder_b=coder_b,
        private_key=key,
        summary=summary,
        source_audit={"unique_matching_headline_hashes": 75},
        seed=7,
        allocation=allocation,
    )

    assert manifest["seed"] == 7
    assert manifest["allocation"] == allocation
    assert manifest["sample_rows"] == 60
    assert manifest["double_code_rows"] == 60
    assert manifest["files"]["coder_a_200.csv"] == sha256_file(tmp_path / "coder_a_200.csv")
    assert manifest["worksheet_identity_sha256"]["coder_a_200.csv"] == worksheet_identity_sha256(coder_a)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert tuple(pd.read_csv(tmp_path / "coder_a_200.csv").columns) == CODER_COLUMNS


def test_full_text_loader_validates_hash_and_chooses_longest_body(
    tmp_path: Path,
) -> None:
    headline = "Company signs supply agreement"
    digest = headline_norm_sha256(headline)
    path = tmp_path / "articles.jsonl"
    rows = [
        {
            "headline": headline,
            "clean_text": "short body",
            "version_created": "2026-01-01T00:00:00Z",
            "article_id": "a1",
        },
        {
            "headline": headline,
            "clean_text": "longer body describing the signed agreement",
            "version_created": "2026-01-01T01:00:00Z",
            "article_id": "a2",
        },
        {"headline": "No body", "clean_text": ""},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "status": "completed",
        "sharing": {"licensed_full_text": True},
        "files": {"articles_jsonl": {"sha256": sha256_file(path)}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    candidates, audit = load_full_text_candidates(path, eligible_hashes={digest})

    assert len(candidates) == 1
    assert candidates.loc[0, "article_id"] == "a2"
    assert candidates.loc[0, "proxy_stratum"] == "d1"
    assert audit["rows_with_nonempty_full_text"] == 2
    assert audit["unique_matching_headline_hashes"] == 1


def test_coder_sheet_validation_reports_progress_and_rejects_partial_schema() -> None:
    sheet = pd.DataFrame(
        [
            ["a", "headline", "text", "0", "none", "span", "high", ""],
            ["b", "headline", "text", "1", "", "", "", ""],
            ["c", "headline", "text", "", "", "", "", ""],
        ],
        columns=CODER_COLUMNS,
    )

    status = validate_coder_sheet(sheet, expected_rows=3, expected_ids={"a", "b", "c"})

    assert status == {
        "rows": 3,
        "complete_rows": 1,
        "partial_rows": 1,
        "unstarted_rows": 1,
        "is_complete": False,
    }
    invalid = sheet.copy()
    invalid.loc[0, "human_distance"] = "4"
    with pytest.raises(ValueError, match="invalid human_distance"):
        validate_coder_sheet(invalid, expected_rows=3)


def test_audit_state_allows_labels_but_rejects_immutable_text_changes(
    tmp_path: Path,
) -> None:
    allocation = {stratum: 12 for stratum in AUDIT_STRATA}
    coder_a, coder_b, key, summary = build_blinded_audit_sample(_candidate_frame(), allocation=allocation, seed=7)
    write_audit_pack(
        tmp_path,
        coder_a=coder_a,
        coder_b=coder_b,
        private_key=key,
        summary=summary,
        source_audit={},
        seed=7,
        allocation=allocation,
    )
    labels = [str(index % 4) for index in range(len(coder_a))]
    labelled_a = _complete_sheet(coder_a, labels)
    labelled_a.to_csv(tmp_path / "coder_a_200.csv", index=False)

    state = load_cash_flow_audit_state(tmp_path)

    assert state["coder_a_status"]["is_complete"] is True
    assert state["coder_b_status"]["is_complete"] is False
    assert state["ready_for_reliability"] is False

    labelled_a.loc[0, "clean_text"] = "tampered text"
    labelled_a.to_csv(tmp_path / "coder_a_200.csv", index=False)
    with pytest.raises(ValueError, match="immutable coder worksheet content changed"):
        load_cash_flow_audit_state(tmp_path)


def test_reliability_gate_passes_perfect_varied_labels() -> None:
    base = pd.DataFrame(
        {
            "audit_id": [f"a-{index}" for index in range(60)],
            "headline": "headline",
            "clean_text": "text",
        }
    )
    for column in CODER_COLUMNS[3:]:
        base[column] = ""
    labels = [str(index % 4) for index in range(59)] + ["NA"]
    coder_a = _complete_sheet(base.loc[:, CODER_COLUMNS], labels)
    coder_b = _complete_sheet(base.loc[:, CODER_COLUMNS], labels)

    metrics, matrix, bootstrap = evaluate_cash_flow_reliability(coder_a, coder_b, replications=200, seed=11)

    assert metrics["gate_pass"] is True
    assert metrics["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert metrics["exact_agreement"] == pytest.approx(1.0)
    assert metrics["adjacent_agreement"] == pytest.approx(1.0)
    assert metrics["joint_numeric_rows"] == 59
    assert matrix.to_numpy().sum() == 60
    assert len(bootstrap) == 200


def test_reliability_gate_fails_insufficient_numeric_overlap_and_disagreement() -> None:
    base = pd.DataFrame(
        {
            "audit_id": [f"a-{index}" for index in range(60)],
            "headline": "headline",
            "clean_text": "text",
        }
    )
    for column in CODER_COLUMNS[3:]:
        base[column] = ""
    labels_a = [str(index % 4) for index in range(60)]
    labels_b = [str(3 - (index % 4)) for index in range(47)] + ["NA"] * 13
    coder_a = _complete_sheet(base.loc[:, CODER_COLUMNS], labels_a)
    coder_b = _complete_sheet(base.loc[:, CODER_COLUMNS], labels_b)

    metrics, _, _ = evaluate_cash_flow_reliability(coder_a, coder_b, replications=200, seed=11)

    assert metrics["joint_numeric_rows"] == 47
    assert metrics["gate_pass"] is False
    assert metrics["exact_agreement"] < 0.5
