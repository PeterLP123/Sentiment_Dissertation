"""Blinded human-audit preparation for LSEG cash-flow distance.

The source corpus contains licensed Reuters text and is local-only.  This
module validates that corpus, constructs a deterministic return-blind sample,
and writes coder worksheets only inside the caller-selected ignored output
directory.  No realised return, sentiment score, or machine proxy is exposed
to either coder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.novelty import headline_norm_sha256
from final_experiments.lib.sparse_events import classify_proxy_cash_flow_distance
from sentiment_benchmark.artifact_io import sha256_file

AUDIT_SEED = 20260805
AUDIT_STRATA: tuple[str, ...] = ("d0", "d1", "d2", "d3", "unmatched")
AUDIT_ALLOCATION: dict[str, int] = {
    "d0": 35,
    "d1": 35,
    "d2": 35,
    "d3": 60,
    "unmatched": 35,
}
DOUBLE_CODE_PER_STRATUM = 12
CODER_COLUMNS: tuple[str, ...] = (
    "audit_id",
    "headline",
    "clean_text",
    "human_distance",
    "unresolved_prerequisites",
    "evidence_span",
    "confidence",
    "notes",
)
FORBIDDEN_CODER_COLUMNS: frozenset[str] = frozenset(
    {
        "headline_sha256",
        "proxy_stratum",
        "proxy_cash_flow_distance",
        "score",
        "label",
        "return",
        "raw_open_h1",
        "ar_open_h1",
        "matched_symbols",
        "version_created",
    }
)


def _proxy_stratum(text: pd.Series) -> pd.Series:
    distance = classify_proxy_cash_flow_distance(text)
    return distance.map({0: "d0", 1: "d1", 2: "d2", 3: "d3"}).fillna(
        "unmatched"
    )


def load_full_text_candidates(
    articles_path: str | Path,
    *,
    eligible_hashes: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one deterministic full-text Reuters representative per headline.

    The longest cleaned body is retained for a repeated normalised headline;
    ties are broken by the earliest version timestamp and then article id.
    Returned rows intentionally contain licensed text and must remain local.
    """

    path = Path(articles_path)
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"articles corpus or manifest is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("articles manifest is not completed")
    if not bool((manifest.get("sharing") or {}).get("licensed_full_text")):
        raise ValueError("articles manifest does not declare licensed full text")
    contract = (manifest.get("files") or {}).get("articles_jsonl") or {}
    actual_hash = sha256_file(path)
    if contract.get("sha256") != actual_hash:
        raise ValueError("articles corpus hash mismatch")

    records: list[dict[str, Any]] = []
    rows_read = 0
    rows_with_text = 0
    rows_matching_scores = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_read += 1
            headline = str(row.get("headline") or "").strip()
            clean_text = str(row.get("clean_text") or "").strip()
            if not headline or not clean_text:
                continue
            rows_with_text += 1
            digest = headline_norm_sha256(headline)
            if digest not in eligible_hashes:
                continue
            rows_matching_scores += 1
            records.append(
                {
                    "headline_sha256": digest,
                    "headline": headline,
                    "clean_text": clean_text,
                    "cleaned_chars": len(clean_text),
                    "version_created": str(row.get("version_created") or ""),
                    "article_id": str(row.get("article_id") or ""),
                }
            )

    if not records:
        raise ValueError("no full-text rows match the eligible score population")
    candidates = pd.DataFrame.from_records(records)
    candidates = candidates.sort_values(
        ["headline_sha256", "cleaned_chars", "version_created", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).drop_duplicates("headline_sha256", keep="first")
    candidates["proxy_stratum"] = _proxy_stratum(
        candidates["headline"] + "\n" + candidates["clean_text"]
    )
    candidates = candidates.sort_values("headline_sha256", kind="mergesort").reset_index(
        drop=True
    )
    counts = (
        candidates["proxy_stratum"]
        .value_counts()
        .reindex(AUDIT_STRATA, fill_value=0)
        .astype(int)
    )
    audit = {
        "rows_read": rows_read,
        "rows_with_nonempty_full_text": rows_with_text,
        "rows_matching_successful_gemma_hashes": rows_matching_scores,
        "unique_matching_headline_hashes": int(len(candidates)),
        "candidate_stratum_counts": counts.to_dict(),
        "articles_sha256": actual_hash,
    }
    return candidates, audit


def build_blinded_audit_sample(
    candidates: pd.DataFrame,
    *,
    allocation: dict[str, int] | None = None,
    seed: int = AUDIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return coder A, coder B, private key, and aggregate stratum counts."""

    required = {
        "headline_sha256",
        "headline",
        "clean_text",
        "proxy_stratum",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    target = dict(AUDIT_ALLOCATION if allocation is None else allocation)
    if set(target) != set(AUDIT_STRATA):
        raise ValueError(f"allocation must cover exactly {list(AUDIT_STRATA)}")
    if any(int(value) <= 0 for value in target.values()):
        raise ValueError("every audit stratum allocation must be positive")

    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for stratum in AUDIT_STRATA:
        pool = candidates.loc[candidates["proxy_stratum"].eq(stratum)].copy()
        requested = int(target[stratum])
        if len(pool) < requested:
            raise ValueError(
                f"stratum {stratum} has {len(pool)} candidates; {requested} required"
            )
        chosen = rng.choice(pool.index.to_numpy(), size=requested, replace=False)
        sample = pool.loc[chosen].copy()
        sample["inclusion_probability"] = requested / len(pool)
        selected_parts.append(sample)
        summary_rows.append(
            {
                "proxy_stratum": stratum,
                "candidate_count": int(len(pool)),
                "sample_count": requested,
                "inclusion_probability": requested / len(pool),
                "double_code_count": DOUBLE_CODE_PER_STRATUM,
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected["audit_id"] = [f"CFDA-{i:03d}" for i in range(1, len(selected) + 1)]
    selected["double_code"] = False
    for stratum in AUDIT_STRATA:
        positions = selected.index[selected["proxy_stratum"].eq(stratum)].to_numpy()
        if len(positions) < DOUBLE_CODE_PER_STRATUM:
            raise ValueError(f"stratum {stratum} cannot supply double-code subset")
        double_positions = rng.choice(
            positions, size=DOUBLE_CODE_PER_STRATUM, replace=False
        )
        selected.loc[double_positions, "double_code"] = True

    coder_base = selected[["audit_id", "headline", "clean_text"]].copy()
    for column in CODER_COLUMNS[3:]:
        coder_base[column] = ""
    coder_a = coder_base.loc[:, CODER_COLUMNS]
    coder_b = coder_base.loc[selected["double_code"]].copy()
    coder_b = coder_b.iloc[rng.permutation(len(coder_b))].reset_index(drop=True)
    coder_b = coder_b.loc[:, CODER_COLUMNS]

    key_columns = [
        "audit_id",
        "headline_sha256",
        "proxy_stratum",
        "inclusion_probability",
        "double_code",
        "version_created",
        "article_id",
        "cleaned_chars",
    ]
    private_key = selected[[c for c in key_columns if c in selected.columns]].copy()
    summary = pd.DataFrame.from_records(summary_rows)

    if FORBIDDEN_CODER_COLUMNS & set(coder_a.columns):
        raise AssertionError("coder A worksheet leaks blinded fields")
    if FORBIDDEN_CODER_COLUMNS & set(coder_b.columns):
        raise AssertionError("coder B worksheet leaks blinded fields")
    if len(coder_a) != sum(target.values()):
        raise AssertionError("coder A sample size does not match allocation")
    expected_double = DOUBLE_CODE_PER_STRATUM * len(AUDIT_STRATA)
    if len(coder_b) != expected_double:
        raise AssertionError("coder B subset size does not match frozen design")
    return coder_a, coder_b, private_key, summary


def write_audit_pack(
    output_dir: str | Path,
    *,
    coder_a: pd.DataFrame,
    coder_b: pd.DataFrame,
    private_key: pd.DataFrame,
    summary: pd.DataFrame,
    source_audit: dict[str, Any],
    seed: int = AUDIT_SEED,
    allocation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Write the local audit package and a non-text manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    coder_a.to_csv(output / "coder_a_200.csv", index=False)
    coder_b.to_csv(output / "coder_b_double_code_60.csv", index=False)
    private_key.to_csv(output / "private_sampling_key.csv", index=False)
    summary.to_csv(output / "sampling_summary.csv", index=False)
    manifest = {
        "status": "prepared_unlabelled",
        "seed": int(seed),
        "estimand": (
            "human inter-coder reliability and criterion validity of a 0-3 "
            "cash-flow-distance construct; no return analysis is opened here"
        ),
        "sample_rows": int(len(coder_a)),
        "double_code_rows": int(len(coder_b)),
        "allocation": dict(AUDIT_ALLOCATION if allocation is None else allocation),
        "blinding": {
            "omitted_from_coder_sheets": sorted(FORBIDDEN_CODER_COLUMNS),
            "machine_stratum_visible_to_coders": False,
            "sentiment_visible_to_coders": False,
            "returns_visible_to_coders": False,
        },
        "source_audit": source_audit,
        "files": {
            name: sha256_file(output / name)
            for name in (
                "coder_a_200.csv",
                "coder_b_double_code_60.csv",
                "private_sampling_key.csv",
                "sampling_summary.csv",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "AUDIT_ALLOCATION",
    "AUDIT_SEED",
    "AUDIT_STRATA",
    "CODER_COLUMNS",
    "DOUBLE_CODE_PER_STRATUM",
    "build_blinded_audit_sample",
    "load_full_text_candidates",
    "write_audit_pack",
]
