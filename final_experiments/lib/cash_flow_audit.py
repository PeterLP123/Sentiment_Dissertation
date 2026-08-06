"""Blinded human-audit preparation for LSEG cash-flow distance.

The source corpus contains licensed Reuters text and is local-only.  This
module validates that corpus, constructs a deterministic return-blind sample,
and writes coder worksheets only inside the caller-selected ignored output
directory.  No realised return, sentiment score, or machine proxy is exposed
to either coder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

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
DISTANCE_LABELS: tuple[str, ...] = ("0", "1", "2", "3", "NA")
CONFIDENCE_LABELS: tuple[str, ...] = ("high", "medium", "low")
RELIABILITY_BOOTSTRAP_REPLICATIONS = 9_999
RELIABILITY_SEED = 20260819
MIN_JOINT_NUMERIC = 48
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


def worksheet_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash immutable worksheet content while allowing coding fields to change."""

    identity_columns = ("audit_id", "headline", "clean_text")
    missing = set(identity_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"worksheet identity columns missing: {sorted(missing)}")
    canonical = frame.loc[:, identity_columns].fillna("").astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_coder_sheet(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate one worksheet and report completion without exposing its text."""

    if tuple(frame.columns) != CODER_COLUMNS:
        raise ValueError("coder worksheet columns do not match the frozen schema")
    sheet = frame.fillna("").astype(str)
    ids = sheet["audit_id"].str.strip()
    if len(sheet) != expected_rows:
        raise ValueError(f"coder worksheet has {len(sheet)} rows; {expected_rows} required")
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("coder worksheet audit_id values must be nonblank and unique")
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise ValueError("coder worksheet audit_id set does not match the frozen sample")

    distance = sheet["human_distance"].str.strip().str.upper()
    confidence = sheet["confidence"].str.strip().str.lower()
    invalid_distance = distance.loc[distance.ne("") & ~distance.isin(DISTANCE_LABELS)]
    invalid_confidence = confidence.loc[confidence.ne("") & ~confidence.isin(CONFIDENCE_LABELS)]
    if not invalid_distance.empty:
        raise ValueError("coder worksheet contains an invalid human_distance value")
    if not invalid_confidence.empty:
        raise ValueError("coder worksheet contains an invalid confidence value")

    required = sheet.loc[
        :,
        (
            "human_distance",
            "unresolved_prerequisites",
            "evidence_span",
            "confidence",
        ),
    ].apply(lambda column: column.str.strip().ne(""))
    complete = required.all(axis=1)
    started = required.any(axis=1) | sheet["notes"].str.strip().ne("")
    partially_complete = started & ~complete

    prerequisites = sheet["unresolved_prerequisites"].str.strip().str.lower()
    bad_zero = complete & distance.eq("0") & prerequisites.ne("none")
    bad_nonzero = complete & distance.isin(("1", "2", "3")) & prerequisites.eq("none")
    if bad_zero.any():
        raise ValueError("completed distance-0 rows must record prerequisites as 'none'")
    if bad_nonzero.any():
        raise ValueError("completed distance-1/2/3 rows must name a prerequisite")

    return {
        "rows": int(len(sheet)),
        "complete_rows": int(complete.sum()),
        "partial_rows": int(partially_complete.sum()),
        "unstarted_rows": int((~started).sum()),
        "is_complete": bool(complete.all()),
    }


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
    return distance.map({0: "d0", 1: "d1", 2: "d2", 3: "d3"}).fillna("unmatched")


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
    candidates["proxy_stratum"] = _proxy_stratum(candidates["headline"] + "\n" + candidates["clean_text"])
    candidates = candidates.sort_values("headline_sha256", kind="mergesort").reset_index(drop=True)
    counts = candidates["proxy_stratum"].value_counts().reindex(AUDIT_STRATA, fill_value=0).astype(int)
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
            raise ValueError(f"stratum {stratum} has {len(pool)} candidates; {requested} required")
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
        double_positions = rng.choice(positions, size=DOUBLE_CODE_PER_STRATUM, replace=False)
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
            "human inter-coder reliability and criterion validity of a 0-3 cash-flow-distance construct; no return analysis is opened here"
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
        "worksheet_identity_sha256": {
            "coder_a_200.csv": worksheet_identity_sha256(coder_a),
            "coder_b_double_code_60.csv": worksheet_identity_sha256(coder_b),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_cash_flow_audit_state(output_dir: str | Path) -> dict[str, Any]:
    """Load a prepared audit pack and fail closed on sample or text changes."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"audit manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = {
        "coder_a_200.csv",
        "coder_b_double_code_60.csv",
        "private_sampling_key.csv",
        "sampling_summary.csv",
    }
    if not required_files.issubset(set(manifest.get("files") or {})):
        raise ValueError("audit manifest does not contain the frozen file contract")
    identities = manifest.get("worksheet_identity_sha256") or {}
    if set(identities) != {
        "coder_a_200.csv",
        "coder_b_double_code_60.csv",
    }:
        raise ValueError("audit manifest lacks immutable worksheet identities")

    for name in ("private_sampling_key.csv", "sampling_summary.csv"):
        path = output / name
        if not path.is_file() or sha256_file(path) != manifest["files"][name]:
            raise ValueError(f"frozen audit artifact hash mismatch: {name}")

    coder_a = pd.read_csv(output / "coder_a_200.csv", dtype=str, keep_default_na=False)
    coder_b = pd.read_csv(output / "coder_b_double_code_60.csv", dtype=str, keep_default_na=False)
    private_key = pd.read_csv(output / "private_sampling_key.csv")
    if "audit_id" not in private_key or "double_code" not in private_key:
        raise ValueError("private sampling key lacks audit_id or double_code")
    key_ids = set(private_key["audit_id"].astype(str))
    double_code = private_key["double_code"]
    if double_code.dtype != bool:
        double_code = double_code.astype(str).str.strip().str.lower().map({"true": True, "false": False})
        if double_code.isna().any():
            raise ValueError("private sampling key contains invalid double_code values")
    double_ids = set(private_key.loc[double_code, "audit_id"].astype(str))

    for name, sheet in (
        ("coder_a_200.csv", coder_a),
        ("coder_b_double_code_60.csv", coder_b),
    ):
        if worksheet_identity_sha256(sheet) != identities[name]:
            raise ValueError(f"immutable coder worksheet content changed: {name}")

    coder_a_status = validate_coder_sheet(
        coder_a,
        expected_rows=int(manifest["sample_rows"]),
        expected_ids=key_ids,
    )
    coder_b_status = validate_coder_sheet(
        coder_b,
        expected_rows=int(manifest["double_code_rows"]),
        expected_ids=double_ids,
    )
    return {
        "manifest": manifest,
        "coder_a": coder_a,
        "coder_b": coder_b,
        "private_key": private_key,
        "coder_a_status": coder_a_status,
        "coder_b_status": coder_b_status,
        "ready_for_reliability": bool(coder_a_status["is_complete"] and coder_b_status["is_complete"]),
    }


def evaluate_cash_flow_reliability(
    coder_a: pd.DataFrame,
    coder_b: pd.DataFrame,
    *,
    replications: int = RELIABILITY_BOOTSTRAP_REPLICATIONS,
    seed: int = RELIABILITY_SEED,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Evaluate the frozen double-code gate after both sheets are complete.

    Ordinal kappa excludes pairs containing ``NA``. Exact and adjacent
    agreement use every event: two ``NA`` labels agree and a mixed ``NA`` pair
    disagrees. Bootstrap resampling is by event across the complete overlap.
    """

    if replications <= 0:
        raise ValueError("replications must be positive")
    a_status = validate_coder_sheet(coder_a, expected_rows=len(coder_a))
    b_status = validate_coder_sheet(coder_b, expected_rows=len(coder_b))
    if not a_status["is_complete"] or not b_status["is_complete"]:
        raise ValueError("both coder worksheets must be complete before reliability")

    overlap = coder_b[["audit_id", "human_distance"]].merge(
        coder_a[["audit_id", "human_distance"]],
        on="audit_id",
        how="left",
        validate="one_to_one",
        suffixes=("_b", "_a"),
    )
    if overlap["human_distance_a"].isna().any():
        raise ValueError("coder B contains audit IDs absent from coder A")
    a = overlap["human_distance_a"].astype(str).str.strip().str.upper()
    b = overlap["human_distance_b"].astype(str).str.strip().str.upper()
    numeric = a.isin(("0", "1", "2", "3")) & b.isin(("0", "1", "2", "3"))
    joint_numeric = int(numeric.sum())

    def _weighted_kappa(left: pd.Series, right: pd.Series) -> float:
        if len(left) == 0:
            return float("nan")
        return float(
            cohen_kappa_score(
                left.astype(int),
                right.astype(int),
                labels=[0, 1, 2, 3],
                weights="quadratic",
            )
        )

    kappa = _weighted_kappa(a.loc[numeric], b.loc[numeric])
    exact = a.eq(b)
    adjacent = pd.Series(False, index=overlap.index)
    adjacent.loc[a.eq("NA") & b.eq("NA")] = True
    adjacent.loc[numeric] = a.loc[numeric].astype(int).sub(b.loc[numeric].astype(int)).abs().le(1)

    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(replications):
        positions = rng.integers(0, len(overlap), size=len(overlap))
        sampled_a = a.iloc[positions].reset_index(drop=True)
        sampled_b = b.iloc[positions].reset_index(drop=True)
        sampled_numeric = sampled_a.isin(("0", "1", "2", "3")) & sampled_b.isin(("0", "1", "2", "3"))
        sampled_kappa = _weighted_kappa(sampled_a.loc[sampled_numeric], sampled_b.loc[sampled_numeric])
        if np.isfinite(sampled_kappa):
            bootstrap_values.append(sampled_kappa)
    bootstrap = np.asarray(bootstrap_values, dtype=float)
    lower = float(np.quantile(bootstrap, 0.025)) if len(bootstrap) else float("nan")
    upper = float(np.quantile(bootstrap, 0.975)) if len(bootstrap) else float("nan")
    adjacent_rate = float(adjacent.mean())
    gate_pass = bool(
        joint_numeric >= MIN_JOINT_NUMERIC
        and np.isfinite(kappa)
        and kappa >= 0.60
        and np.isfinite(lower)
        and lower >= 0.40
        and adjacent_rate >= 0.80
    )
    metrics = {
        "overlap_rows": int(len(overlap)),
        "joint_numeric_rows": joint_numeric,
        "minimum_joint_numeric_rows": MIN_JOINT_NUMERIC,
        "quadratic_weighted_kappa": kappa,
        "bootstrap_replications_requested": int(replications),
        "bootstrap_replications_valid": int(len(bootstrap)),
        "bootstrap_seed": int(seed),
        "kappa_ci_lower_95": lower,
        "kappa_ci_upper_95": upper,
        "exact_agreement": float(exact.mean()),
        "adjacent_agreement": adjacent_rate,
        "gate_pass": gate_pass,
    }
    matrix = pd.DataFrame(
        confusion_matrix(a, b, labels=list(DISTANCE_LABELS)),
        index=pd.Index(DISTANCE_LABELS, name="coder_a"),
        columns=pd.Index(DISTANCE_LABELS, name="coder_b"),
    )
    return metrics, matrix, bootstrap


__all__ = [
    "AUDIT_ALLOCATION",
    "AUDIT_SEED",
    "AUDIT_STRATA",
    "CODER_COLUMNS",
    "CONFIDENCE_LABELS",
    "DISTANCE_LABELS",
    "DOUBLE_CODE_PER_STRATUM",
    "MIN_JOINT_NUMERIC",
    "RELIABILITY_BOOTSTRAP_REPLICATIONS",
    "RELIABILITY_SEED",
    "build_blinded_audit_sample",
    "evaluate_cash_flow_reliability",
    "load_cash_flow_audit_state",
    "load_full_text_candidates",
    "validate_coder_sheet",
    "worksheet_identity_sha256",
    "write_audit_pack",
]
