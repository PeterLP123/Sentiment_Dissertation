"""Run the frozen Notebook 85 post-hoc outcome and label sensitivities.

Writes aggregate CSVs only. Does not execute sealed Notebook 75, does not
touch LSEG, and does not rebuild the original nine-rule family.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.lib.aggregators import (
    benjamini_hochberg,
    benjamini_hochberg_q_values,
    firm_day_soft_negative_mass,
)
from experiments.lib.conditional_aggregation import conditional_negative_share_test
from experiments.lib.distribution import load_story_scores
from experiments.lib.factor_residuals import load_french_daily_ff3, trailing_ff3_residuals
from experiments.lib.panel import (
    DEFAULT_EVENTS_DB,
    DEFAULT_PRICE_ZIP,
    FROZEN_DEV_END,
    attach_close_returns,
    load_adjusted_open_prices,
    session_close_returns,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC_NAME = "fnspid_development_outcome_and_label_sensitivities_v1_20260813.json"
NB75 = ROOT / "experiments/notebooks/75_fnspid_evaluation_beyond_mean_replication.ipynb"
SOURCE_ARCHIVE = Path("/Users/peterprendergast/Documents/Sentiment_Dissertation")
SOURCE_PRICES = Path(
    "/Users/peterprendergast/Documents/Sentiment_Dissertation_data/FNSPID/"
    "bf9189c41527198897d1af3e17b1a0095279fc45/raw/Stock_price/full_history.zip"
)
AGGREGATORS_HASH = "1ca1c38109bc06af232119adeff17de66a567174d457038bc923a930a2da61f1"
PRICE_HASH = "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("none of these paths exist: " + ", ".join(str(p) for p in candidates))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def resolve_inputs(spec: dict[str, Any]) -> dict[str, Path]:
    agg = first_existing(
        [
            ROOT / "experiments/generated/03_aggregation/firm_day_aggregators.parquet",
            SOURCE_ARCHIVE / "final_experiments/outputs/03_aggregation/firm_day_aggregators.parquet",
        ]
    )
    prices = first_existing(
        [
            DEFAULT_PRICE_ZIP,
            Path(spec["inputs"]["fnspid_price_archive"]["path"]),
            SOURCE_PRICES,
        ]
    )
    events = first_existing(
        [
            DEFAULT_EVENTS_DB,
            SOURCE_ARCHIVE / "reports/loop_moment2_finbert_20260719/finbert_checkpoint.sqlite3",
        ]
    )
    factors = first_existing(
        [
            ROOT / "data/factors/french_ff3_daily.csv",
            ROOT / spec["inputs"]["french_daily_ff3"]["path"],
        ]
    )
    return {
        "aggregators": agg,
        "prices": prices,
        "events": events,
        "factors": factors,
    }


def _summary_row(
    *,
    name: str,
    role: str,
    outcome: str,
    focal: str,
    summary: dict[str, Any],
    audit: dict[str, int],
    in_family: bool,
) -> dict[str, Any]:
    return {
        "test": name,
        "role": role,
        "outcome": outcome,
        "focal_regressor": focal,
        "estimate": summary["estimate"],
        "se": summary["se"],
        "t": summary["t"],
        "p_two_sided": summary["p_two_sided"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "n_clusters": summary["n_clusters"],
        "n_rows_complete": summary["n_rows_complete"],
        "dates_used": audit["dates_used"],
        "dates_below_min_names": audit["dates_below_min_names"],
        "in_posthoc_family": in_family,
        "expected_direction": "negative",
        "direction_matches": summary["direction_matches"],
    }


def run(*, output_dir: Path | None = None) -> dict[str, Path]:
    if not NB75.exists():
        raise FileNotFoundError("sealed evaluation notebook path is missing")

    spec_path = ROOT / "experiments/specs" / SPEC_NAME
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    paths = resolve_inputs(spec)
    output = output_dir or (ROOT / "experiments/generated/85_fnspid_development_outcome_and_label_sensitivities")
    output.mkdir(parents=True, exist_ok=True)

    agg_hash = sha256(paths["aggregators"])
    price_hash = sha256(paths["prices"])
    if agg_hash != AGGREGATORS_HASH:
        raise RuntimeError(f"aggregators parquet hash mismatch: {agg_hash}")
    if price_hash != PRICE_HASH:
        raise RuntimeError(f"price zip hash mismatch: {price_hash}")

    panel = pd.read_parquet(paths["aggregators"])
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str)
    development = panel.loc[
        (panel["split"] == "development")
        & (panel["session_date"] <= pd.Timestamp(FROZEN_DEV_END))
    ].copy()

    prices, missing = load_adjusted_open_prices(
        paths["prices"],
        set(development["symbol"].str.upper()) | {"SPY"},
        start="2010-01-01",
        end="2020-01-31",
    )
    development = attach_close_returns(development, prices)
    dense_close = session_close_returns(prices)
    factors = load_french_daily_ff3(paths["factors"])
    residuals = trailing_ff3_residuals(dense_close, factors, window=252, min_obs=126)
    development = development.merge(
        residuals[["symbol", "session_date", "ff3_residual", "n_beta_obs"]],
        on=["symbol", "session_date"],
        how="left",
        validate="m:1",
    )

    stories = load_story_scores(
        paths["events"],
        development[["symbol", "session_date"]],
    )
    soft = firm_day_soft_negative_mass(stories)
    development = development.merge(soft, on=["symbol", "session_date"], how="left", validate="m:1")

    baseline_summary, _, baseline_audit = conditional_negative_share_test(
        development,
        date_col="session_date",
        outcome_col="ar_open_h1",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
    )
    close_summary, _, close_audit = conditional_negative_share_test(
        development,
        date_col="session_date",
        outcome_col="ar_close_h1",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
    )
    ff3_summary, _, ff3_audit = conditional_negative_share_test(
        development,
        date_col="session_date",
        outcome_col="ff3_residual",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
    )
    soft_summary, _, soft_audit = conditional_negative_share_test(
        development,
        date_col="session_date",
        outcome_col="ar_open_h1",
        mean_col="mean_continuous",
        negative_share_col="soft_negative_mass",
        count_col="n",
    )

    table = pd.DataFrame(
        [
            _summary_row(
                name="headline_open_hard_share",
                role="predeclared_baseline_not_in_family",
                outcome="ar_open_h1",
                focal="negative_share",
                summary=baseline_summary,
                audit=baseline_audit,
                in_family=False,
            ),
            _summary_row(
                name="close_to_close_hard_share",
                role="posthoc_sensitivity",
                outcome="ar_close_h1",
                focal="negative_share",
                summary=close_summary,
                audit=close_audit,
                in_family=True,
            ),
            _summary_row(
                name="ff3_residual_hard_share",
                role="posthoc_sensitivity",
                outcome="ff3_residual",
                focal="negative_share",
                summary=ff3_summary,
                audit=ff3_audit,
                in_family=True,
            ),
            _summary_row(
                name="open_soft_negative_mass",
                role="posthoc_sensitivity",
                outcome="ar_open_h1",
                focal="soft_negative_mass",
                summary=soft_summary,
                audit=soft_audit,
                in_family=True,
            ),
        ]
    )
    family = table["in_posthoc_family"].astype(bool)
    q_values = pd.Series(pd.NA, index=table.index, dtype="Float64")
    reject = pd.Series(False, index=table.index)
    family_p = table.loc[family, "p_two_sided"].astype(float).tolist()
    q_values.loc[family] = benjamini_hochberg_q_values(family_p)
    reject.loc[family] = benjamini_hochberg(family_p, q=0.05)
    table["q_bh_three_test"] = q_values
    table["bh_reject_q05"] = reject
    table["bh_family"] = [
        "posthoc_three_estimand_sensitivity" if flag else "predeclared_baseline_excluded"
        for flag in family
    ]

    coverage = pd.DataFrame(
        [
            {
                "item": "development_rows",
                "value": int(len(development)),
            },
            {
                "item": "development_symbols",
                "value": int(development["symbol"].nunique()),
            },
            {
                "item": "missing_price_symbols",
                "value": len(missing),
            },
            {
                "item": "rows_with_ar_close_h1",
                "value": int(development["ar_close_h1"].notna().sum()),
            },
            {
                "item": "rows_with_ff3_residual",
                "value": int(development["ff3_residual"].notna().sum()),
            },
            {
                "item": "rows_with_soft_negative_mass",
                "value": int(development["soft_negative_mass"].notna().sum()),
            },
            {
                "item": "story_rows_loaded",
                "value": int(len(stories)),
            },
        ]
    )

    table_path = output / "sensitivity_coefficients.csv"
    coverage_path = output / "coverage_audit.csv"
    table.to_csv(table_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    manifest = {
        "notebook": "85_fnspid_development_outcome_and_label_sensitivities",
        "status": "posthoc_development_sensitivity_not_confirmation",
        "git_commit_at_execution": git_commit(),
        "built_at_utc": datetime.now(UTC).isoformat(),
        "spec": str(spec_path.relative_to(ROOT)),
        "spec_sha256": sha256(spec_path),
        "inputs": {
            "aggregators": {"path": str(paths["aggregators"]), "sha256": agg_hash},
            "prices": {"path": str(paths["prices"]), "sha256": price_hash},
            "events": {"path": str(paths["events"]), "sha256": sha256(paths["events"])},
            "factors": {"path": str(paths["factors"]), "sha256": sha256(paths["factors"])},
        },
        "sealed_notebook_75": str(NB75.relative_to(ROOT)),
        "missing_price_symbols": missing,
        "coefficients": table.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "claim_boundary": spec["claim_boundaries"],
        "outputs": {
            "sensitivity_coefficients": str(table_path),
            "coverage_audit": str(coverage_path),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "coefficients": table_path,
        "coverage": coverage_path,
        "manifest": manifest_path,
    }


def main() -> int:
    written = run()
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
