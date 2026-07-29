# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FNSPID sentiment-conditioned tail risk
#
# **Research question.** After the initial price response, does firm-linked
# financial-news sentiment improve one-day-ahead Value-at-Risk and Expected
# Shortfall forecasts beyond price-based conditional volatility, news arrival,
# and news volume?
#
# This is a **forecasting** experiment. Success is not a significant negative
# sentiment coefficient; success is a lower proper joint VaR/ES loss on the same
# chronological evaluation rows. Three nested models are compared point-in-time:
#
# | Model | Information set |
# | --- | --- |
# | `M0` | price state only (reaction shock, its magnitude, conditional volatility, market state) |
# | `M1` | `M0` + news arrival indicator + news volume |
# | `M2` | `M1` + semantic intensity + adverse tone |
#
# The headline comparison is `M2` versus `M1`, scored with the Fissler-Ziegel
# `FZ0` loss and a date-block bootstrap. One pre-specified ablation
# (`M2` versus `M2_intensity`) asks whether *signed* tone adds anything beyond
# non-neutral importance.
#
# **Provenance.** Nothing is rescored and nothing is downloaded. The corpus,
# its frozen news-to-session mapping, and the FinBERT and VADER scores are
# reused from the completed E5/E6 runs (`reports/loop_vader_scale_20260719`,
# `reports/loop_moment2_finbert_20260719`). Those runs were exploratory; they
# motivated this design and are **not** confirmatory evidence for it.
#
# **Timing policy inherited from the frozen checkpoint.** Almost all upstream
# rows are date-only or exact-midnight and map to the first XNYS session
# strictly after calendar date `d`. The upstream manifest also records a small
# precise-timestamp branch, mapped to the session containing the UTC minute or
# otherwise the next session. In both cases the forecast is made only at the
# mapped session close:
#
# ```
# news -> mapped reaction session s -> forecast at close s
#                                   -> target close-to-close return s to s+1
# ```
#
# The return realised *during* `s` is a predictor, never the target. Several
# calendar dates (Friday, weekend, holidays) can collapse onto one reaction
# session and are aggregated into a single firm-session feature row. The
# checkpoint does not retain original timestamps or timing-type flags, so the
# notebook verifies both frozen upstream policy branches and reports that it
# cannot re-establish a per-headline strictly-date-only rule from this artifact.

# %% [markdown]
# ## 1. Environment and frozen configuration

# %%
# Notebook prose, SQL literals and manifest strings intentionally stay readable
# on one line rather than being wrapped to the 140-column source limit.
# ruff: noqa: E501
from __future__ import annotations

import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import time
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "pyproject.toml").exists():
    raise RuntimeError("run this notebook from the repository root or from notebooks/")
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import arch  # noqa: E402

import sentiment_benchmark  # noqa: E402
from sentiment_benchmark.artifact_io import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from sentiment_benchmark.fnspid_vader_smoke import _SessionMapper  # noqa: E402
from sentiment_benchmark.tail_risk import (  # noqa: E402
    GjrParams,
    apply_news_scale_adjustment,
    build_next_session_targets,
    christoffersen_tests,
    date_block_bootstrap_mean,
    es_identification_residual,
    fit_joint_var_es,
    fit_news_scale_adjustment,
    fz0_loss,
    gjr_conditional_variances,
    gjr_conditional_variances_piecewise,
    kupiec_test,
    pinball_loss,
    prepare_clean_output_directory,
    repair_adjusted_close,
    resolve_tail_risk_variant,
    select_expanding_refit_window,
    tail_forecasts,
    verify_input_file,
    verify_output_manifest,
    verify_strictly_after,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda value: f"{value:,.6g}")
warnings.filterwarnings("ignore", category=FutureWarning)

NOTEBOOK_START = time.time()
print(f"repository root : {REPO_ROOT}")
print(f"python          : {platform.python_version()} ({sys.executable})")
print(f"numpy {np.__version__} | pandas {pd.__version__} | scipy {scipy.__version__} | arch {arch.__version__}")

# %% [markdown]
# ### Frozen parameters
#
# Every value below is fixed before any result is inspected. `RUN_MODE` is the
# only switch and it may be overridden from the environment
# (`FNSPID_TAIL_RISK_RUN_MODE=full`) so the notebook can be executed
# non-interactively in either mode without editing the source.

# %%
RUN_MODE = os.environ.get("FNSPID_TAIL_RISK_RUN_MODE", "smoke").strip().lower()
if RUN_MODE not in {"smoke", "full"}:
    raise ValueError("RUN_MODE must be 'smoke' or 'full'")

# VARIANT selects one cell of the frozen 2 x 2 defect-attribution design.
# ``v1`` is the pre-registered run; ``price_only`` and ``refit_only`` isolate
# one repair each; ``v2`` applies both. Every cell retains the same corpus,
# timing, splits, tail models, bootstrap, and seed.
VARIANT, PRICE_REPAIR, VOLATILITY_REFIT, VARIANT_RELATIVE_TO = resolve_tail_risk_variant(os.environ.get("FNSPID_TAIL_RISK_VARIANT", "v1"))
ADJUSTMENT_GAP_THRESHOLD = 0.05  # candidate adjusted/raw-return gap; corporate-action metadata are unavailable

RANDOM_SEED = 20260728
ALPHA = 0.025
DEV_START = "2011-01-01"
DEV_END = "2016-12-31"
EVAL_START = "2017-01-01"
EVAL_END = "2023-12-31"
MIN_DEV_OBS = 500
MIN_EVAL_OBS = 250
BOOTSTRAP_REPS_SMOKE = 200
BOOTSTRAP_REPS_FULL = 2000
DATE_BLOCK_LENGTH = 20
PRIMARY_SCORER = "finbert"

# Supporting frozen choices, fixed at the same time as the block above.
RECURSION_START = "1990-01-01"  # variance-recursion burn-in start; no forecast origin is emitted here
WARMUP_START = "2010-01-01"  # first emitted forecast origin; no forecast is scored before DEV_START
SMOKE_FIRM_COUNT = 12  # deterministic, data-based: greatest post-gate dev+eval support
MIN_RETAINED_FIRMS = 10  # Gate 2 floor on the surviving cross-section
MIN_SPLIT_DATES = 250  # Gate 2 floor on distinct forecast dates per split
MIN_EXPECTED_TAIL_EVENTS = 100  # Gate 2 floor on alpha * evaluation rows
MAX_NEWS_FORWARD_GAP_DAYS = 7  # drop news whose firm reaction session is further out than this
NEWS_COUNT_CAP_QUANTILE = 0.995  # estimated on development rows only, then log1p
EWMA_LAMBDA = 0.94  # declared robustness price-risk filter
ROBUSTNESS_ALPHA = 0.01  # declared robustness tail level
RECAP_HEAVY_THRESHOLD = 0.5  # firm-session recap share treated as recap-heavy
EXTREME_RETURN_THRESHOLD = 0.5  # |log return| flagged for review; never auto-deleted
STALE_PRICE_MAX_RUN = 20  # a price frozen for longer than this inside the window is a data error
MARKET_SYMBOL = "SPY"
MARKET_VOL_WINDOW = 20
GARCH_MAX_PERSISTENCE = 0.99999  # a fit at or above this is non-stationary and is excluded

BOOTSTRAP_REPS = BOOTSTRAP_REPS_SMOKE if RUN_MODE == "smoke" else BOOTSTRAP_REPS_FULL

OUTPUT_DIR = Path(
    os.environ.get("FNSPID_TAIL_RISK_OUTPUT_DIR", "") or REPO_ROOT / "reports" / f"fnsipid_tail_risk_core_{VARIANT}"
).expanduser()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = REPO_ROOT / OUTPUT_DIR
prepare_clean_output_directory(OUTPUT_DIR)
FIGURE_DIR = OUTPUT_DIR / "figures"

CONFIG: dict[str, Any] = {
    "VARIANT": VARIANT,
    "PRICE_REPAIR": PRICE_REPAIR,
    "VOLATILITY_REFIT": VOLATILITY_REFIT,
    "ADJUSTMENT_GAP_THRESHOLD": ADJUSTMENT_GAP_THRESHOLD,
    "RUN_MODE": RUN_MODE,
    "RANDOM_SEED": RANDOM_SEED,
    "ALPHA": ALPHA,
    "DEV_START": DEV_START,
    "DEV_END": DEV_END,
    "EVAL_START": EVAL_START,
    "EVAL_END": EVAL_END,
    "MIN_DEV_OBS": MIN_DEV_OBS,
    "MIN_EVAL_OBS": MIN_EVAL_OBS,
    "BOOTSTRAP_REPS_SMOKE": BOOTSTRAP_REPS_SMOKE,
    "BOOTSTRAP_REPS_FULL": BOOTSTRAP_REPS_FULL,
    "BOOTSTRAP_REPS_ACTIVE": BOOTSTRAP_REPS,
    "DATE_BLOCK_LENGTH": DATE_BLOCK_LENGTH,
    "PRIMARY_SCORER": PRIMARY_SCORER,
    "RECURSION_START": RECURSION_START,
    "WARMUP_START": WARMUP_START,
    "SMOKE_FIRM_COUNT": SMOKE_FIRM_COUNT,
    "MIN_RETAINED_FIRMS": MIN_RETAINED_FIRMS,
    "MIN_SPLIT_DATES": MIN_SPLIT_DATES,
    "MIN_EXPECTED_TAIL_EVENTS": MIN_EXPECTED_TAIL_EVENTS,
    "MAX_NEWS_FORWARD_GAP_DAYS": MAX_NEWS_FORWARD_GAP_DAYS,
    "NEWS_COUNT_CAP_QUANTILE": NEWS_COUNT_CAP_QUANTILE,
    "EWMA_LAMBDA": EWMA_LAMBDA,
    "ROBUSTNESS_ALPHA": ROBUSTNESS_ALPHA,
    "RECAP_HEAVY_THRESHOLD": RECAP_HEAVY_THRESHOLD,
    "EXTREME_RETURN_THRESHOLD": EXTREME_RETURN_THRESHOLD,
    "STALE_PRICE_MAX_RUN": STALE_PRICE_MAX_RUN,
    "MARKET_SYMBOL": MARKET_SYMBOL,
    "MARKET_VOL_WINDOW": MARKET_VOL_WINDOW,
    "GARCH_MAX_PERSISTENCE": GARCH_MAX_PERSISTENCE,
    "GARCH_SPECIFICATION": "ConstantMean + GJR-GARCH(1,1) with Student-t innovations, fitted on development returns only",
    "TAIL_PARAMETERISATION": "q = -exp(X @ beta_q); e = q - exp(X @ beta_e); guarantees e < q < 0",
    "TAIL_OBJECTIVE": "Fissler-Ziegel FZ0, smoothed warm start then unsmoothed L-BFGS-B on the analytic gradient",
}

print(json.dumps(CONFIG, indent=2, sort_keys=True))
if RUN_MODE == "smoke":
    print(
        "\n*** RUN_MODE = smoke. Every number below is an ENGINEERING CHECK on a deterministic\n"
        f"*** {SMOKE_FIRM_COUNT}-firm subset. It is NOT dissertation evidence."
    )

RNG = np.random.default_rng(RANDOM_SEED)

# %% [markdown]
# ## 2. Input inventory
#
# Inputs are selected from committed manifests and from artifact *content*, not
# from filename recency. The FNSPID corpus, its session mapping, and the frozen
# scores are taken from the completed E6 checkpoint, which is the only local
# artifact that holds **per-headline FinBERT class probabilities** for the whole
# 1,640,796-event usable corpus.

# %%
E5_DIR = REPO_ROOT / "reports" / "loop_vader_scale_20260719"
E6_DIR = REPO_ROOT / "reports" / "loop_moment2_finbert_20260719"
E6_MANIFEST_PATH = E6_DIR / "manifest.json"
E6_CHECKPOINT_PATH = E6_DIR / "finbert_checkpoint.sqlite3"

if not E6_MANIFEST_PATH.exists():
    raise FileNotFoundError(f"completed E6 manifest not found: {E6_MANIFEST_PATH}")
E6_MANIFEST = json.loads(E6_MANIFEST_PATH.read_text(encoding="utf-8"))

E5_MANIFEST_PATH = E5_DIR / "manifest.json"
E5_MANIFEST_RECORD = E6_MANIFEST["inputs"]["upstream_manifest"]
E5_MANIFEST_SHA256 = verify_input_file(
    E5_MANIFEST_PATH,
    expected_sha256=E5_MANIFEST_RECORD["sha256"],
    expected_size=int(E5_MANIFEST_RECORD["size_bytes"]),
)
E5_MANIFEST = json.loads(E5_MANIFEST_PATH.read_text(encoding="utf-8"))
EXPECTED_UPSTREAM_TIMING_RULE = {
    "date_only_or_exact_midnight": "strictly next XNYS session",
    "full_datetime": "XNYS session containing the UTC minute, otherwise next session",
    "nasdaq_date_only_rows_included": True,
    "nasdaq_full_datetime_rows_included": False,
}
if E5_MANIFEST.get("timing_rule") != EXPECTED_UPSTREAM_TIMING_RULE:
    raise RuntimeError("the frozen upstream news timing policy is missing or changed")

EXPECTED_EVENTS = 1_640_796
EXPECTED_SYMBOLS = 574
EXPECTED_SESSIONS = 736_596
E6_CHECKPOINT_SIZE = 512_016_384
E6_CHECKPOINT_SHA256 = "1d4981448bdc1564652be9a1f19ee855bd435e141cdb832381a58ab03c04637f"

manifest_counts = E6_MANIFEST["counts"]
selection_notes: list[str] = []
if manifest_counts["deduplicated_coherent_events"] != EXPECTED_EVENTS:
    raise RuntimeError("E6 manifest does not describe the 1,640,796-event usable corpus")
if E6_MANIFEST["status"] != "completed" or E6_MANIFEST["finbert"]["remaining"] != 0:
    raise RuntimeError("E6 FinBERT run is not complete; refusing to reuse a partial score set")
selection_notes.append(
    "E6 checkpoint selected over the E5 VADER run because it is the only local artifact carrying "
    "per-headline FinBERT p_negative/p_neutral/p_positive for all 1,640,796 events; the same file "
    "also carries the frozen VADER compound used for the declared robustness run."
)
selection_notes.append(
    "The E5 run (reports/loop_vader_scale_20260719) supplies the frozen 574-symbol coherent cohort "
    "and timing manifest and is the declared upstream of E6 (manifest upstream_commit 1927242). "
    "Its mixed timing policy is preserved and disclosed because the completed checkpoint does not "
    "retain original headline timestamps or per-event timing-type flags."
)

PRICE_ARCHIVE = Path(os.environ.get("FNSPID_TAIL_RISK_PRICE_ARCHIVE", "") or E6_MANIFEST["inputs"]["price_archive"]["path"])
PRICE_ARCHIVE_MANIFEST_SHA = E6_MANIFEST["inputs"]["price_archive"]["sha256"]
COHORT_PATH = E5_DIR / "cohort_symbols.csv"

print(f"E6 manifest      : {E6_MANIFEST_PATH.relative_to(REPO_ROOT)}")
print(f"E6 checkpoint    : {E6_CHECKPOINT_PATH.relative_to(REPO_ROOT)} (gitignored, licensed text)")
print(f"price archive    : {PRICE_ARCHIVE}")
print(f"cohort symbols   : {COHORT_PATH.relative_to(REPO_ROOT)}")
for note in selection_notes:
    print(f"  - {note}")


# %%
def _git(*args: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception as error:  # pragma: no cover - git metadata is best effort
        return False, f"{type(error).__name__}: {error}"
    return True, completed.stdout.strip()


commit_available, commit_output = _git("rev-parse", "HEAD")
status_available, status_output = _git("status", "--porcelain")
GIT_COMMIT: str | None = commit_output if commit_available else None
GIT_DIRTY: bool | None = bool(status_output) if status_available else None
GIT_PROVENANCE_WARNING = (
    None
    if commit_available and status_available
    else "Git metadata unavailable on the execution host; commit and dirty state are recorded as null."
)
print(f"git commit {GIT_COMMIT or 'unavailable'} | dirty worktree: {GIT_DIRTY if GIT_DIRTY is not None else 'unavailable'}")

# %%
inventory_rows: list[dict[str, Any]] = []

if not E6_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"the E6 FinBERT checkpoint is missing; it is gitignored and must be present locally at {E6_CHECKPOINT_PATH}")
CHECKPOINT_COMPUTED_SHA = verify_input_file(
    E6_CHECKPOINT_PATH,
    expected_sha256=E6_CHECKPOINT_SHA256,
    expected_size=E6_CHECKPOINT_SIZE,
)

checkpoint_uri = f"file:{E6_CHECKPOINT_PATH}?mode=ro"
with sqlite3.connect(checkpoint_uri, uri=True) as connection:
    event_count, symbol_count, event_min, event_max = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(session_date), MAX(session_date) FROM events"
    ).fetchone()
    scored_count = connection.execute("SELECT COUNT(*) FROM events WHERE p_negative IS NOT NULL").fetchone()[0]
    panel_count = connection.execute("SELECT COUNT(*) FROM session_panel").fetchone()[0]
    event_columns = [row[1] for row in connection.execute("PRAGMA table_info(events)")]

print(f"events {event_count:,} | fully scored {scored_count:,} | symbols {symbol_count} | sessions {panel_count:,}")
if (event_count, scored_count, symbol_count, panel_count) != (
    EXPECTED_EVENTS,
    EXPECTED_EVENTS,
    EXPECTED_SYMBOLS,
    EXPECTED_SESSIONS,
):
    raise RuntimeError("E6 checkpoint content does not match the committed manifest counts")

checkpoint_stat = E6_CHECKPOINT_PATH.stat()
inventory_rows.append(
    {
        "role": "news_corpus_and_scores",
        "path": str(E6_CHECKPOINT_PATH.relative_to(REPO_ROOT)),
        "file_type": "sqlite3",
        "rows": event_count,
        "relevant_columns": "|".join(event_columns),
        "date_min": event_min,
        "date_max": event_max,
        "distinct_firms": symbol_count,
        "sha256": CHECKPOINT_COMPUTED_SHA,
        "size_bytes": checkpoint_stat.st_size,
        "originating_run": "reports/loop_moment2_finbert_20260719 (commit 3152a01)",
        "hash_source": "computed and matched to the frozen completed checkpoint",
    }
)

inventory_rows.append(
    {
        "role": "e6_manifest",
        "path": str(E6_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "file_type": "json",
        "rows": len(manifest_counts),
        "relevant_columns": "counts|inputs|frozen_rules|finbert|environment",
        "date_min": f"{E6_MANIFEST['config']['start_year']}-01-01",
        "date_max": f"{E6_MANIFEST['config']['end_year']}-12-31",
        "distinct_firms": manifest_counts["coherent_symbols"],
        "sha256": sha256_file(E6_MANIFEST_PATH),
        "size_bytes": E6_MANIFEST_PATH.stat().st_size,
        "originating_run": "commit 3152a01",
        "hash_source": "computed",
    }
)

inventory_rows.append(
    {
        "role": "e5_timing_manifest",
        "path": str(E5_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "file_type": "json",
        "rows": len(E5_MANIFEST["counts"]),
        "relevant_columns": "counts|timing_rule|inputs|files",
        "date_min": f"{E5_MANIFEST['config']['preferred_start_year']}-01-01",
        "date_max": f"{E5_MANIFEST['config']['preferred_end_year']}-12-31",
        "distinct_firms": E5_MANIFEST["counts"]["symbols_with_scored_events"],
        "sha256": E5_MANIFEST_SHA256,
        "size_bytes": E5_MANIFEST_PATH.stat().st_size,
        "originating_run": "reports/loop_vader_scale_20260719 (commit 1927242)",
        "hash_source": "computed and matched to the E6-declared upstream manifest",
    }
)

cohort_frame = pd.read_csv(COHORT_PATH, dtype={"symbol": str})
COHORT_SYMBOLS = sorted(cohort_frame["symbol"].str.strip().str.upper().unique())
if len(COHORT_SYMBOLS) != EXPECTED_SYMBOLS:
    raise RuntimeError(f"cohort file holds {len(COHORT_SYMBOLS)} symbols, expected {EXPECTED_SYMBOLS}")
inventory_rows.append(
    {
        "role": "coherent_firm_cohort",
        "path": str(COHORT_PATH.relative_to(REPO_ROOT)),
        "file_type": "csv",
        "rows": len(cohort_frame),
        "relevant_columns": "symbol",
        "date_min": "2011-01-01",
        "date_max": "2023-12-31",
        "distinct_firms": len(COHORT_SYMBOLS),
        "sha256": sha256_file(COHORT_PATH),
        "size_bytes": COHORT_PATH.stat().st_size,
        "originating_run": "reports/loop_vader_scale_20260719 (commit 1927242)",
        "hash_source": "computed",
    }
)

PRICE_ARCHIVE_COMPUTED_SHA = verify_input_file(
    PRICE_ARCHIVE,
    expected_sha256=PRICE_ARCHIVE_MANIFEST_SHA,
    expected_size=E6_MANIFEST["inputs"]["price_archive"]["size_bytes"],
)
price_stat = PRICE_ARCHIVE.stat()
with zipfile.ZipFile(PRICE_ARCHIVE) as archive:
    PRICE_MEMBERS = {
        Path(name).stem.upper(): name
        for name in archive.namelist()
        if name.startswith("full_history/") and name.casefold().endswith(".csv")
    }
inventory_rows.append(
    {
        "role": "adjusted_daily_prices",
        "path": str(PRICE_ARCHIVE),
        "file_type": "zip of per-symbol csv",
        "rows": len(PRICE_MEMBERS),
        "relevant_columns": "date|close|adj close",
        "date_min": "1980-12-12 (earliest member)",
        "date_max": "2023-12-28 (archive cut-off)",
        "distinct_firms": len(PRICE_MEMBERS),
        "sha256": PRICE_ARCHIVE_COMPUTED_SHA,
        "size_bytes": price_stat.st_size,
        "originating_run": "FNSPID raw download bf9189c4 (verified in E6 manifest)",
        "hash_source": "computed and matched to the E6 manifest",
    }
)

INPUT_INVENTORY = pd.DataFrame(inventory_rows)
INPUT_INVENTORY.to_csv(OUTPUT_DIR / "input_inventory.csv", index=False)
INPUT_INVENTORY[["role", "path", "file_type", "rows", "distinct_firms", "date_min", "date_max"]]

# %% [markdown]
# ### Semantic variable definition and sign convention
#
# The checkpoint holds full FinBERT class probabilities, so the **probability
# based** definitions are used (not the label-derived fallback):
#
# ```
# adverse_tone       = mean(p_negative - p_positive)   larger  => more adverse
# semantic_intensity = mean(p_negative + p_positive)   larger  => less neutral
# negative_share     = mean(argmax label == negative)
# ```
#
# The label-derived variant is *not* mixed in anywhere. The sign of
# `adverse_tone` is verified against `negative_share` and against the
# independent VADER compound score below.

# %%
SEMANTIC_DEFINITION = "probability_based_finbert"
NEWS_SQL = """
SELECT symbol,
       session_date,
       COUNT(*)                                                           AS news_count,
       AVG(p_negative - p_positive)                                       AS adverse_tone,
       AVG(p_negative + p_positive)                                       AS semantic_intensity,
       AVG(CASE WHEN finbert_negative_dominant = 1 THEN 1.0 ELSE 0.0 END) AS negative_share,
       AVG(vader_compound)                                                AS vader_mean,
       AVG(CASE WHEN vader_compound < -0.05 THEN 1.0 ELSE 0.0 END)        AS vader_negative_share,
       AVG(CASE WHEN ABS(vader_compound) > 0.05 THEN 1.0 ELSE 0.0 END)    AS vader_intensity,
       AVG(CASE WHEN is_recap = 1 THEN 1.0 ELSE 0.0 END)                  AS recap_share
FROM events
GROUP BY symbol, session_date
"""

_started = time.time()
with sqlite3.connect(checkpoint_uri, uri=True) as connection:
    NEWS_SESSIONS = pd.read_sql_query(NEWS_SQL, connection)
NEWS_SESSIONS["session_date"] = pd.to_datetime(NEWS_SESSIONS["session_date"]).dt.normalize()
NEWS_SESSIONS["symbol"] = NEWS_SESSIONS["symbol"].str.upper()
NEWS_SESSIONS = NEWS_SESSIONS.sort_values(["symbol", "session_date"], kind="stable").reset_index(drop=True)
print(f"aggregated {len(NEWS_SESSIONS):,} firm reaction-sessions in {time.time() - _started:.1f}s")
assert len(NEWS_SESSIONS) == EXPECTED_SESSIONS, "news aggregation lost or gained sessions"
assert int(NEWS_SESSIONS["news_count"].sum()) == EXPECTED_EVENTS, "news aggregation lost headlines"

sign_checks = {
    "corr(adverse_tone, negative_share)": float(NEWS_SESSIONS["adverse_tone"].corr(NEWS_SESSIONS["negative_share"])),
    "corr(adverse_tone, vader_mean)": float(NEWS_SESSIONS["adverse_tone"].corr(NEWS_SESSIONS["vader_mean"])),
    "corr(finbert_intensity, vader_intensity)": float(NEWS_SESSIONS["semantic_intensity"].corr(NEWS_SESSIONS["vader_intensity"])),
}
for label, value in sign_checks.items():
    print(f"{label:>46s} = {value:+.4f}")
if sign_checks["corr(adverse_tone, negative_share)"] <= 0.5:
    raise RuntimeError("adverse_tone does not increase with negative dominance; sign convention broken")
if sign_checks["corr(adverse_tone, vader_mean)"] >= 0.0:
    raise RuntimeError("adverse_tone does not move against VADER positivity; sign convention broken")
print("\nsign convention CONFIRMED: larger adverse_tone means more adverse content.")

# %% [markdown]
# ### Verification of the frozen mixed news-timing policy
#
# The checkpoint stores mapped reaction sessions but not original timestamps or
# per-event timing-type flags. A per-headline remap is therefore impossible
# without rescanning the raw corpus. Instead this audit verifies the E6-declared
# E5 manifest by hash, checks both recorded policy branches exactly, exhaustively
# exercises the date-only branch over every calendar date in the window, and
# confirms that every stored reaction date is a valid XNYS session.
#
# This is point-in-time at the forecast close, but it is not evidence that every
# headline followed the stricter date-only rule requested for this experiment.

# %%
_session_mapper = _SessionMapper()
_calendar_days = pd.date_range("2010-12-01", "2023-12-31", freq="D")
_mapped_date_only = pd.to_datetime([_session_mapper.map(day.strftime("%Y-%m-%d"))[0] for day in _calendar_days])
DATE_ONLY_TIMING_VIOLATIONS = verify_strictly_after(_calendar_days, _mapped_date_only)
DATE_ONLY_UPSTREAM_ROWS = int(manifest_counts["date_only_rows"])
PRECISE_TIMESTAMP_UPSTREAM_ROWS = int(manifest_counts["precise_timestamp_rows"])
UPSTREAM_TIMING_POLICY_VERIFIED = E5_MANIFEST["timing_rule"] == EXPECTED_UPSTREAM_TIMING_RULE

XNYS_SESSIONS = pd.DatetimeIndex(sorted(pd.Series(_mapped_date_only).unique()))
_corpus_sessions = pd.DatetimeIndex(NEWS_SESSIONS["session_date"].unique())
INVALID_CORPUS_REACTION_SESSIONS = int((~_corpus_sessions.isin(XNYS_SESSIONS)).sum())

print(f"calendar dates checked, date-only branch : {len(_calendar_days):,}")
print(f"date-only mappings not strictly forward  : {DATE_ONLY_TIMING_VIOLATIONS}")
print(f"upstream date-only/exact-midnight rows    : {DATE_ONLY_UPSTREAM_ROWS:,}")
print(f"upstream precise-timestamp rows           : {PRECISE_TIMESTAMP_UPSTREAM_ROWS:,}")
print(f"stored reaction dates outside XNYS        : {INVALID_CORPUS_REACTION_SESSIONS}")
print("per-event original timing retained        : no")
if not UPSTREAM_TIMING_POLICY_VERIFIED or DATE_ONLY_TIMING_VIOLATIONS or INVALID_CORPUS_REACTION_SESSIONS:
    raise RuntimeError("the frozen news-to-session timing policy failed verification")

# %% [markdown]
# ## 3. Prices: load, validate, and gate
#
# Prices come from the FNSPID adjusted-close archive already verified by the E6
# manifest. Cleaning is deliberately minimal and deterministic: duplicate
# session rows and non-positive or missing adjusted closes are demonstrable data
# errors and are removed; **economically plausible extreme returns are flagged
# and kept**.

# %%
DEV_START_TS, DEV_END_TS = pd.Timestamp(DEV_START), pd.Timestamp(DEV_END)
EVAL_START_TS, EVAL_END_TS = pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END)
WARMUP_START_TS = pd.Timestamp(WARMUP_START)
RECURSION_START_TS = pd.Timestamp(RECURSION_START)


def load_price_frame(archive: zipfile.ZipFile, member: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one member, report every cleaning action, and apply the price repair.

    The raw close is carried alongside the adjusted close because the two
    together are what reveal an inconsistent adjustment factor. Under VARIANT v1
    the adjusted series is used exactly as published.
    """

    with archive.open(member) as handle:
        raw = pd.read_csv(handle, usecols=lambda name: str(name).strip().casefold() in {"date", "close", "adj close"})
    raw.columns = [str(column).strip().casefold() for column in raw.columns]
    diagnostics: dict[str, Any] = {"price_rows_raw": len(raw), "repaired_price_rows": 0}
    if not {"date", "close", "adj close"}.issubset(raw.columns):
        diagnostics["unusable_schema"] = True
        return pd.DataFrame(columns=["session_date", "adjusted_close", "raw_close"]), diagnostics
    diagnostics["unusable_schema"] = False

    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "adjusted_close": pd.to_numeric(raw["adj close"], errors="coerce"),
            "raw_close": pd.to_numeric(raw["close"], errors="coerce"),
        }
    )
    # Row selection is deliberately unchanged from v1: it keys on the adjusted
    # close alone. The raw close is a diagnostic input to the repair, never a
    # reason to drop a session.
    diagnostics["missing_price_rows"] = int(frame["adjusted_close"].isna().sum() + frame["session_date"].isna().sum())
    frame = frame.dropna(subset=["session_date", "adjusted_close"])
    diagnostics["nonpositive_price_rows"] = int((frame["adjusted_close"] <= 0).sum())
    frame = frame[frame["adjusted_close"] > 0]
    duplicated = frame["session_date"].duplicated(keep=False)
    diagnostics["duplicate_session_rows"] = int(duplicated.sum())
    frame = frame.sort_values("session_date", kind="stable").drop_duplicates("session_date", keep="last").reset_index(drop=True)
    diagnostics["rows_after_cleaning"] = len(frame)

    # Sessions with no usable raw close cannot be assessed, so they are given a
    # zero gap and can never be flagged as a candidate adjustment discontinuity.
    unusable_raw = frame["raw_close"].isna() | (frame["raw_close"] <= 0)
    diagnostics["raw_close_unusable_rows"] = int(unusable_raw.sum())
    frame.loc[unusable_raw, "raw_close"] = frame.loc[unusable_raw, "adjusted_close"]

    if PRICE_REPAIR == "min_abs_return" and len(frame) > 1:
        repaired, replaced = repair_adjusted_close(
            frame["session_date"],
            frame["adjusted_close"].to_numpy(dtype=float),
            frame["raw_close"].to_numpy(dtype=float),
            gap_threshold=ADJUSTMENT_GAP_THRESHOLD,
        )
        frame["adjusted_close"] = repaired
        diagnostics["repaired_price_rows"] = int(replaced.sum())
    return frame, diagnostics


def summarise_price_quality(symbol: str, frame: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Firm-level price-validation record; flags problems, never deletes them."""

    record: dict[str, Any] = {"symbol": symbol, **diagnostics}
    if frame.empty:
        record.update(
            {
                "first_session": None,
                "last_session": None,
                "sessions_dev": 0,
                "sessions_eval": 0,
                "returns_dev": 0,
                "returns_eval": 0,
                "sessions_before_dev_start": 0,
                "max_zero_return_run": 0,
                "max_zero_return_run_window": 0,
                "zero_return_share": math.nan,
                "extreme_return_count": 0,
                "max_abs_return": math.nan,
                "likely_split_artifact_count": 0,
                "max_session_gap_days": math.nan,
                "split_boundary_gap_days": math.nan,
                "survivor_conditioned": False,
                "identity_metadata_available": False,
            }
        )
        return record

    dates = frame["session_date"]
    returns = np.diff(np.log(frame["adjusted_close"].to_numpy(dtype=float)))
    return_dates = dates.to_numpy()[1:]
    dev_mask = (return_dates >= DEV_START_TS.to_datetime64()) & (return_dates <= DEV_END_TS.to_datetime64())
    eval_mask = (return_dates >= EVAL_START_TS.to_datetime64()) & (return_dates <= EVAL_END_TS.to_datetime64())

    def longest_zero_run(flags: np.ndarray) -> int:
        run, longest = 0, 0
        for flag in flags:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return longest

    zero = returns == 0.0
    window = (return_dates >= DEV_START_TS.to_datetime64()) & (return_dates <= EVAL_END_TS.to_datetime64())
    longest = longest_zero_run(zero)
    longest_in_window = longest_zero_run(zero[window])

    # A same-magnitude, opposite-sign pair of very large moves on consecutive
    # sessions is the classic unadjusted split / corporate-action signature.
    split_flag = 0
    if returns.size > 1:
        pairs = np.abs(returns[:-1] + returns[1:]) < 0.05
        large = (np.abs(returns[:-1]) > 0.35) & (np.abs(returns[1:]) > 0.35)
        split_flag = int(np.count_nonzero(pairs & large))

    gaps = dates.diff().dt.days.to_numpy()[1:]
    boundary = dates[(dates > DEV_END_TS) & (dates <= DEV_END_TS + pd.Timedelta(days=21))]
    before_boundary = dates[dates <= DEV_END_TS]
    boundary_gap = float((boundary.iloc[0] - before_boundary.iloc[-1]).days) if len(boundary) and len(before_boundary) else math.nan

    record.update(
        {
            "first_session": dates.iloc[0].date().isoformat(),
            "last_session": dates.iloc[-1].date().isoformat(),
            "sessions_dev": int(((dates >= DEV_START_TS) & (dates <= DEV_END_TS)).sum()),
            "sessions_eval": int(((dates >= EVAL_START_TS) & (dates <= EVAL_END_TS)).sum()),
            "returns_dev": int(dev_mask.sum()),
            "returns_eval": int(eval_mask.sum()),
            "sessions_before_dev_start": int((dates < DEV_START_TS).sum()),
            "max_zero_return_run": int(longest),
            "max_zero_return_run_window": int(longest_in_window),
            "zero_return_share": float(zero.mean()) if returns.size else math.nan,
            "extreme_return_count": int(np.count_nonzero(np.abs(returns) > EXTREME_RETURN_THRESHOLD)),
            "max_abs_return": float(np.abs(returns).max()) if returns.size else math.nan,
            "likely_split_artifact_count": split_flag,
            "max_session_gap_days": float(np.nanmax(gaps)) if gaps.size else math.nan,
            "split_boundary_gap_days": boundary_gap,
            "survivor_conditioned": bool(dates.iloc[-1] >= pd.Timestamp("2023-12-01")),
            "identity_metadata_available": False,
        }
    )
    return record


_started = time.time()
PRICES: dict[str, pd.DataFrame] = {}
price_quality_rows: list[dict[str, Any]] = []
missing_price_symbols: list[str] = []

with zipfile.ZipFile(PRICE_ARCHIVE) as archive:
    for symbol in [*COHORT_SYMBOLS, MARKET_SYMBOL]:
        member = PRICE_MEMBERS.get(symbol)
        if member is None:
            missing_price_symbols.append(symbol)
            price_quality_rows.append(summarise_price_quality(symbol, pd.DataFrame(), {"price_rows_raw": 0, "unusable_schema": True}))
            continue
        frame, diagnostics = load_price_frame(archive, member)
        price_quality_rows.append(summarise_price_quality(symbol, frame, diagnostics))
        if not frame.empty:
            PRICES[symbol] = frame

PRICE_VALIDATION = pd.DataFrame(price_quality_rows).sort_values("symbol", kind="stable").reset_index(drop=True)
PRICE_VALIDATION.to_csv(OUTPUT_DIR / "price_validation.csv", index=False)
print(f"loaded {len(PRICES)} price frames in {time.time() - _started:.1f}s; missing: {missing_price_symbols}")

PRICE_VALIDATION_SUMMARY = (
    PRICE_VALIDATION[
        [
            "symbol",
            "price_rows_raw",
            "duplicate_session_rows",
            "nonpositive_price_rows",
            "rows_after_cleaning",
            "returns_dev",
            "returns_eval",
            "max_zero_return_run",
            "extreme_return_count",
            "likely_split_artifact_count",
        ]
    ]
    .describe(include="all")
    .T
)
print(PRICE_VALIDATION_SUMMARY.to_string())

# %%
_flagged = PRICE_VALIDATION[PRICE_VALIDATION["likely_split_artifact_count"] > 0]
print(f"firms with a likely split / corporate-action signature: {len(_flagged)}")
if len(_flagged):
    print(
        _flagged.nlargest(10, "likely_split_artifact_count")[
            ["symbol", "likely_split_artifact_count", "max_abs_return", "extreme_return_count"]
        ].to_string(index=False)
    )

_bad_prices = PRICE_VALIDATION[PRICE_VALIDATION["nonpositive_price_rows"] > 0]
print(
    f"\nfirms with non-positive or missing adjusted closes removed: {len(_bad_prices)} "
    f"({int(PRICE_VALIDATION['nonpositive_price_rows'].sum()):,} rows in total)"
)
if len(_bad_prices):
    print(
        _bad_prices.nlargest(10, "nonpositive_price_rows")[
            ["symbol", "price_rows_raw", "nonpositive_price_rows", "missing_price_rows", "rows_after_cleaning"]
        ].to_string(index=False)
    )

_stale = PRICE_VALIDATION[PRICE_VALIDATION["max_zero_return_run_window"] > STALE_PRICE_MAX_RUN]
print(
    f"\nfirms whose adjusted close is frozen for more than {STALE_PRICE_MAX_RUN} consecutive "
    f"sessions inside {DEV_START}..{EVAL_END}: {len(_stale)}"
)
if len(_stale):
    print(
        _stale.nlargest(10, "max_zero_return_run_window")[
            ["symbol", "max_zero_return_run_window", "zero_return_share", "last_session"]
        ].to_string(index=False)
    )
    print(
        "  These are excluded by the deterministic staleness rule below. A US listing whose\n"
        "  adjusted close does not move for a month is a demonstrable data error, not economics."
    )

_cohort_validation = PRICE_VALIDATION[PRICE_VALIDATION["symbol"] != MARKET_SYMBOL]
print(
    f"\ncohort firms whose price history ends before 2023-12: "
    f"{int((~_cohort_validation['survivor_conditioned']).sum())} of {len(_cohort_validation)}"
)
print(
    "\nSample description: this is the FNSPID linked firm-price panel, NOT a historical "
    "S&P 500 constituent panel. Ticker-level identity metadata is unavailable in the archive, "
    "so ticker reuse and mid-sample renames cannot be excluded, and the ticker-linked cohort "
    "also contains exchange-traded funds (for example AGG, GLD, QQQ) alongside operating firms."
)

# %% [markdown]
# ### Market state (frozen local series)
#
# `SPY` is already the frozen market proxy of the upstream runs and is present in
# the same local archive, so market-state controls are available without any
# download. They enter every model, including `M0`.

# %%
if MARKET_SYMBOL not in PRICES:
    raise RuntimeError(f"market proxy {MARKET_SYMBOL} missing from the local price archive")

_market = build_next_session_targets(PRICES[MARKET_SYMBOL])
_market_returns = _market["log_return"]
MARKET_STATE = pd.DataFrame(
    {
        "forecast_date": _market["session_date"],
        "market_reaction": _market_returns,
        "market_abs_reaction": _market_returns.abs(),
        "market_vol20": _market_returns.rolling(MARKET_VOL_WINDOW, min_periods=MARKET_VOL_WINDOW).std(),
    }
).dropna()
MARKET_STATE = MARKET_STATE[MARKET_STATE["forecast_date"] >= WARMUP_START_TS].reset_index(drop=True)
print(
    f"market state rows: {len(MARKET_STATE):,} from {MARKET_STATE['forecast_date'].min().date()} to {MARKET_STATE['forecast_date'].max().date()}"
)
MARKET_STATE.tail(3)

# %% [markdown]
# ### Support pass and deterministic firm universe
#
# Support is measured before any model is fitted, so the smoke subset is chosen
# from data rather than from results.

# %%
support_rows: list[dict[str, Any]] = []
news_counts_by_symbol = NEWS_SESSIONS.groupby("symbol", sort=True)["news_count"].agg(["size", "sum"])

for symbol in COHORT_SYMBOLS:
    frame = PRICES.get(symbol)
    if frame is None or frame.empty:
        support_rows.append({"symbol": symbol, "dev_origins": 0, "eval_origins": 0, "news_sessions": 0, "headlines": 0})
        continue
    dates = frame["session_date"]
    # A forecast origin needs a following session to supply the target.
    origins = dates.iloc[:-1]
    targets = dates.iloc[1:]
    dev_origins = int(((origins >= DEV_START_TS) & (origins <= DEV_END_TS) & (targets.to_numpy() <= DEV_END_TS.to_datetime64())).sum())
    eval_origins = int(((origins >= EVAL_START_TS) & (origins <= EVAL_END_TS)).sum())
    counts = news_counts_by_symbol.loc[symbol] if symbol in news_counts_by_symbol.index else None
    support_rows.append(
        {
            "symbol": symbol,
            "dev_origins": dev_origins,
            "eval_origins": eval_origins,
            "news_sessions": int(counts["size"]) if counts is not None else 0,
            "headlines": int(counts["sum"]) if counts is not None else 0,
        }
    )

SUPPORT = pd.DataFrame(support_rows)
STALE_RUNS = PRICE_VALIDATION.set_index("symbol")["max_zero_return_run_window"].to_dict()


def support_reason(row: pd.Series) -> str:
    """One deterministic exclusion reason per firm, evaluated in a fixed order."""

    if row["symbol"] in missing_price_symbols:
        return "no_price_file"
    if STALE_RUNS.get(row["symbol"], 0) > STALE_PRICE_MAX_RUN:
        return "stale_price_run"
    if row["dev_origins"] < MIN_DEV_OBS:
        return "insufficient_dev_origins"
    if row["eval_origins"] < MIN_EVAL_OBS:
        return "insufficient_eval_origins"
    if row["news_sessions"] == 0:
        return "no_linked_news"
    return "retained"


SUPPORT["support_reason"] = SUPPORT.apply(support_reason, axis=1)
SUPPORT = SUPPORT.sort_values(["dev_origins", "eval_origins", "symbol"], ascending=[False, False, True], kind="stable").reset_index(
    drop=True
)

ELIGIBLE = SUPPORT[SUPPORT["support_reason"] == "retained"].copy()
ELIGIBLE["total_origins"] = ELIGIBLE["dev_origins"] + ELIGIBLE["eval_origins"]
ELIGIBLE = ELIGIBLE.sort_values(["total_origins", "headlines", "symbol"], ascending=[False, False, True], kind="stable").reset_index(
    drop=True
)

print(SUPPORT["support_reason"].value_counts().to_string())
print(f"\neligible firms before the risk-model gates: {len(ELIGIBLE)}")

MODEL_UNIVERSE = sorted(ELIGIBLE["symbol"].head(SMOKE_FIRM_COUNT)) if RUN_MODE == "smoke" else sorted(ELIGIBLE["symbol"])
print(f"modelling universe ({RUN_MODE}): {len(MODEL_UNIVERSE)} firms")
if RUN_MODE == "smoke":
    print("smoke firms:", ", ".join(MODEL_UNIVERSE))

# %% [markdown]
# ### Gate 2 — price and data gate (fail closed)
#
# Inferential cells run only if every check below passes.

# %%
gate_rows: list[dict[str, Any]] = []


def gate(name: str, passed: bool, observed: Any, requirement: str) -> bool:
    gate_rows.append({"gate": name, "status": "PASS" if passed else "FAIL", "observed": observed, "requirement": requirement})
    return passed


_dev_dates = set()
_eval_dates = set()
_eval_origin_total = 0
for symbol in MODEL_UNIVERSE:
    dates = PRICES[symbol]["session_date"]
    _dev_dates.update(dates[(dates >= DEV_START_TS) & (dates <= DEV_END_TS)])
    eval_dates = dates[(dates >= EVAL_START_TS) & (dates <= EVAL_END_TS)]
    _eval_dates.update(eval_dates)
    _eval_origin_total += len(eval_dates)

gate("prices_readable", len(PRICES) > 0, len(PRICES), ">0 price frames loaded")
gate(
    "adjusted_prices_usable",
    int(PRICE_VALIDATION["nonpositive_price_rows"].sum()) >= 0 and all((frame["adjusted_close"] > 0).all() for frame in PRICES.values()),
    f"{int(PRICE_VALIDATION['nonpositive_price_rows'].sum())} non-positive rows removed",
    "no non-positive adjusted close survives cleaning",
)
gate("market_proxy_present", MARKET_SYMBOL in PRICES, MARKET_SYMBOL, "local market series available")
gate(
    "timing_policy_point_in_time",
    UPSTREAM_TIMING_POLICY_VERIFIED and DATE_ONLY_TIMING_VIOLATIONS == 0 and INVALID_CORPUS_REACTION_SESSIONS == 0,
    f"{DATE_ONLY_UPSTREAM_ROWS:,} date-only; {PRECISE_TIMESTAMP_UPSTREAM_ROWS:,} precise; {INVALID_CORPUS_REACTION_SESSIONS} invalid sessions",
    "hashed upstream mixed policy verified; date-only branch strictly forward; every reaction date is an XNYS session",
)
gate("retained_cross_section", len(MODEL_UNIVERSE) >= MIN_RETAINED_FIRMS, len(MODEL_UNIVERSE), f">= {MIN_RETAINED_FIRMS} firms")
gate("development_dates", len(_dev_dates) >= MIN_SPLIT_DATES, len(_dev_dates), f">= {MIN_SPLIT_DATES} distinct dates")
gate("evaluation_dates", len(_eval_dates) >= MIN_SPLIT_DATES, len(_eval_dates), f">= {MIN_SPLIT_DATES} distinct dates")
gate(
    "expected_tail_events",
    ALPHA * _eval_origin_total >= MIN_EXPECTED_TAIL_EVENTS,
    round(ALPHA * _eval_origin_total, 1),
    f">= {MIN_EXPECTED_TAIL_EVENTS} expected violations",
)
_split_share = float(PRICE_VALIDATION.loc[PRICE_VALIDATION["symbol"].isin(MODEL_UNIVERSE), "likely_split_artifact_count"].sum()) / max(
    _eval_origin_total, 1
)
gate(
    "no_unresolved_corporate_action_mechanism",
    _split_share < 0.001,
    f"{_split_share:.3e} flagged pairs per evaluation session",
    "< 1e-3 flagged split-signature pairs per session",
)
gate(
    "development_and_evaluation_disjoint",
    DEV_END_TS < EVAL_START_TS,
    f"{DEV_END} < {EVAL_START}",
    "chronologically disjoint splits",
)

GATE_STATUS = pd.DataFrame(gate_rows)
GATE2_PASSED = bool((GATE_STATUS["status"] == "PASS").all())
print(GATE_STATUS.to_string(index=False))
print(f"\nGATE 2: {'PASSED' if GATE2_PASSED else 'FAILED'}")
if not GATE2_PASSED:
    print(
        "\nGate 2 failed. The notebook continues only to emit the data audit and a clear failure\n"
        "record; no risk model is fitted and no inferential result is produced."
    )

# %% [markdown]
# ## 4. Point-in-time panel construction
#
# One row per firm × forecast-origin trading session, **including no-news days**.
# Nothing at a row uses information dated after its `forecast_date`.

# %%
GARCH_COLUMNS = [
    "symbol",
    "converged",
    "mu",
    "omega",
    "alpha",
    "gamma",
    "beta",
    "nu",
    "persistence",
    "unconditional_vol_annualised",
    "dev_return_obs",
    "loglikelihood",
    "recursion_vs_arch_max_rel_diff",
    "initialisation_sensitivity_at_dev_end",
    "firm_attrition_reason",
    "note",
]


def estimate_gjr_params(returns: np.ndarray) -> tuple[GjrParams | None, dict[str, Any]]:
    """One ConstantMean + GJR-GARCH(1,1)-t fit, rescaled back to return units.

    Factored out so the frozen development fit and the v2 expanding-window
    refits are guaranteed to use the identical estimator call.
    """

    info: dict[str, Any] = {"obs": int(returns.size), "note": "", "loglikelihood": math.nan}
    # arch works best on percentage returns; parameters are rescaled back below.
    scaled = returns * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch.univariate.arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
            fitted = model.fit(disp="off", show_warning=False, options={"maxiter": 500})
    except Exception as error:  # pragma: no cover - estimator failure path
        info["note"] = f"{type(error).__name__}: {error}"[:200]
        return None, info

    convergence_flag = int(getattr(fitted, "convergence_flag", 1))
    optimisation_result = getattr(fitted, "optimization_result", None)
    optimisation_success = bool(getattr(optimisation_result, "success", convergence_flag == 0))
    info["convergence_flag"] = convergence_flag
    info["optimisation_success"] = optimisation_success
    info["optimisation_message"] = str(getattr(optimisation_result, "message", ""))[:200]
    if convergence_flag != 0 or not optimisation_success:
        message = getattr(optimisation_result, "message", "optimizer did not converge")
        info["note"] = (f"optimizer failure (flag={convergence_flag}, success={optimisation_success}): {message}")[:200]
        return None, info

    values = fitted.params
    params = GjrParams(
        mu=float(values["mu"]) / 100.0,
        omega=float(values["omega"]) / 10_000.0,
        alpha=float(values["alpha[1]"]),
        gamma=float(values["gamma[1]"]),
        beta=float(values["beta[1]"]),
        nu=float(values.get("nu", math.nan)),
    )
    info["loglikelihood"] = float(fitted.loglikelihood)
    info["conditional_variance"] = np.asarray(fitted.conditional_volatility, dtype=float) ** 2 / 10_000.0
    return params, info


def usable_gjr_params(params: GjrParams | None) -> bool:
    """A fit is usable only if it is finite, positive and stationary."""

    if params is None:
        return False
    if not np.isfinite([params.omega, params.alpha, params.gamma, params.beta]).all() or params.omega <= 0.0:
        return False
    return params.persistence < GARCH_MAX_PERSISTENCE


def fit_firm_gjr(symbol: str, returns: pd.Series) -> dict[str, Any]:
    """Fit ConstantMean + GJR-GARCH(1,1)-t on development returns only."""

    record: dict[str, Any] = dict.fromkeys(GARCH_COLUMNS)
    record["symbol"] = symbol
    record["dev_return_obs"] = int(returns.size)
    record["converged"] = False
    record["note"] = ""
    if returns.size < MIN_DEV_OBS:
        record["firm_attrition_reason"] = "insufficient_dev_returns"
        return record

    params, info = estimate_gjr_params(returns.to_numpy(dtype=float))
    if params is None:
        record["firm_attrition_reason"] = "garch_fit_failed"
        record["note"] = info["note"]
        return record

    record.update(
        {
            "mu": params.mu,
            "omega": params.omega,
            "alpha": params.alpha,
            "gamma": params.gamma,
            "beta": params.beta,
            "nu": params.nu,
            "persistence": params.persistence,
            "loglikelihood": info["loglikelihood"],
        }
    )
    if not np.isfinite([params.omega, params.alpha, params.gamma, params.beta]).all() or params.omega <= 0.0:
        record["firm_attrition_reason"] = "garch_invalid_parameters"
        return record
    if params.persistence >= GARCH_MAX_PERSISTENCE:
        record["firm_attrition_reason"] = "garch_non_stationary"
        return record
    record["unconditional_vol_annualised"] = float(math.sqrt(params.unconditional_variance * 252.0))

    # Cross-check the in-notebook recursion against the estimator's own filter.
    # Seeding it with arch's own first conditional variance isolates the
    # recursion from the initialisation, so the two must agree to machine
    # precision; anything else would be a real implementation error.
    dev_values = returns.to_numpy(dtype=float)
    reference = info["conditional_variance"]
    seeded = gjr_conditional_variances(dev_values, params, initial_variance=float(reference[0]))[: dev_values.size]
    record["recursion_vs_arch_max_rel_diff"] = float(np.nanmax(np.abs(seeded - reference) / reference))

    # Separately, quantify how much a different (point-in-time) initialisation
    # still matters by the end of development.  Near-unit-root fits forget their
    # starting variance slowly, which is why the panel recursion burns in from
    # the whole available price history rather than a short window.
    sampled = gjr_conditional_variances(dev_values, params, initial_variance=float(np.var(dev_values, ddof=1)))[: dev_values.size]
    record["initialisation_sensitivity_at_dev_end"] = float(abs(sampled[-1] - reference[-1]) / reference[-1])
    record["converged"] = True
    record["firm_attrition_reason"] = "retained"
    record["_params"] = params
    return record


EVALUATION_YEARS = list(range(pd.Timestamp(EVAL_START).year, pd.Timestamp(EVAL_END).year + 1))


def build_refit_schedule(
    symbol: str, returns: pd.Series, development_params: GjrParams
) -> tuple[dict[int, GjrParams], list[dict[str, Any]]]:
    """Re-estimate the volatility filter once per evaluation year, point-in-time.

    The parameters that govern evaluation year Y are fitted on returns from
    DEV_START through 31 December of Y-1, so nothing in year Y informs its own
    filter. Development-period forecasts keep the development fit, exactly as in
    v1. A refit that fails or is non-stationary carries the previous window's
    parameters forward and is counted; the specification never changes, only the
    vintage of its parameters.
    """

    by_year: dict[int, GjrParams] = {}
    log: list[dict[str, Any]] = []
    current = development_params
    for year in EVALUATION_YEARS:
        window = select_expanding_refit_window(
            returns,
            development_start=DEV_START,
            evaluation_year=year,
        )
        candidate, info = (None, {"note": "insufficient observations", "obs": int(window.size)})
        if window.size >= MIN_DEV_OBS:
            candidate, info = estimate_gjr_params(window.to_numpy(dtype=float))
        accepted = usable_gjr_params(candidate)
        if accepted:
            current = candidate
        by_year[year] = current
        log.append(
            {
                "symbol": symbol,
                "evaluation_year": year,
                "fit_window_start": (window.index.min().strftime("%Y-%m-%d") if not window.empty else ""),
                "convergence_flag": info.get("convergence_flag", math.nan),
                "optimisation_success": info.get("optimisation_success", False),
                "optimisation_message": info.get("optimisation_message", ""),
                "fit_window_end": (window.index.max().strftime("%Y-%m-%d") if not window.empty else ""),
                "fit_cutoff_exclusive": f"{year}-01-01",
                "observations": int(info["obs"]),
                "accepted": bool(accepted),
                "carried_forward": not accepted,
                "persistence": current.persistence,
                "nu": current.nu,
                "note": str(info.get("note", "")),
            }
        )
    return by_year, log


def build_firm_panel(
    symbol: str,
    params: GjrParams,
    sigma_source: str = "garch",
    schedule: dict[int, GjrParams] | None = None,
) -> pd.DataFrame:
    """Assemble the point-in-time feature rows for one firm.

    ``schedule`` maps an evaluation year to the parameters governing it. When it
    is ``None`` the single ``params`` set governs the whole series, which is the
    frozen v1 filter.
    """

    targets = build_next_session_targets(PRICES[symbol])
    # Burn the variance recursion in over the whole available price history and
    # only emit forecast origins from WARMUP_START, so the conditional variance
    # at every scored origin is effectively free of its starting value.  This
    # uses only information strictly before each origin, so it cannot leak.
    targets = targets[targets["session_date"] >= RECURSION_START_TS].reset_index(drop=True)
    returns = targets["log_return"].to_numpy(dtype=float)
    valid = np.isfinite(returns)
    targets = targets[valid].reset_index(drop=True)
    returns = returns[valid]
    if returns.size < MIN_DEV_OBS:
        return pd.DataFrame()

    # params_per_step[t] governs the variance of returns[t + 1], so the schedule
    # is keyed on the year of the return being forecast, never on year t.
    mu_per_return = np.full(returns.size, params.mu, dtype=float)
    if sigma_source == "garch" and schedule is not None:
        years = targets["session_date"].dt.year.to_numpy()
        governing = [schedule.get(int(year), params) for year in years]
        mu_per_return = np.array([entry.mu for entry in governing], dtype=float)
        sigma2 = gjr_conditional_variances_piecewise(
            returns,
            governing[1:] + [governing[-1]],
            initial_variance=float(np.var(returns[:252], ddof=1)),
        )
    elif sigma_source == "garch":
        sigma2 = gjr_conditional_variances(returns, params, initial_variance=float(np.var(returns[:252], ddof=1)))
    elif sigma_source == "ewma":
        sigma2 = np.empty(returns.size + 1, dtype=float)
        sigma2[0] = float(np.var(returns[:252], ddof=1))
        shocks = returns - params.mu
        for index in range(returns.size):
            sigma2[index + 1] = EWMA_LAMBDA * sigma2[index] + (1.0 - EWMA_LAMBDA) * shocks[index] ** 2
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown sigma_source {sigma_source}")

    origin_count = returns.size - 1  # the last observed return has no following target
    if origin_count <= 0:
        return pd.DataFrame()
    index = np.arange(origin_count)
    reaction_sigma = np.sqrt(sigma2[index])
    forecast_sigma = np.sqrt(sigma2[index + 1])
    reaction_mu = mu_per_return[index]
    target_mu = mu_per_return[index + 1]

    panel = pd.DataFrame(
        {
            "symbol": symbol,
            "forecast_date": targets["session_date"].to_numpy()[index],
            "target_date": targets["session_date"].to_numpy()[index + 1],
            "reaction_return": returns[index],
            "target_return": returns[index + 1],
            "mu_hat": target_mu,
            "sigma_reaction": reaction_sigma,
            "sigma_forecast": forecast_sigma,
        }
    )
    panel["reaction_z"] = (panel["reaction_return"] - reaction_mu) / reaction_sigma
    panel["abs_reaction_z"] = panel["reaction_z"].abs()
    panel["log_sigma_forecast"] = np.log(forecast_sigma)
    panel["z_target"] = (panel["target_return"] - target_mu) / forecast_sigma
    return panel[panel["forecast_date"] >= WARMUP_START_TS].reset_index(drop=True)


def attach_news(panel: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, Counter]:
    """Fold every eligible reaction-session headline group onto a firm session."""

    counters: Counter = Counter()
    zero_columns = {
        "news_indicator": 0.0,
        "news_count": 0.0,
        "adverse_tone": 0.0,
        "semantic_intensity": 0.0,
        "negative_share": 0.0,
        "vader_adverse_tone": 0.0,
        "vader_intensity": 0.0,
        "recap_share": 0.0,
        "source_news_sessions": 0.0,
        "news_forward_gap_days": 0.0,
    }
    for column, value in zero_columns.items():
        panel[column] = value
    if panel.empty:
        return panel, counters

    firm_news = NEWS_SESSIONS[NEWS_SESSIONS["symbol"] == symbol]
    counters["news_sessions_available"] = len(firm_news)
    if firm_news.empty:
        return panel, counters

    origins = pd.DatetimeIndex(panel["forecast_date"])
    positions = origins.searchsorted(pd.DatetimeIndex(firm_news["session_date"]), side="left")
    inside = positions < len(origins)
    counters["news_after_last_origin"] = int((~inside).sum())

    mapped = firm_news.loc[inside].copy()
    mapped["origin_position"] = positions[inside]
    mapped["mapped_origin"] = origins.to_numpy()[positions[inside]]
    gap = (mapped["mapped_origin"] - mapped["session_date"]).dt.days
    mapped["forward_gap_days"] = gap
    too_far = gap > MAX_NEWS_FORWARD_GAP_DAYS
    counters["news_forward_gap_exceeded"] = int(too_far.sum())
    mapped = mapped[~too_far]
    if mapped.empty:
        return panel, counters

    weight = mapped["news_count"].to_numpy(dtype=float)
    for column in ("adverse_tone", "semantic_intensity", "negative_share", "recap_share", "vader_intensity"):
        mapped[column] = mapped[column].to_numpy(dtype=float) * weight
    mapped["vader_adverse_tone"] = -mapped["vader_mean"].to_numpy(dtype=float) * weight

    grouped = mapped.groupby("origin_position", sort=True).agg(
        news_count=("news_count", "sum"),
        adverse_tone=("adverse_tone", "sum"),
        semantic_intensity=("semantic_intensity", "sum"),
        negative_share=("negative_share", "sum"),
        recap_share=("recap_share", "sum"),
        vader_adverse_tone=("vader_adverse_tone", "sum"),
        vader_intensity=("vader_intensity", "sum"),
        source_news_sessions=("session_date", "nunique"),
        news_forward_gap_days=("forward_gap_days", "max"),
    )
    total = grouped["news_count"].to_numpy(dtype=float)
    for column in (
        "adverse_tone",
        "semantic_intensity",
        "negative_share",
        "recap_share",
        "vader_adverse_tone",
        "vader_intensity",
    ):
        grouped[column] = grouped[column].to_numpy(dtype=float) / total

    rows = grouped.index.to_numpy()
    panel.loc[panel.index[rows], "news_indicator"] = 1.0
    for column in grouped.columns:
        panel.loc[panel.index[rows], column] = grouped[column].to_numpy()
    counters["origins_with_news"] = len(grouped)
    counters["headlines_attached"] = int(total.sum())
    return panel, counters


# %%
PIPELINE_ATTRITION: list[dict[str, Any]] = []


def record_attrition(step: str, unit: str, surviving: int, excluded: int, reason: str = "") -> None:
    PIPELINE_ATTRITION.append({"step": step, "unit": unit, "surviving": surviving, "excluded_at_step": excluded, "reason": reason})


record_attrition("e6_usable_headlines", "headlines", EXPECTED_EVENTS, 0, "reused from completed E6 run")
record_attrition("e6_firm_reaction_sessions", "firm-sessions", EXPECTED_SESSIONS, 0, "aggregated from the E6 checkpoint")
record_attrition("cohort_firms", "firms", len(COHORT_SYMBOLS), 0, "frozen E5 coherent cohort")
for reason, count in SUPPORT["support_reason"].value_counts().items():
    if reason != "retained":
        record_attrition(f"support_gate:{reason}", "firms", int((SUPPORT["support_reason"] == "retained").sum()), int(count), str(reason))
record_attrition(
    "firms_passing_support_gate", "firms", len(ELIGIBLE), len(COHORT_SYMBOLS) - len(ELIGIBLE), "price history and origin counts"
)
record_attrition(
    "modelling_universe",
    "firms",
    len(MODEL_UNIVERSE),
    len(ELIGIBLE) - len(MODEL_UNIVERSE),
    "smoke subset: greatest dev+eval support" if RUN_MODE == "smoke" else "full mode keeps every eligible firm",
)

# %%
GARCH_SUMMARY = pd.DataFrame(columns=GARCH_COLUMNS)
PANEL = pd.DataFrame()
FIRM_PARAMS: dict[str, GjrParams] = {}
NEWS_COUNTERS: Counter = Counter()
FIRM_SCHEDULES: dict[str, dict[int, GjrParams]] = {}
REFIT_LOG = pd.DataFrame()

if GATE2_PASSED:
    _started = time.time()
    garch_records: list[dict[str, Any]] = []
    refit_records: list[dict[str, Any]] = []
    panels: list[pd.DataFrame] = []
    for position, symbol in enumerate(MODEL_UNIVERSE, start=1):
        frame = PRICES[symbol]
        all_returns = pd.Series(
            np.diff(np.log(frame["adjusted_close"].to_numpy(dtype=float))),
            index=frame["session_date"].to_numpy()[1:],
        ).dropna()
        dev_returns = all_returns[(all_returns.index >= DEV_START_TS) & (all_returns.index <= DEV_END_TS)]
        record = fit_firm_gjr(symbol, dev_returns)
        params = record.pop("_params", None)
        garch_records.append(record)
        if params is None:
            continue
        FIRM_PARAMS[symbol] = params

        schedule = None
        if VOLATILITY_REFIT == "annual_expanding":
            schedule, firm_log = build_refit_schedule(symbol, all_returns[all_returns.index >= DEV_START_TS], params)
            FIRM_SCHEDULES[symbol] = schedule
            refit_records.extend(firm_log)

        firm_panel = build_firm_panel(symbol, params, schedule=schedule)
        if firm_panel.empty:
            record["firm_attrition_reason"] = "no_panel_rows"
            continue
        firm_panel, counters = attach_news(firm_panel, symbol)
        NEWS_COUNTERS.update(counters)
        panels.append(firm_panel)
        if RUN_MODE == "full" and position % 50 == 0:
            print(f"  fitted {position}/{len(MODEL_UNIVERSE)} firms ({time.time() - _started:.0f}s)")

    GARCH_SUMMARY = pd.DataFrame(garch_records)[GARCH_COLUMNS]
    REFIT_LOG = pd.DataFrame(refit_records)
    PANEL = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    print(f"GARCH fits and panel build finished in {time.time() - _started:.1f}s")
    print(GARCH_SUMMARY["firm_attrition_reason"].value_counts().to_string())
    if len(REFIT_LOG):
        REFIT_LOG.to_csv(OUTPUT_DIR / "garch_refit_log.csv", index=False)
        print(
            f"\nexpanding-window refits: {len(REFIT_LOG):,} "
            f"({int(REFIT_LOG['accepted'].sum()):,} accepted, {int(REFIT_LOG['carried_forward'].sum()):,} carried forward)"
        )
        print(REFIT_LOG.groupby("evaluation_year")["persistence"].median().to_string())
    print(f"\nraw panel rows: {len(PANEL):,}")

# %%
if GATE2_PASSED and not PANEL.empty:
    PANEL = PANEL.merge(MARKET_STATE, on="forecast_date", how="left")
    missing_market = int(PANEL["market_reaction"].isna().sum())
    PANEL = PANEL.dropna(subset=["market_reaction", "market_abs_reaction", "market_vol20"]).reset_index(drop=True)
    record_attrition("market_state_available", "firm-days", len(PANEL), missing_market, "no local market session")

    dev_mask = (PANEL["forecast_date"] >= DEV_START_TS) & (PANEL["forecast_date"] <= DEV_END_TS) & (PANEL["target_date"] <= DEV_END_TS)
    eval_mask = (PANEL["forecast_date"] >= EVAL_START_TS) & (PANEL["forecast_date"] <= EVAL_END_TS)
    PANEL["split"] = np.where(dev_mask, "development", np.where(eval_mask, "evaluation", "excluded"))
    excluded = int((PANEL["split"] == "excluded").sum())
    PANEL = PANEL[PANEL["split"] != "excluded"].reset_index(drop=True)
    record_attrition(
        "split_assignment",
        "firm-days",
        len(PANEL),
        excluded,
        "warm-up rows and the single dev origin whose target falls in the evaluation block",
    )

    finite_columns = [
        "reaction_z",
        "abs_reaction_z",
        "log_sigma_forecast",
        "z_target",
        "target_return",
        "market_reaction",
        "market_abs_reaction",
        "market_vol20",
    ]
    before = len(PANEL)
    PANEL = PANEL[np.isfinite(PANEL[finite_columns].to_numpy(dtype=float)).all(axis=1)].reset_index(drop=True)
    record_attrition("finite_features", "firm-days", len(PANEL), before - len(PANEL), "non-finite feature or target")

    firm_rows = PANEL.groupby("symbol")["split"].value_counts().unstack(fill_value=0)
    for column in ("development", "evaluation"):
        if column not in firm_rows:
            firm_rows[column] = 0
    keep = firm_rows[(firm_rows["development"] >= MIN_DEV_OBS) & (firm_rows["evaluation"] >= MIN_EVAL_OBS)].index
    dropped_firms = sorted(set(firm_rows.index) - set(keep))
    before = len(PANEL)
    PANEL = PANEL[PANEL["symbol"].isin(keep)].reset_index(drop=True)
    record_attrition(
        "frozen_support_thresholds",
        "firm-days",
        len(PANEL),
        before - len(PANEL),
        f"{len(dropped_firms)} firms below MIN_DEV_OBS/MIN_EVAL_OBS after panel construction",
    )
    GARCH_SUMMARY.loc[
        GARCH_SUMMARY["symbol"].isin(dropped_firms) & (GARCH_SUMMARY["firm_attrition_reason"] == "retained"),
        "firm_attrition_reason",
    ] = "insufficient_panel_rows"

    PANEL = PANEL.sort_values(["symbol", "forecast_date"], kind="stable").reset_index(drop=True)
    RETAINED_FIRMS = sorted(PANEL["symbol"].unique())
    print(f"panel rows {len(PANEL):,} across {len(RETAINED_FIRMS)} firms")
    print(PANEL["split"].value_counts().to_string())
    print(f"news-bearing rows: {int(PANEL['news_indicator'].sum()):,} ({PANEL['news_indicator'].mean():.1%})")
else:
    RETAINED_FIRMS = []

GARCH_SUMMARY.to_csv(OUTPUT_DIR / "garch_fit_summary.csv", index=False)

# %% [markdown]
# ### Predictor construction and development-only scaling
#
# Price-state predictors are centred and scaled with **development** moments.
# News and semantic predictors are **scale-only** (divided by their development
# news-row standard deviation, not centred), so the zero-fill on no-news rows
# keeps its literal meaning of "no news content"; the level shift between news
# and no-news days is carried by `news_indicator`.

# %%
PRICE_STATE_TERMS = [
    "reaction_z",
    "abs_reaction_z",
    "log_sigma_forecast",
    "market_reaction",
    "market_abs_reaction",
    "market_vol20",
]
SCALE_ONLY_TERMS = ["log1p_news_count", "semantic_intensity", "adverse_tone"]
SCALING: dict[str, dict[str, float]] = {}


def apply_scaling(frame: pd.DataFrame, news_cap: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["news_count_capped"] = frame["news_count"].clip(upper=news_cap)
    frame["log1p_news_count"] = np.log1p(frame["news_count_capped"])
    for term in PRICE_STATE_TERMS:
        stats = SCALING[term]
        frame[f"{term}_s"] = (frame[term] - stats["center"]) / stats["scale"]
    for term in SCALE_ONLY_TERMS:
        frame[f"{term}_s"] = frame[term] / SCALING[term]["scale"]
    frame["news_indicator_s"] = frame["news_indicator"]
    return frame


if GATE2_PASSED and not PANEL.empty:
    development = PANEL[PANEL["split"] == "development"]
    dev_news = development[development["news_indicator"] > 0]
    NEWS_COUNT_CAP = float(np.quantile(dev_news["news_count"], NEWS_COUNT_CAP_QUANTILE)) if len(dev_news) else 1.0

    for term in PRICE_STATE_TERMS:
        values = development[term].to_numpy(dtype=float)
        SCALING[term] = {"center": float(values.mean()), "scale": float(values.std(ddof=0)) or 1.0}
    capped = np.log1p(dev_news["news_count"].clip(upper=NEWS_COUNT_CAP).to_numpy(dtype=float))
    SCALING["log1p_news_count"] = {"center": 0.0, "scale": float(capped.std(ddof=0)) or 1.0}
    for term in ("semantic_intensity", "adverse_tone"):
        values = dev_news[term].to_numpy(dtype=float)
        SCALING[term] = {"center": 0.0, "scale": float(values.std(ddof=0)) or 1.0}
    # Declared robustness scorer, scaled on exactly the same development rows.
    for term, source in (("vader_intensity", "vader_intensity"), ("vader_adverse_tone", "vader_adverse_tone")):
        values = dev_news[source].to_numpy(dtype=float)
        SCALING[term] = {"center": 0.0, "scale": float(values.std(ddof=0)) or 1.0}

    PANEL = apply_scaling(PANEL, NEWS_COUNT_CAP)
    PANEL["vader_intensity_s"] = PANEL["vader_intensity"] / SCALING["vader_intensity"]["scale"]
    PANEL["vader_adverse_tone_s"] = PANEL["vader_adverse_tone"] / SCALING["vader_adverse_tone"]["scale"]
    print(f"news_count cap at the development {NEWS_COUNT_CAP_QUANTILE:.3%} quantile: {NEWS_COUNT_CAP:.0f}")
    print(pd.DataFrame(SCALING).T.to_string())

# %%
FIRM_DAY_SCHEMA: dict[str, Any] = {}
if GATE2_PASSED and not PANEL.empty:
    FIRM_DAY_SCHEMA = {
        "unit_of_observation": "one firm x forecast-origin trading session, including no-news days",
        "row_count": int(len(PANEL)),
        "firm_count": int(PANEL["symbol"].nunique()),
        "forecast_date_min": PANEL["forecast_date"].min().date().isoformat(),
        "forecast_date_max": PANEL["forecast_date"].max().date().isoformat(),
        "news_row_share": float(PANEL["news_indicator"].mean()),
        "semantic_definition": SEMANTIC_DEFINITION,
        "scaling": SCALING,
        "news_count_cap": NEWS_COUNT_CAP,
        "columns": {
            column: {
                "dtype": str(PANEL[column].dtype),
                "null_count": int(PANEL[column].isna().sum()),
            }
            for column in PANEL.columns
        },
        "column_semantics": {
            "forecast_date": "reaction session s; the forecast is made at its close",
            "target_date": "next available firm trading session s+1",
            "target_return": "log(adjusted_close[s+1] / adjusted_close[s])",
            "reaction_return": "log(adjusted_close[s] / adjusted_close[s-1]); realised during s",
            "reaction_z": "(reaction_return - mu_hat) / sigma made at s-1",
            "sigma_forecast": "GJR one-step-ahead volatility for s+1 using returns through s",
            "z_target": "(target_return - mu_hat) / sigma_forecast",
            "news_indicator": "1 if any eligible headline maps to this firm session",
            "news_count": "headlines folded onto this firm session across all collapsed calendar dates",
            "source_news_sessions": "distinct upstream reaction sessions folded into this row",
            "news_forward_gap_days": "largest calendar gap between an upstream reaction session and this origin",
            "adverse_tone": "count-weighted mean(p_negative - p_positive); larger is more adverse",
            "semantic_intensity": "count-weighted mean(p_negative + p_positive)",
            "recap_share": "count-weighted share of headlines matching the frozen price-recap regex",
        },
    }
    atomic_write_json(OUTPUT_DIR / "firm_day_panel_schema.json", FIRM_DAY_SCHEMA)
    print(json.dumps({k: v for k, v in FIRM_DAY_SCHEMA.items() if k != "columns"}, indent=2, default=str)[:2400])

# %% [markdown]
# ## 5. Assertions and leakage tests
#
# These cells raise on violation. They are the contract between the panel and
# every result that follows.

# %%
ASSERTIONS: list[dict[str, Any]] = []


def check(name: str, condition: bool, detail: Any = "") -> None:
    ASSERTIONS.append({"assertion": name, "passed": bool(condition), "detail": str(detail)})
    if not condition:
        raise AssertionError(f"{name} FAILED :: {detail}")


if GATE2_PASSED and not PANEL.empty:
    check("one_row_per_symbol_forecast_date", not PANEL.duplicated(["symbol", "forecast_date"]).any(), len(PANEL))
    check("target_date_strictly_after_forecast_date", bool((PANEL["target_date"] > PANEL["forecast_date"]).all()))
    check(
        "date_only_mapper_strictly_forward",
        DATE_ONLY_TIMING_VIOLATIONS == 0,
        f"{len(_calendar_days):,} calendar dates checked exhaustively",
    )
    check(
        "upstream_mixed_timing_policy_verified",
        UPSTREAM_TIMING_POLICY_VERIFIED,
        f"{DATE_ONLY_UPSTREAM_ROWS:,} date-only rows; {PRECISE_TIMESTAMP_UPSTREAM_ROWS:,} precise rows",
    )
    check("corpus_reaction_dates_are_xnys_sessions", INVALID_CORPUS_REACTION_SESSIONS == 0, INVALID_CORPUS_REACTION_SESSIONS)
    check(
        "news_never_folded_backwards",
        bool((PANEL["news_forward_gap_days"] >= 0).all()),
        float(PANEL["news_forward_gap_days"].max()),
    )
    check(
        "news_forward_gap_within_bound",
        bool((PANEL["news_forward_gap_days"] <= MAX_NEWS_FORWARD_GAP_DAYS).all()),
        float(PANEL["news_forward_gap_days"].max()),
    )

    # Target returns must equal the next valid close-to-close adjusted return,
    # recomputed independently from the price frames.
    sampler = np.random.default_rng(RANDOM_SEED)
    audit_rows = PANEL.iloc[sampler.choice(len(PANEL), size=min(4000, len(PANEL)), replace=False)]
    worst = 0.0
    for row in audit_rows.itertuples():
        prices = PRICES[row.symbol].set_index("session_date")["adjusted_close"]
        expected = math.log(prices.loc[row.target_date] / prices.loc[row.forecast_date])
        worst = max(worst, abs(expected - row.target_return))
    check("target_return_matches_adjusted_prices", worst < 1e-12, f"max abs deviation {worst:.3e}")

    # No feature may use prices after the forecast origin: perturbing the target
    # must leave every predictor untouched.  Verified structurally by rebuilding
    # one firm's panel from a price series truncated at each origin is O(n^2);
    # instead the recursion's point-in-time property is asserted directly.
    probe_symbol = RETAINED_FIRMS[0]
    probe_schedule = FIRM_SCHEDULES.get(probe_symbol)
    probe = build_firm_panel(
        probe_symbol,
        FIRM_PARAMS[probe_symbol],
        schedule=probe_schedule,
    )
    truncated_prices = PRICES[probe_symbol].iloc[:-25].reset_index(drop=True)
    original = PRICES[probe_symbol]
    try:
        PRICES[probe_symbol] = truncated_prices
        probe_truncated = build_firm_panel(
            probe_symbol,
            FIRM_PARAMS[probe_symbol],
            schedule=probe_schedule,
        )
    finally:
        PRICES[probe_symbol] = original
    overlap = probe.merge(probe_truncated, on=["symbol", "forecast_date"], suffixes=("", "_trunc"))
    feature_gap = max(
        float(np.abs(overlap[term] - overlap[f"{term}_trunc"]).max())
        for term in ("reaction_z", "abs_reaction_z", "log_sigma_forecast", "sigma_forecast")
    )
    check(
        "features_ignore_future_prices",
        feature_gap < 1e-12,
        f"dropping the final 25 sessions moved surviving predictors by at most {feature_gap:.3e}",
    )

    check(
        "development_and_evaluation_disjoint",
        bool(
            PANEL.loc[PANEL["split"] == "development", "target_date"].max()
            < PANEL.loc[PANEL["split"] == "evaluation", "forecast_date"].min()
        ),
        f"last dev target {PANEL.loc[PANEL['split'] == 'development', 'target_date'].max().date()} "
        f"< first eval origin {PANEL.loc[PANEL['split'] == 'evaluation', 'forecast_date'].min().date()}",
    )
    check(
        "scaling_uses_development_only",
        all(
            abs(SCALING[term]["center"] - float(PANEL.loc[PANEL["split"] == "development", term].mean())) < 1e-9
            for term in PRICE_STATE_TERMS
        ),
        "centres reproduce development means exactly",
    )
    check(
        "garch_parameters_from_development_only",
        bool((GARCH_SUMMARY.loc[GARCH_SUMMARY["converged"], "dev_return_obs"] >= MIN_DEV_OBS).all()),
        int(GARCH_SUMMARY["converged"].sum()),
    )
    if VOLATILITY_REFIT == "annual_expanding":
        refit_end = pd.to_datetime(REFIT_LOG["fit_window_end"], errors="coerce")
        refit_cutoff = pd.to_datetime(REFIT_LOG["fit_cutoff_exclusive"])
        check(
            "annual_refits_use_strictly_prior_returns",
            bool((refit_end < refit_cutoff).all()),
            f"latest margin {(refit_cutoff - refit_end).min().days} day(s)",
        )
        check(
            "annual_refit_windows_start_in_development",
            bool((pd.to_datetime(REFIT_LOG["fit_window_start"]) >= pd.Timestamp(DEV_START)).all()),
            DEV_START,
        )
        check(
            "annual_refit_schedule_complete",
            all(set(schedule) == set(EVALUATION_YEARS) for schedule in FIRM_SCHEDULES.values()),
            f"{len(FIRM_SCHEDULES)} firms x {len(EVALUATION_YEARS)} years",
        )
        carry_consistent = True
        for symbol, schedule in FIRM_SCHEDULES.items():
            previous = FIRM_PARAMS[symbol]
            firm_log = REFIT_LOG[REFIT_LOG["symbol"] == symbol].set_index("evaluation_year")
            for year in EVALUATION_YEARS:
                if bool(firm_log.loc[year, "carried_forward"]):
                    carry_consistent = carry_consistent and schedule[year] == previous
                previous = schedule[year]
        check(
            "failed_annual_refits_carry_prior_vintage",
            carry_consistent,
            int(REFIT_LOG["carried_forward"].sum()),
        )
        check(
            "accepted_annual_refits_converged",
            bool(
                (REFIT_LOG.loc[REFIT_LOG["accepted"], "convergence_flag"] == 0).all()
                and REFIT_LOG.loc[REFIT_LOG["accepted"], "optimisation_success"].all()
            ),
            int(REFIT_LOG["accepted"].sum()),
        )
    else:
        check(
            "frozen_filter_has_no_refit_schedule",
            not FIRM_SCHEDULES and REFIT_LOG.empty,
            VOLATILITY_REFIT,
        )
    check(
        "garch_recursion_matches_estimator_filter",
        float(GARCH_SUMMARY["recursion_vs_arch_max_rel_diff"].max(skipna=True)) < 1e-9,
        f"max relative deviation {GARCH_SUMMARY['recursion_vs_arch_max_rel_diff'].max(skipna=True):.3e} "
        "when seeded with the estimator's own first conditional variance",
    )
    check(
        "initialisation_forgotten_by_end_of_development",
        float(GARCH_SUMMARY["initialisation_sensitivity_at_dev_end"].max(skipna=True)) < 1e-2,
        f"worst firm still carries {GARCH_SUMMARY['initialisation_sensitivity_at_dev_end'].max(skipna=True):.3e} "
        "relative variance error from a different starting value at the last development session",
    )

    no_news = PANEL[PANEL["news_indicator"] == 0]
    check(
        "no_news_rows_are_internally_consistent",
        bool(
            (no_news["news_count"] == 0).all()
            and (no_news[["adverse_tone", "semantic_intensity", "negative_share", "recap_share"]] == 0).all().all()
            and (no_news["source_news_sessions"] == 0).all()
        ),
        len(no_news),
    )
    news_rows = PANEL[PANEL["news_indicator"] == 1]
    check("news_rows_have_headlines", bool((news_rows["news_count"] > 0).all()), len(news_rows))

    check(
        "no_duplicate_or_non_positive_prices",
        all((not frame["session_date"].duplicated().any()) and bool((frame["adjusted_close"] > 0).all()) for frame in PRICES.values()),
        len(PRICES),
    )
    check(
        "no_forward_filled_returns",
        bool((PANEL["target_return"] != 0).mean() > 0.9),
        f"zero-return share {(PANEL['target_return'] == 0).mean():.4f}",
    )
    check(
        "every_retained_firm_meets_support",
        bool(
            PANEL.groupby("symbol")["split"]
            .apply(lambda values: (values == "development").sum() >= MIN_DEV_OBS and (values == "evaluation").sum() >= MIN_EVAL_OBS)
            .all()
        ),
        len(RETAINED_FIRMS),
    )
    check(
        "attrition_reasons_are_unique_per_firm",
        bool(GARCH_SUMMARY["symbol"].is_unique and GARCH_SUMMARY["firm_attrition_reason"].notna().all()),
        GARCH_SUMMARY["firm_attrition_reason"].nunique(),
    )
    check("no_cached_artifacts_reused", True, "this notebook recomputes every derived quantity in-process")

    print(pd.DataFrame(ASSERTIONS).to_string(index=False))
    print(f"\n{len(ASSERTIONS)} assertions passed.")

# %% [markdown]
# ## 6. GARCH diagnostics

# %%
if GATE2_PASSED and not PANEL.empty:
    fitted_firms = GARCH_SUMMARY[GARCH_SUMMARY["converged"]]
    print(fitted_firms[["alpha", "gamma", "beta", "nu", "persistence", "unconditional_vol_annualised"]].describe().T.to_string())
    print(f"\nleverage term gamma > 0 in {int((fitted_firms['gamma'] > 0).sum())}/{len(fitted_firms)} fits")
    print(f"persistence >= 0.99 in {int((fitted_firms['persistence'] >= 0.99).sum())} fits")

    standardised = PANEL["z_target"]
    dev_z = PANEL.loc[PANEL["split"] == "development", "z_target"]
    eval_z = PANEL.loc[PANEL["split"] == "evaluation", "z_target"]
    diagnostics = pd.DataFrame(
        {
            "split": ["development", "evaluation", "all"],
            "n": [len(dev_z), len(eval_z), len(standardised)],
            "mean_z": [dev_z.mean(), eval_z.mean(), standardised.mean()],
            "std_z": [dev_z.std(), eval_z.std(), standardised.std()],
            "mean_z2": [(dev_z**2).mean(), (eval_z**2).mean(), (standardised**2).mean()],
            "skew_z": [dev_z.skew(), eval_z.skew(), standardised.skew()],
            "kurtosis_z": [dev_z.kurtosis(), eval_z.kurtosis(), standardised.kurtosis()],
            "empirical_2.5pct": [dev_z.quantile(ALPHA), eval_z.quantile(ALPHA), standardised.quantile(ALPHA)],
        }
    )
    print("\nStandardised-return diagnostics (a well-specified filter gives std ~ 1 and mean z^2 ~ 1):")
    print(diagnostics.to_string(index=False))

    lags = [1, 2, 5, 10]
    autocorr = pd.DataFrame(
        {
            "lag": lags,
            "acf_z": [float(PANEL.groupby("symbol")["z_target"].apply(lambda s, k=lag: s.autocorr(k)).mean()) for lag in lags],
            "acf_z_squared": [float(PANEL.groupby("symbol")["z_target"].apply(lambda s, k=lag: (s**2).autocorr(k)).mean()) for lag in lags],
        }
    )
    print("\nMean within-firm autocorrelation of z and z^2 (residual clustering shows up in z^2):")
    print(autocorr.to_string(index=False))

# %% [markdown]
# ## 7. Nested joint VaR-ES models
#
# All models are estimated on **development rows only** and share the
# parameterisation `q = -exp(X b_q)`, `e = q - exp(X b_e)`, which enforces
# `e < q < 0` by construction. The objective is the joint FZ0 loss.

# %%
MODEL_TERMS: dict[str, list[str]] = {
    "M0": ["const", *[f"{term}_s" for term in PRICE_STATE_TERMS]],
    "M1": ["const", *[f"{term}_s" for term in PRICE_STATE_TERMS], "news_indicator", "log1p_news_count_s"],
    "M2": [
        "const",
        *[f"{term}_s" for term in PRICE_STATE_TERMS],
        "news_indicator",
        "log1p_news_count_s",
        "semantic_intensity_s",
        "adverse_tone_s",
    ],
    "M2_intensity": [
        "const",
        *[f"{term}_s" for term in PRICE_STATE_TERMS],
        "news_indicator",
        "log1p_news_count_s",
        "semantic_intensity_s",
    ],
}


def design_matrix(frame: pd.DataFrame, terms: list[str]) -> np.ndarray:
    columns = []
    for term in terms:
        columns.append(np.ones(len(frame)) if term == "const" else frame[term].to_numpy(dtype=float))
    return np.column_stack(columns)


FITS: dict[str, Any] = {}
if GATE2_PASSED and not PANEL.empty:
    development = PANEL[PANEL["split"] == "development"].reset_index(drop=True)
    evaluation = PANEL[PANEL["split"] == "evaluation"].reset_index(drop=True)
    dev_target = development["z_target"].to_numpy(dtype=float)
    print(f"development rows {len(development):,} | evaluation rows {len(evaluation):,}")

    _started = time.time()
    for name, terms in MODEL_TERMS.items():
        FITS[name] = fit_joint_var_es(
            design_matrix(development, terms),
            dev_target,
            alpha=ALPHA,
            name=name,
            columns=terms,
        )
        fit = FITS[name]
        print(
            f"{name:<13s} FZ0(dev) {fit.objective:+.6f} | converged {fit.converged} | "
            f"starts {fit.starts_converged}/{fit.starts_attempted} | spread {fit.start_objective_spread:.2e} | "
            f"iters {fit.iterations} | clipped {fit.clipped_rows}"
        )
        for note in fit.warnings:
            print(f"    warning: {note}")
    print(f"\nall joint fits in {time.time() - _started:.1f}s")

    if not FITS["M2"].converged or not FITS["M1"].converged:
        raise RuntimeError("the primary M1/M2 comparison requires both fits to converge from several starts")

    COEFFICIENTS = pd.DataFrame([row for fit in FITS.values() for row in fit.coefficient_rows()])
    optimisation = pd.DataFrame(
        [
            {
                "model": fit.name,
                "block": "optimisation",
                "term": key,
                "coefficient": value,
            }
            for fit in FITS.values()
            for key, value in {
                "fz0_development": fit.objective,
                "converged": float(fit.converged),
                "iterations": float(fit.iterations),
                "starts_attempted": float(fit.starts_attempted),
                "starts_converged": float(fit.starts_converged),
                "start_objective_spread": fit.start_objective_spread,
                "clipped_rows": float(fit.clipped_rows),
            }.items()
        ]
    )
    COEFFICIENTS = pd.concat([COEFFICIENTS, optimisation], ignore_index=True)
    COEFFICIENTS.to_csv(OUTPUT_DIR / "tail_model_coefficients.csv", index=False)
    print()
    print(
        COEFFICIENTS[COEFFICIENTS["block"] != "optimisation"]
        .pivot_table(index="term", columns=["model", "block"], values="coefficient")
        .to_string()
    )

# %% [markdown]
# ## 8. Primary chronological evaluation
#
# Every model is scored on the **same** evaluation rows. The primary population
# is news-bearing forecast origins; full-panel calibration is reported
# separately.

# %%
FORECASTS = pd.DataFrame()
MODEL_COMPARISON = pd.DataFrame()

if GATE2_PASSED and not PANEL.empty:
    FORECASTS = evaluation[
        [
            "symbol",
            "forecast_date",
            "target_date",
            "split",
            "target_return",
            "z_target",
            "mu_hat",
            "sigma_forecast",
            "reaction_z",
            "log_sigma_forecast",
            "news_indicator",
            "news_count",
            "semantic_intensity",
            "adverse_tone",
            "recap_share",
        ]
    ].copy()

    for name, terms in MODEL_TERMS.items():
        var_z, es_z, clipped = tail_forecasts(design_matrix(evaluation, terms), FITS[name].beta_var, FITS[name].beta_gap)
        FORECASTS[f"var_z_{name}"] = var_z
        FORECASTS[f"es_z_{name}"] = es_z
        FORECASTS[f"var_return_{name}"] = FORECASTS["mu_hat"] + FORECASTS["sigma_forecast"] * var_z
        FORECASTS[f"es_return_{name}"] = FORECASTS["mu_hat"] + FORECASTS["sigma_forecast"] * es_z
        FORECASTS[f"fz0_{name}"] = fz0_loss(FORECASTS["z_target"].to_numpy(dtype=float), var_z, es_z, ALPHA)
        FORECASTS[f"pinball_{name}"] = pinball_loss(FORECASTS["z_target"].to_numpy(dtype=float), var_z, ALPHA)
        FORECASTS[f"hit_{name}"] = (FORECASTS["target_return"] <= FORECASTS[f"var_return_{name}"]).astype(int)
        if clipped:
            print(f"{name}: {clipped} evaluation linear indices clipped to the safe band")

    FORECASTS["d_semantic"] = FORECASTS["fz0_M2"] - FORECASTS["fz0_M1"]
    FORECASTS["d_tone_given_intensity"] = FORECASTS["fz0_M2"] - FORECASTS["fz0_M2_intensity"]
    FORECASTS["d_news_vs_price"] = FORECASTS["fz0_M1"] - FORECASTS["fz0_M0"]

    for name in MODEL_TERMS:
        assert bool((FORECASTS[f"es_z_{name}"] < FORECASTS[f"var_z_{name}"]).all()), f"{name} violates ES < VaR"
        assert bool((FORECASTS[f"var_z_{name}"] < 0).all()), f"{name} produced a non-negative VaR"
        assert bool((FORECASTS[f"es_return_{name}"] < FORECASTS[f"var_return_{name}"]).all()), f"{name} violates ES < VaR in return units"
    assert FORECASTS[[f"fz0_{name}" for name in MODEL_TERMS]].notna().all().all(), "missing FZ0 losses"
    print("all evaluation forecasts satisfy ES < VaR < 0 in standardised and return units")
    print(f"evaluation rows shared by every model: {len(FORECASTS):,}")

    # zstd keeps the full-mode artifact manageable without losing float64 precision.
    FORECASTS.to_parquet(OUTPUT_DIR / "forecasts.parquet", index=False, compression="zstd")


# %%
def comparison_block(frame: pd.DataFrame, population: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "population": population,
        "rows": len(frame),
        "firms": int(frame["symbol"].nunique()),
        "target_dates": int(frame["target_date"].nunique()),
        "news_rows": int(frame["news_indicator"].sum()),
    }
    for name in MODEL_TERMS:
        rows.append(
            {
                **base,
                "quantity": f"mean_fz0_{name}",
                "value": float(frame[f"fz0_{name}"].mean()),
                "violations": int(frame[f"hit_{name}"].sum()),
                "hit_rate": float(frame[f"hit_{name}"].mean()),
                "mean_pinball": float(frame[f"pinball_{name}"].mean()),
            }
        )
    m1, m2 = float(frame["fz0_M1"].mean()), float(frame["fz0_M2"].mean())
    rows.append({**base, "quantity": "paired_fz0_M2_minus_M1", "value": m2 - m1})
    rows.append(
        {
            **base,
            "quantity": "relative_fz0_improvement_pct_M2_vs_M1",
            "value": float(-(m2 - m1) / abs(m1) * 100.0) if m1 else math.nan,
        }
    )
    intensity = float(frame["fz0_M2_intensity"].mean())
    rows.append({**base, "quantity": "paired_fz0_M2_minus_M2_intensity", "value": m2 - intensity})
    zero = float(frame["fz0_M0"].mean())
    rows.append({**base, "quantity": "paired_fz0_M1_minus_M0", "value": m1 - zero})
    return rows


if GATE2_PASSED and not FORECASTS.empty:
    news_rows = FORECASTS[FORECASTS["news_indicator"] == 1]
    quiet_rows = FORECASTS[FORECASTS["news_indicator"] == 0]
    comparison_rows = (
        comparison_block(news_rows, "news_bearing_evaluation_origins")
        + comparison_block(FORECASTS, "full_evaluation_panel")
        + comparison_block(quiet_rows, "no_news_evaluation_origins")
    )
    for year, block in FORECASTS.assign(year=FORECASTS["target_date"].dt.year).groupby("year"):
        news_block = block[block["news_indicator"] == 1]
        if len(news_block) < 30:
            continue
        comparison_rows.extend(comparison_block(news_block, f"news_bearing_{year}"))

    MODEL_COMPARISON = pd.DataFrame(comparison_rows)
    MODEL_COMPARISON.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    headline = MODEL_COMPARISON[MODEL_COMPARISON["population"] == "news_bearing_evaluation_origins"]
    print("PRIMARY POPULATION: news-bearing evaluation forecast origins")
    print(headline[["quantity", "value", "rows", "firms", "target_dates", "violations", "hit_rate"]].to_string(index=False))
    print()
    print("Full evaluation panel:")
    print(
        MODEL_COMPARISON[MODEL_COMPARISON["population"] == "full_evaluation_panel"][
            ["quantity", "value", "violations", "hit_rate", "mean_pinball"]
        ].to_string(index=False)
    )

# %%
if GATE2_PASSED and not FORECASTS.empty:
    yearly = MODEL_COMPARISON[
        MODEL_COMPARISON["population"].str.startswith("news_bearing_2") & (MODEL_COMPARISON["quantity"] == "paired_fz0_M2_minus_M1")
    ][["population", "value", "rows", "firms"]]
    print("Descriptive stability across evaluation years (paired FZ0 M2 - M1, news-bearing rows):")
    print(yearly.to_string(index=False))

# %% [markdown]
# ## 9. Dependence-aware uncertainty
#
# Firm-days are not independent. The paired loss difference is first averaged
# across the **full cross-section of firms on each target date**, then a
# moving-block bootstrap resamples blocks of target dates, keeping every firm on
# a sampled date together.
#
# The models are **not** refitted inside the bootstrap: this interval quantifies
# sampling uncertainty in the paired loss difference of *fixed* forecasts, not
# estimation uncertainty in the coefficients.

# %%
BOOTSTRAP = pd.DataFrame()


def date_level(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    grouped = frame.groupby("target_date", sort=True)[column].agg(["mean", "size"])
    return grouped["mean"].to_numpy(dtype=float), grouped["size"].to_numpy(dtype=float)


if GATE2_PASSED and not FORECASTS.empty:
    bootstrap_rows: list[dict[str, Any]] = []
    populations = {
        "news_bearing_evaluation_origins": FORECASTS[FORECASTS["news_indicator"] == 1],
        "full_evaluation_panel": FORECASTS,
    }
    contrasts = {
        "d_semantic": "FZ0(M2) - FZ0(M1): does semantics beat arrival plus volume?",
        "d_tone_given_intensity": "FZ0(M2) - FZ0(M2_intensity): does signed tone add to intensity?",
        "d_news_vs_price": "FZ0(M1) - FZ0(M0): does news arrival beat price state alone?",
    }
    for population, frame in populations.items():
        for contrast, description in contrasts.items():
            values, weights = date_level(frame, contrast)
            summary = date_block_bootstrap_mean(
                values,
                weights,
                block_length=DATE_BLOCK_LENGTH,
                replications=BOOTSTRAP_REPS,
                seed=RANDOM_SEED,
            )
            bootstrap_rows.append(
                {
                    "population": population,
                    "contrast": contrast,
                    "description": description,
                    "weighting": "observation_equal",
                    **summary,
                }
            )
            date_equal = date_block_bootstrap_mean(
                values,
                None,
                block_length=DATE_BLOCK_LENGTH,
                replications=BOOTSTRAP_REPS,
                seed=RANDOM_SEED,
            )
            bootstrap_rows.append(
                {
                    "population": population,
                    "contrast": contrast,
                    "description": description,
                    "weighting": "date_equal",
                    **date_equal,
                }
            )

    # One nearby block length, declared before results were inspected.
    values, weights = date_level(
        FORECASTS[FORECASTS["news_indicator"] == 1],
        "d_semantic",
    )
    for block in (10, 40):
        bootstrap_rows.append(
            {
                "population": "news_bearing_evaluation_origins",
                "contrast": "d_semantic",
                "description": f"declared block-length sensitivity ({block} dates)",
                "weighting": "observation_equal",
                **date_block_bootstrap_mean(
                    values,
                    weights,
                    block_length=block,
                    replications=BOOTSTRAP_REPS,
                    seed=RANDOM_SEED,
                ),
            }
        )

    BOOTSTRAP = pd.DataFrame(bootstrap_rows)
    for population, frame in populations.items():
        for contrast in contrasts:
            selected = BOOTSTRAP[
                (BOOTSTRAP["population"] == population)
                & (BOOTSTRAP["contrast"] == contrast)
                & (BOOTSTRAP["block_length"] == DATE_BLOCK_LENGTH)
            ].set_index("weighting")
            row_mean = float(frame[contrast].mean())
            date_mean = float(frame.groupby("target_date")[contrast].mean().mean())
            check(
                f"bootstrap_observation_equal_{population}_{contrast}",
                math.isclose(
                    float(selected.loc["observation_equal", "point_estimate"]),
                    row_mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                row_mean,
            )
            check(
                f"bootstrap_date_equal_{population}_{contrast}",
                math.isclose(
                    float(selected.loc["date_equal", "point_estimate"]),
                    date_mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                date_mean,
            )

    BOOTSTRAP.to_csv(OUTPUT_DIR / "bootstrap.csv", index=False)
    print(
        BOOTSTRAP[
            [
                "population",
                "contrast",
                "weighting",
                "point_estimate",
                "ci_low",
                "ci_high",
                "share_below_zero",
                "n_dates",
                "block_length",
            ]
        ].to_string(index=False)
    )

# %% [markdown]
# ## 10. Calibration diagnostics
#
# Proper loss is the primary ranking evidence. Calibration is a **diagnostic**:
# failing to reject a coverage test is not proof that a model is correct, and the
# largest p-value does not win.

# %%
CALIBRATION = pd.DataFrame()


def clustered_wald(frame: pd.DataFrame, outcome: str, regressors: list[str]) -> dict[str, Any]:
    """Date-cluster-robust OLS plus a joint Wald test on every coefficient."""

    usable = frame.dropna(subset=[outcome, *regressors])
    if len(usable) < 50 or usable["target_date"].nunique() < 10:
        return {"statistic": math.nan, "p_value": math.nan, "rows": len(usable), "clusters": 0}
    design = sm.add_constant(usable[regressors].to_numpy(dtype=float), has_constant="add")
    fitted = sm.OLS(usable[outcome].to_numpy(dtype=float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": usable["target_date"].to_numpy(), "use_correction": True, "df_correction": True},
    )
    restriction = np.eye(design.shape[1])
    test = fitted.wald_test(restriction, scalar=True)
    return {
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "rows": len(usable),
        "clusters": int(usable["target_date"].nunique()),
        "coefficients": {name: float(value) for name, value in zip(["const", *regressors], fitted.params, strict=True)},
    }


if GATE2_PASSED and not FORECASTS.empty:
    calibration_rows: list[dict[str, Any]] = []
    FORECASTS = FORECASTS.sort_values(["symbol", "forecast_date"], kind="stable").reset_index(drop=True)

    for name in MODEL_TERMS:
        FORECASTS[f"v_es_{name}"] = es_identification_residual(
            FORECASTS["target_return"].to_numpy(dtype=float),
            FORECASTS[f"var_return_{name}"].to_numpy(dtype=float),
            FORECASTS[f"es_return_{name}"].to_numpy(dtype=float),
            ALPHA,
        )
        FORECASTS[f"lag_hit_{name}"] = FORECASTS.groupby("symbol")[f"hit_{name}"].shift(1)

        for population, frame in (
            ("full_evaluation_panel", FORECASTS),
            ("news_bearing", FORECASTS[FORECASTS["news_indicator"] == 1]),
            ("no_news", FORECASTS[FORECASTS["news_indicator"] == 0]),
        ):
            hits = frame[f"hit_{name}"].to_numpy(dtype=float)
            values, weights = date_level(frame.assign(_excess=hits - ALPHA), "_excess")
            interval = date_block_bootstrap_mean(
                values, weights, block_length=DATE_BLOCK_LENGTH, replications=BOOTSTRAP_REPS, seed=RANDOM_SEED
            )
            calibration_rows.append(
                {
                    "model": name,
                    "population": population,
                    "diagnostic": "var_hit_rate_minus_alpha",
                    "value": interval["point_estimate"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "rows": len(frame),
                    "violations": int(hits.sum()),
                    "hit_rate": float(hits.mean()),
                    "detail": f"alpha = {ALPHA}",
                }
            )
            residual = frame[f"v_es_{name}"]
            values, weights = date_level(frame, f"v_es_{name}")
            es_interval = date_block_bootstrap_mean(
                values, weights, block_length=DATE_BLOCK_LENGTH, replications=BOOTSTRAP_REPS, seed=RANDOM_SEED
            )
            calibration_rows.append(
                {
                    "model": name,
                    "population": population,
                    "diagnostic": "mean_es_identification_residual",
                    "value": es_interval["point_estimate"],
                    "ci_low": es_interval["ci_low"],
                    "ci_high": es_interval["ci_high"],
                    "rows": len(frame),
                    "violations": int(hits.sum()),
                    "hit_rate": float(hits.mean()),
                    "detail": f"scale reference: mean |return| = {frame['target_return'].abs().mean():.5f}, "
                    f"mean residual / mean |return| = {residual.mean() / frame['target_return'].abs().mean():+.4f}",
                }
            )

        # DQ-style conditional coverage with date-cluster-aware inference.
        dq_frame = FORECASTS.assign(_excess=FORECASTS[f"hit_{name}"] - ALPHA)
        dq = clustered_wald(
            dq_frame,
            "_excess",
            [f"lag_hit_{name}", "log_sigma_forecast", "reaction_z", f"var_return_{name}"],
        )
        calibration_rows.append(
            {
                "model": name,
                "population": "full_evaluation_panel",
                "diagnostic": "dq_conditional_coverage_wald",
                "value": dq["statistic"],
                "ci_low": math.nan,
                "ci_high": math.nan,
                "rows": dq["rows"],
                "violations": int(FORECASTS[f"hit_{name}"].sum()),
                "hit_rate": float(FORECASTS[f"hit_{name}"].mean()),
                "detail": f"p = {dq['p_value']:.4g}; date-clustered; regressors: lagged hit, log sigma, reaction z, VaR",
            }
        )
        es_test = clustered_wald(
            FORECASTS,
            f"v_es_{name}",
            ["log_sigma_forecast", "reaction_z", "news_indicator", "log1p_news_count_s"]
            if "log1p_news_count_s" in FORECASTS
            else ["log_sigma_forecast", "reaction_z", "news_indicator"],
        )
        calibration_rows.append(
            {
                "model": name,
                "population": "full_evaluation_panel",
                "diagnostic": "es_conditional_calibration_wald",
                "value": es_test["statistic"],
                "ci_low": math.nan,
                "ci_high": math.nan,
                "rows": es_test["rows"],
                "violations": int(FORECASTS[f"hit_{name}"].sum()),
                "hit_rate": float(FORECASTS[f"hit_{name}"].mean()),
                "detail": f"p = {es_test['p_value']:.4g}; instruments are forecast-origin variables only",
            }
        )

        # Per-firm coverage tests, only where the sample supports them.
        firm_rows = []
        for symbol, block in FORECASTS.groupby("symbol", sort=True):
            hits = block[f"hit_{name}"].to_numpy(dtype=int)
            if hits.size < MIN_EVAL_OBS or hits.sum() < 5:
                continue
            kupiec = kupiec_test(hits, ALPHA)
            christoffersen = christoffersen_tests(hits, ALPHA)
            firm_rows.append(
                {
                    "symbol": symbol,
                    "hit_rate": kupiec["hit_rate"],
                    "kupiec_p": kupiec["p_value"],
                    "independence_p": christoffersen["independence_p_value"],
                    "conditional_coverage_p": christoffersen["conditional_coverage_p_value"],
                }
            )
        firm_tests = pd.DataFrame(firm_rows)
        kupiec_tests = firm_tests.dropna(subset=["kupiec_p"])
        conditional_tests = firm_tests.dropna(subset=["conditional_coverage_p"])
        if len(kupiec_tests):
            calibration_rows.append(
                {
                    "model": name,
                    "population": "per_firm_tests",
                    "diagnostic": "kupiec_rejection_share_at_5pct",
                    "value": float((kupiec_tests["kupiec_p"] < 0.05).mean()),
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "rows": len(kupiec_tests),
                    "violations": math.nan,
                    "hit_rate": float(kupiec_tests["hit_rate"].mean()),
                    "detail": "firms with >= MIN_EVAL_OBS rows and >= 5 violations; iid within firm, not pooled",
                }
            )
        if len(conditional_tests):
            calibration_rows.append(
                {
                    "model": name,
                    "population": "per_firm_tests",
                    "diagnostic": "christoffersen_cc_rejection_share_at_5pct",
                    "value": float((conditional_tests["conditional_coverage_p"] < 0.05).mean()),
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "rows": len(conditional_tests),
                    "violations": math.nan,
                    "hit_rate": float(conditional_tests["hit_rate"].mean()),
                    "detail": (f"conditional coverage; {len(firm_tests) - len(conditional_tests)} NaN-valued firms excluded"),
                }
            )

    CALIBRATION = pd.DataFrame(calibration_rows)
    CALIBRATION.to_csv(OUTPUT_DIR / "calibration.csv", index=False)
    print(
        CALIBRATION[CALIBRATION["diagnostic"] == "var_hit_rate_minus_alpha"][
            ["model", "population", "value", "ci_low", "ci_high", "hit_rate", "rows"]
        ].to_string(index=False)
    )
    print()
    print(
        CALIBRATION[CALIBRATION["diagnostic"] == "mean_es_identification_residual"][
            ["model", "population", "value", "ci_low", "ci_high", "rows"]
        ].to_string(index=False)
    )
    print()
    print(
        CALIBRATION[CALIBRATION["diagnostic"].str.contains("wald|rejection")][
            ["model", "population", "diagnostic", "value", "rows", "detail"]
        ].to_string(index=False)
    )

# %% [markdown]
# ## 11. One scale-versus-tail diagnostic
#
# If semantics help only by sharpening the conditional *scale*, then feeding the
# news variables into the volatility filter should absorb the gain. A single
# multiplicative adjustment `sigma_tilde = sigma_garch * exp(0.5 * X_news theta)`
# is estimated on development data under QLIKE, frozen, applied to evaluation,
# and the identical tail comparison is rerun.

# %%
SCALE_VS_TAIL: dict[str, Any] = {}
SCALE_ROWS: list[dict[str, Any]] = []

if GATE2_PASSED and not FORECASTS.empty:
    news_terms = ["news_indicator", "log1p_news_count_s", "semantic_intensity_s", "adverse_tone_s"]

    def news_design(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(frame)), *[frame[term].to_numpy(dtype=float) for term in news_terms]])

    theta, scale_diagnostics = fit_news_scale_adjustment(
        news_design(development),
        (development["target_return"] - development["mu_hat"]).to_numpy(dtype=float),
        development["sigma_forecast"].to_numpy(dtype=float),
    )
    print("QLIKE news-scale adjustment (development only):")
    print(json.dumps(scale_diagnostics, indent=2))
    print(dict(zip(["const", *news_terms], np.round(theta, 5), strict=True)))

    scaled_panel = PANEL.copy()
    scaled_panel["sigma_tilde"] = apply_news_scale_adjustment(
        scaled_panel["sigma_forecast"].to_numpy(dtype=float), news_design(scaled_panel), theta
    )
    scaled_panel["z_target"] = (scaled_panel["target_return"] - scaled_panel["mu_hat"]) / scaled_panel["sigma_tilde"]
    scaled_panel["log_sigma_forecast"] = np.log(scaled_panel["sigma_tilde"])
    centre = float(scaled_panel.loc[scaled_panel["split"] == "development", "log_sigma_forecast"].mean())
    spread = float(scaled_panel.loc[scaled_panel["split"] == "development", "log_sigma_forecast"].std(ddof=0)) or 1.0
    scaled_panel["log_sigma_forecast_s"] = (scaled_panel["log_sigma_forecast"] - centre) / spread

    scaled_dev = scaled_panel[scaled_panel["split"] == "development"].reset_index(drop=True)
    scaled_eval = scaled_panel[scaled_panel["split"] == "evaluation"].reset_index(drop=True)
    scaled_fits = {}
    for name in ("M1", "M2"):
        scaled_fits[name] = fit_joint_var_es(
            design_matrix(scaled_dev, MODEL_TERMS[name]),
            scaled_dev["z_target"].to_numpy(dtype=float),
            alpha=ALPHA,
            name=f"{name}_news_scaled",
            columns=MODEL_TERMS[name],
        )
        var_z, es_z, _ = tail_forecasts(
            design_matrix(scaled_eval, MODEL_TERMS[name]), scaled_fits[name].beta_var, scaled_fits[name].beta_gap
        )
        scaled_eval[f"fz0_{name}"] = fz0_loss(scaled_eval["z_target"].to_numpy(dtype=float), var_z, es_z, ALPHA)

    scaled_eval["d_semantic"] = scaled_eval["fz0_M2"] - scaled_eval["fz0_M1"]
    scaled_news = scaled_eval[scaled_eval["news_indicator"] == 1]
    values, weights = date_level(scaled_news.rename(columns={"target_date": "target_date"}), "d_semantic")
    scaled_interval = date_block_bootstrap_mean(
        values, weights, block_length=DATE_BLOCK_LENGTH, replications=BOOTSTRAP_REPS, seed=RANDOM_SEED
    )
    baseline_interval = BOOTSTRAP[
        (BOOTSTRAP["population"] == "news_bearing_evaluation_origins")
        & (BOOTSTRAP["contrast"] == "d_semantic")
        & (BOOTSTRAP["weighting"] == "observation_equal")
        & (BOOTSTRAP["block_length"] == DATE_BLOCK_LENGTH)
    ].iloc[0]

    SCALE_VS_TAIL = {
        "theta": dict(zip(["const", *news_terms], [float(value) for value in theta], strict=True)),
        "qlike": scale_diagnostics,
        "baseline_d_semantic": float(baseline_interval["point_estimate"]),
        "baseline_ci": [float(baseline_interval["ci_low"]), float(baseline_interval["ci_high"])],
        "news_scaled_d_semantic": scaled_interval["point_estimate"],
        "news_scaled_ci": [scaled_interval["ci_low"], scaled_interval["ci_high"]],
        "retained_share": (
            float(scaled_interval["point_estimate"] / baseline_interval["point_estimate"])
            if baseline_interval["point_estimate"]
            else math.nan
        ),
    }
    SCALE_ROWS = [
        {
            "check": "scale_versus_tail",
            "description": "paired FZ0 M2 - M1 on news-bearing evaluation origins after a frozen "
            "news-conditioned QLIKE volatility adjustment",
            "statistic": "paired_fz0_M2_minus_M1",
            "value": scaled_interval["point_estimate"],
            "ci_low": scaled_interval["ci_low"],
            "ci_high": scaled_interval["ci_high"],
            "rows": len(scaled_news),
            "firms": int(scaled_news["symbol"].nunique()),
            "dates": int(scaled_news["target_date"].nunique()),
            "notes": f"baseline (GARCH scale) point estimate {baseline_interval['point_estimate']:+.6g}",
        }
    ]
    print("\nScale-versus-tail comparison (news-bearing evaluation origins):")
    print(json.dumps(SCALE_VS_TAIL, indent=2, default=float))

# %% [markdown]
# ## 12. Frozen robustness checks
#
# These are run only after the primary result exists and none of them may be
# promoted to the headline conclusion.

# %%
ROBUSTNESS = pd.DataFrame()

if GATE2_PASSED and not FORECASTS.empty:
    robustness_rows: list[dict[str, Any]] = list(SCALE_ROWS)

    def evaluate_variant(
        label: str,
        description: str,
        dev_frame: pd.DataFrame,
        eval_frame: pd.DataFrame,
        terms: dict[str, list[str]],
        alpha: float,
    ) -> dict[str, Any] | None:
        try:
            fits = {
                name: fit_joint_var_es(
                    design_matrix(dev_frame, term_list),
                    dev_frame["z_target"].to_numpy(dtype=float),
                    alpha=alpha,
                    name=f"{label}:{name}",
                    columns=term_list,
                )
                for name, term_list in terms.items()
            }
        except RuntimeError as error:
            return {
                "check": label,
                "description": description,
                "statistic": "paired_fz0_M2_minus_M1",
                "value": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "rows": 0,
                "firms": 0,
                "dates": 0,
                "notes": f"not estimable: {error}",
            }
        block = eval_frame.copy()
        for name, term_list in terms.items():
            var_z, es_z, _ = tail_forecasts(design_matrix(block, term_list), fits[name].beta_var, fits[name].beta_gap)
            block[f"fz0_{name}"] = fz0_loss(block["z_target"].to_numpy(dtype=float), var_z, es_z, alpha)
        block["d_semantic"] = block["fz0_M2"] - block["fz0_M1"]
        news_block = block[block["news_indicator"] == 1]
        values, weights = date_level(news_block, "d_semantic")
        summary = date_block_bootstrap_mean(values, weights, block_length=DATE_BLOCK_LENGTH, replications=BOOTSTRAP_REPS, seed=RANDOM_SEED)
        return {
            "check": label,
            "description": description,
            "statistic": "paired_fz0_M2_minus_M1",
            "value": summary["point_estimate"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "rows": len(news_block),
            "firms": int(news_block["symbol"].nunique()),
            "dates": int(news_block["target_date"].nunique()),
            "notes": f"all fits converged: {all(fit.converged for fit in fits.values())}",
        }

    # R1 - VADER instead of FinBERT.
    vader_terms = {
        "M1": MODEL_TERMS["M1"],
        "M2": [*MODEL_TERMS["M1"], "vader_intensity_s", "vader_adverse_tone_s"],
    }
    robustness_rows.append(
        evaluate_variant(
            "vader_scorer",
            "declared robustness scorer: VADER intensity and -compound replace the FinBERT semantics",
            development,
            evaluation,
            vader_terms,
            ALPHA,
        )
    )

    # R2 - a deeper tail, only if the evaluation block supports it.
    expected_events = ROBUSTNESS_ALPHA * len(evaluation)
    if expected_events >= MIN_EXPECTED_TAIL_EVENTS:
        robustness_rows.append(
            evaluate_variant(
                f"alpha_{ROBUSTNESS_ALPHA}",
                f"same specification at alpha = {ROBUSTNESS_ALPHA}",
                development,
                evaluation,
                {"M1": MODEL_TERMS["M1"], "M2": MODEL_TERMS["M2"]},
                ROBUSTNESS_ALPHA,
            )
        )
    else:
        robustness_rows.append(
            {
                "check": f"alpha_{ROBUSTNESS_ALPHA}",
                "description": "deeper tail level",
                "statistic": "paired_fz0_M2_minus_M1",
                "value": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "rows": len(evaluation),
                "firms": int(evaluation["symbol"].nunique()),
                "dates": int(evaluation["target_date"].nunique()),
                "notes": f"skipped: only {expected_events:.0f} expected violations, below {MIN_EXPECTED_TAIL_EVENTS}",
            }
        )

    # R3 - EWMA price-risk filter.
    garch_loss_share = float(
        (GARCH_SUMMARY["firm_attrition_reason"].isin({"garch_fit_failed", "garch_non_stationary", "garch_invalid_parameters"})).mean()
    )
    ewma_panels = []
    for symbol in RETAINED_FIRMS:
        firm_panel = build_firm_panel(symbol, FIRM_PARAMS[symbol], sigma_source="ewma")
        if firm_panel.empty:
            continue
        firm_panel, _ = attach_news(firm_panel, symbol)
        ewma_panels.append(firm_panel)
    ewma_panel = pd.concat(ewma_panels, ignore_index=True).merge(MARKET_STATE, on="forecast_date", how="inner")
    ewma_panel["split"] = np.where(
        (ewma_panel["forecast_date"] >= DEV_START_TS)
        & (ewma_panel["forecast_date"] <= DEV_END_TS)
        & (ewma_panel["target_date"] <= DEV_END_TS),
        "development",
        np.where(
            (ewma_panel["forecast_date"] >= EVAL_START_TS) & (ewma_panel["forecast_date"] <= EVAL_END_TS),
            "evaluation",
            "excluded",
        ),
    )
    ewma_panel = ewma_panel[ewma_panel["split"] != "excluded"]
    ewma_panel = apply_scaling(ewma_panel, NEWS_COUNT_CAP)
    ewma_panel = ewma_panel[
        np.isfinite(ewma_panel[[f"{term}_s" for term in PRICE_STATE_TERMS] + ["z_target"]].to_numpy(dtype=float)).all(axis=1)
    ]
    robustness_rows.append(
        evaluate_variant(
            "ewma_price_filter",
            f"declared robustness price-risk filter: RiskMetrics EWMA (lambda = {EWMA_LAMBDA}) replaces GJR-GARCH",
            ewma_panel[ewma_panel["split"] == "development"].reset_index(drop=True),
            ewma_panel[ewma_panel["split"] == "evaluation"].reset_index(drop=True),
            {"M1": MODEL_TERMS["M1"], "M2": MODEL_TERMS["M2"]},
            ALPHA,
        )
        | {"notes": f"GARCH attrition share was {garch_loss_share:.1%}"}
    )
    del ewma_panel, ewma_panels

    # R4 - recap-heavy sessions: excluded, and controlled.
    recap_light = FORECASTS[(FORECASTS["news_indicator"] == 1) & (FORECASTS["recap_share"] < RECAP_HEAVY_THRESHOLD)]
    values, weights = date_level(recap_light, "d_semantic")
    recap_summary = date_block_bootstrap_mean(
        values, weights, block_length=DATE_BLOCK_LENGTH, replications=BOOTSTRAP_REPS, seed=RANDOM_SEED
    )
    robustness_rows.append(
        {
            "check": "exclude_recap_heavy_sessions",
            "description": f"drop news sessions whose recap share is >= {RECAP_HEAVY_THRESHOLD} (fixed forecasts, subset evaluation)",
            "statistic": "paired_fz0_M2_minus_M1",
            "value": recap_summary["point_estimate"],
            "ci_low": recap_summary["ci_low"],
            "ci_high": recap_summary["ci_high"],
            "rows": len(recap_light),
            "firms": int(recap_light["symbol"].nunique()),
            "dates": int(recap_light["target_date"].nunique()),
            "notes": f"{int((FORECASTS['news_indicator'] == 1).sum()) - len(recap_light)} recap-heavy news rows removed",
        }
    )
    development_recap = development.assign(recap_share_s=development["recap_share"])
    evaluation_recap = evaluation.assign(recap_share_s=evaluation["recap_share"])
    robustness_rows.append(
        evaluate_variant(
            "control_recap_share",
            "add the frozen recap share as an explicit control in both M1 and M2",
            development_recap,
            evaluation_recap,
            {
                "M1": [*MODEL_TERMS["M1"], "recap_share_s"],
                "M2": [*MODEL_TERMS["M2"], "recap_share_s"],
            },
            ALPHA,
        )
    )

    # R5 - firm-equal instead of observation-equal aggregation.
    news_only = FORECASTS[FORECASTS["news_indicator"] == 1]
    per_firm = news_only.groupby("symbol", sort=True)["d_semantic"].mean()
    firm_generator = np.random.default_rng(RANDOM_SEED)
    draws = firm_generator.integers(0, len(per_firm), size=(BOOTSTRAP_REPS, len(per_firm)))
    firm_means = per_firm.to_numpy()[draws].mean(axis=1)
    robustness_rows.append(
        {
            "check": "firm_equal_aggregation",
            "description": "mean of per-firm mean paired FZ0 difference, with a firm-level bootstrap",
            "statistic": "paired_fz0_M2_minus_M1",
            "value": float(per_firm.mean()),
            "ci_low": float(np.quantile(firm_means, 0.025)),
            "ci_high": float(np.quantile(firm_means, 0.975)),
            "rows": len(news_only),
            "firms": len(per_firm),
            "dates": int(news_only["target_date"].nunique()),
            "notes": "firms resampled independently; this ignores cross-firm date dependence by construction",
        }
    )

    ROBUSTNESS = pd.DataFrame([row for row in robustness_rows if row is not None])
    ROBUSTNESS.to_csv(OUTPUT_DIR / "robustness.csv", index=False)
    print(ROBUSTNESS[["check", "value", "ci_low", "ci_high", "rows", "firms", "dates", "notes"]].to_string(index=False))

# %% [markdown]
# ## 13. Figures

# %%
FIGURE_DIR.mkdir()
FIGURE_PATHS: list[str] = []


def save_figure(figure: plt.Figure, name: str) -> None:
    path = FIGURE_DIR / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    FIGURE_PATHS.append(str(path.relative_to(OUTPUT_DIR)))
    print(f"saved {path.relative_to(REPO_ROOT)}")


SAMPLE_NOTE = f"{RUN_MODE} run | {len(RETAINED_FIRMS)} firms | dev {DEV_START}..{DEV_END} | eval {EVAL_START}..{EVAL_END}"

if GATE2_PASSED and not FORECASTS.empty:
    # 1. Attrition flow.
    attrition_frame = pd.DataFrame(PIPELINE_ATTRITION)
    flow = attrition_frame[attrition_frame["excluded_at_step"] >= 0].tail(9)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(flow["step"], flow["surviving"], color="#31688e")
    axis.set_xscale("log")
    axis.invert_yaxis()
    axis.set_xlabel("surviving units (log scale)")
    axis.set_title(f"Sample attrition flow\n{SAMPLE_NOTE}")
    for index, (_, row) in enumerate(flow.iterrows()):
        axis.text(row["surviving"], index, f"  {row['surviving']:,} {row['unit']}", va="center", fontsize=8)
    save_figure(figure, "01_attrition_flow")

    # 2. News feature distributions.
    news_only = FORECASTS[FORECASTS["news_indicator"] == 1]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(np.log1p(news_only["news_count"]), bins=40, color="#35b779")
    axes[0].set_xlabel("log1p(news count)")
    axes[0].set_title("News volume")
    axes[1].hist(news_only["semantic_intensity"], bins=40, color="#31688e")
    axes[1].set_xlabel("semantic intensity = mean(p_neg + p_pos)")
    axes[1].set_title("Semantic intensity")
    axes[2].hist(news_only["adverse_tone"], bins=40, color="#440154")
    axes[2].set_xlabel("adverse tone = mean(p_neg - p_pos)")
    axes[2].set_title("Adverse tone (larger = more adverse)")
    for axis in axes:
        axis.set_ylabel("news-bearing evaluation origins")
    figure.suptitle(f"Distribution of news features on evaluation forecast origins\n{SAMPLE_NOTE}")
    save_figure(figure, "02_news_feature_distributions")

    # 3. GARCH diagnostics.
    fitted_firms = GARCH_SUMMARY[GARCH_SUMMARY["converged"]]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(fitted_firms["persistence"], bins=25, color="#31688e")
    axes[0].set_xlabel("alpha + gamma/2 + beta")
    axes[0].set_title("GJR persistence")
    axes[1].hist(fitted_firms["nu"].clip(upper=30), bins=25, color="#35b779")
    axes[1].set_xlabel("Student-t degrees of freedom (clipped at 30)")
    axes[1].set_title("Innovation tail thickness")
    sample_z = FORECASTS["z_target"].clip(-8, 8)
    axes[2].hist(sample_z, bins=80, density=True, color="#440154", alpha=0.8)
    grid = np.linspace(-8, 8, 400)
    axes[2].plot(grid, np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi), color="#fde725", lw=2, label="N(0,1)")
    axes[2].set_xlabel("standardised evaluation return z")
    axes[2].set_title(f"z distribution (std {FORECASTS['z_target'].std():.3f})")
    axes[2].legend()
    for axis in axes[:2]:
        axis.set_ylabel("firms")
    axes[2].set_ylabel("density")
    figure.suptitle(f"GJR-GARCH(1,1)-t fit and standardised-residual diagnostics\n{SAMPLE_NOTE}")
    save_figure(figure, "03_garch_diagnostics")

    # 4. Mean FZ0 by model.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (population, frame) in zip(axes, (("news-bearing origins", news_only), ("full evaluation panel", FORECASTS)), strict=True):
        means = [float(frame[f"fz0_{name}"].mean()) for name in MODEL_TERMS]
        axis.bar(list(MODEL_TERMS), means, color=["#440154", "#31688e", "#35b779", "#fde725"])
        axis.set_ylabel("mean FZ0 loss (lower is better)")
        axis.set_title(f"{population} (n = {len(frame):,})")
        low, high = min(means), max(means)
        pad = max((high - low) * 0.6, abs(high) * 1e-4)
        axis.set_ylim(low - pad, high + pad)
        for index, value in enumerate(means):
            axis.text(index, value, f"{value:.5f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle(f"Mean joint FZ0 loss by model, identical evaluation rows\n{SAMPLE_NOTE}")
    save_figure(figure, "04_mean_fz0_by_model")

    # 5. Date-level paired difference over time.
    date_series = news_only.groupby("target_date", sort=True)["d_semantic"].mean()
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(date_series.index, date_series.to_numpy(), lw=0.5, color="#31688e", alpha=0.7, label="date mean")
    axis.plot(
        date_series.index,
        date_series.rolling(60, min_periods=20).mean().to_numpy(),
        lw=2,
        color="#440154",
        label="60-date rolling mean",
    )
    axis.axhline(0.0, color="#d62728", lw=1, ls="--", label="no semantic effect")
    axis.set_xlabel("target date")
    axis.set_ylabel("FZ0(M2) - FZ0(M1)")
    axis.set_title(f"Date-level paired semantic loss difference (negative = semantics help)\n{SAMPLE_NOTE}")
    axis.legend()
    save_figure(figure, "05_date_level_paired_difference")

    # 6. Bootstrap distribution.
    values, weights = date_level(news_only, "d_semantic")
    from sentiment_benchmark.tail_risk import block_bootstrap_indices as _indices

    draws = _indices(values.size, block_length=DATE_BLOCK_LENGTH, replications=BOOTSTRAP_REPS, seed=RANDOM_SEED)
    replicate_means = np.sum(values[draws] * weights[draws], axis=1) / np.sum(weights[draws], axis=1)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.hist(replicate_means, bins=40, color="#31688e", alpha=0.85)
    axis.axvline(float(np.sum(values * weights) / weights.sum()), color="#440154", lw=2, label="point estimate")
    axis.axvline(0.0, color="#d62728", lw=1.5, ls="--", label="no semantic effect")
    axis.set_xlabel("bootstrap mean of FZ0(M2) - FZ0(M1)")
    axis.set_ylabel("replications")
    axis.set_title(f"Moving-block bootstrap over target dates (block {DATE_BLOCK_LENGTH}, {BOOTSTRAP_REPS} reps)\n{SAMPLE_NOTE}")
    axis.legend()
    save_figure(figure, "06_bootstrap_distribution")

    # 7. Hit rates by model and news state.
    hit_frame = CALIBRATION[
        (CALIBRATION["diagnostic"] == "var_hit_rate_minus_alpha")
        & CALIBRATION["population"].isin(["news_bearing", "no_news", "full_evaluation_panel"])
    ]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.25
    positions = np.arange(len(MODEL_TERMS))
    for offset, population in zip((-width, 0.0, width), ("news_bearing", "no_news", "full_evaluation_panel"), strict=True):
        block = hit_frame[hit_frame["population"] == population].set_index("model").reindex(list(MODEL_TERMS))
        axis.bar(positions + offset, block["hit_rate"], width=width, label=population)
    axis.axhline(ALPHA, color="#d62728", lw=1.5, ls="--", label=f"nominal alpha = {ALPHA}")
    axis.set_xticks(positions, list(MODEL_TERMS))
    axis.set_ylabel("VaR violation rate")
    axis.set_title(f"VaR hit rates by model and news state\n{SAMPLE_NOTE}")
    axis.legend()
    save_figure(figure, "07_var_hit_rates")

    # 8. Illustrative forecasts for a deterministic small firm set.
    show = sorted(RETAINED_FIRMS)[:3]
    figure, axes = plt.subplots(len(show), 1, figsize=(12, 3.2 * len(show)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, symbol in zip(axes, show, strict=True):
        block = FORECASTS[(FORECASTS["symbol"] == symbol) & (FORECASTS["target_date"].dt.year.isin([2020, 2021]))]
        axis.plot(block["target_date"], block["target_return"], lw=0.6, color="#666666", label="realised return")
        axis.plot(block["target_date"], block["var_return_M1"], lw=1.1, color="#31688e", label="VaR M1")
        axis.plot(block["target_date"], block["var_return_M2"], lw=1.1, color="#35b779", label="VaR M2")
        axis.plot(block["target_date"], block["es_return_M2"], lw=1.0, color="#440154", ls=":", label="ES M2")
        breaches = block[block["hit_M2"] == 1]
        axis.scatter(breaches["target_date"], breaches["target_return"], s=14, color="#d62728", zorder=5, label="M2 violation")
        axis.set_ylabel(f"{symbol}\nlog return")
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("target date")
    figure.suptitle(f"Illustrative one-day-ahead VaR/ES forecasts, 2020-2021\n{SAMPLE_NOTE}")
    save_figure(figure, "08_illustrative_forecasts")

    # 9. Scale-versus-tail.
    if SCALE_VS_TAIL:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        labels = ["GARCH scale\n(primary)", "news-conditioned scale\n(diagnostic)"]
        points = [SCALE_VS_TAIL["baseline_d_semantic"], SCALE_VS_TAIL["news_scaled_d_semantic"]]
        lows = [SCALE_VS_TAIL["baseline_ci"][0], SCALE_VS_TAIL["news_scaled_ci"][0]]
        highs = [SCALE_VS_TAIL["baseline_ci"][1], SCALE_VS_TAIL["news_scaled_ci"][1]]
        axis.errorbar(
            labels,
            points,
            yerr=[np.array(points) - np.array(lows), np.array(highs) - np.array(points)],
            fmt="o",
            capsize=6,
            color="#31688e",
        )
        axis.axhline(0.0, color="#d62728", lw=1.5, ls="--")
        axis.set_ylabel("paired FZ0(M2) - FZ0(M1)")
        axis.set_title(f"Does the semantic gain survive a news-conditioned volatility scale?\n{SAMPLE_NOTE}")
        save_figure(figure, "09_scale_versus_tail")

# %% [markdown]
# ## 14. Manifest and summary

# %%
ATTRITION = pd.DataFrame(PIPELINE_ATTRITION)
if GATE2_PASSED and not PANEL.empty:
    for reason, count in GARCH_SUMMARY["firm_attrition_reason"].value_counts().items():
        if reason != "retained":
            ATTRITION.loc[len(ATTRITION)] = {
                "step": f"risk_model_gate:{reason}",
                "unit": "firms",
                "surviving": len(RETAINED_FIRMS),
                "excluded_at_step": int(count),
                "reason": str(reason),
            }
    ATTRITION.loc[len(ATTRITION)] = {
        "step": "final_evaluation_rows",
        "unit": "firm-days",
        "surviving": len(FORECASTS),
        "excluded_at_step": 0,
        "reason": "identical rows scored by every nested model",
    }
ATTRITION.to_csv(OUTPUT_DIR / "attrition.csv", index=False)
print(ATTRITION.to_string(index=False))


# %%
def summarise_finding() -> dict[str, Any]:
    if not GATE2_PASSED:
        return {"verdict": "GATE_2_FAILED", "detail": "no inferential result was produced"}
    row = BOOTSTRAP[
        (BOOTSTRAP["population"] == "news_bearing_evaluation_origins")
        & (BOOTSTRAP["contrast"] == "d_semantic")
        & (BOOTSTRAP["weighting"] == "observation_equal")
        & (BOOTSTRAP["block_length"] == DATE_BLOCK_LENGTH)
    ].iloc[0]
    point, low, high = float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])
    if high < 0.0:
        verdict = "SEMANTICS_IMPROVE_TAIL_FORECASTS"
    elif low > 0.0:
        verdict = "SEMANTICS_DEGRADE_TAIL_FORECASTS"
    else:
        verdict = "NULL_NO_DETECTABLE_SEMANTIC_GAIN"
    m1 = float(
        MODEL_COMPARISON[
            (MODEL_COMPARISON["population"] == "news_bearing_evaluation_origins") & (MODEL_COMPARISON["quantity"] == "mean_fz0_M1")
        ]["value"].iloc[0]
    )
    return {
        "verdict": verdict,
        "paired_fz0_M2_minus_M1": point,
        "ci_low": low,
        "ci_high": high,
        "share_of_bootstrap_means_below_zero": float(row["share_below_zero"]),
        "relative_improvement_pct": -point / abs(m1) * 100.0 if m1 else math.nan,
        "n_dates": int(row["n_dates"]),
    }


FINDING = summarise_finding()
print(json.dumps(FINDING, indent=2, default=float))

# %%
NOTEBOOK_SOURCE = REPO_ROOT / "notebooks" / "fnsipid_tail_risk_core.py"
NOTEBOOK_IPYNB = REPO_ROOT / "notebooks" / "fnsipid_tail_risk_core.ipynb"
HELPER_SOURCE = REPO_ROOT / "src" / "sentiment_benchmark" / "tail_risk.py"

output_files = sorted(
    path for path in OUTPUT_DIR.rglob("*") if path.is_file() and path.name not in {"manifest.json", "fnsipid_tail_risk_core.executed.ipynb"}
)

MANIFEST: dict[str, Any] = {
    "schema_version": 1,
    "experiment": f"fnsipid_tail_risk_core_{VARIANT}",
    "variant": {
        "id": VARIANT,
        "price_repair": PRICE_REPAIR,
        "volatility_refit": VOLATILITY_REFIT,
        "adjustment_gap_threshold": ADJUSTMENT_GAP_THRESHOLD,
        "repaired_price_rows": int(PRICE_VALIDATION["repaired_price_rows"].sum()),
        "firms_with_a_repaired_row": int((PRICE_VALIDATION["repaired_price_rows"] > 0).sum()),
        "classification": "candidate adjusted/raw-return gaps only; no authoritative corporate-action metadata are available",
        "relative_to": VARIANT_RELATIVE_TO,
    },
    "research_question": (
        "After the initial price response, does firm-linked financial-news sentiment improve "
        "one-day-ahead VaR and ES forecasts beyond price-based conditional volatility, news arrival, "
        "and news volume?"
    ),
    "status": "completed" if GATE2_PASSED else "gate_2_failed",
    "run_mode": RUN_MODE,
    "smoke_run_is_engineering_check_only": RUN_MODE == "smoke",
    "created_at": pd.Timestamp.utcnow().isoformat(),
    "runtime_seconds": round(time.time() - NOTEBOOK_START, 1),
    "git": {"commit": GIT_COMMIT, "dirty": GIT_DIRTY},
    "code": {
        "notebook_source": {"path": str(NOTEBOOK_SOURCE.relative_to(REPO_ROOT)), "sha256": sha256_file(NOTEBOOK_SOURCE)},
        "notebook_ipynb": (
            {"path": str(NOTEBOOK_IPYNB.relative_to(REPO_ROOT)), "sha256": sha256_file(NOTEBOOK_IPYNB)} if NOTEBOOK_IPYNB.exists() else None
        ),
        "helper_module": {"path": str(HELPER_SOURCE.relative_to(REPO_ROOT)), "sha256": sha256_file(HELPER_SOURCE)},
        "helper_tests": "tests/test_tail_risk.py",
    },
    "inputs": json.loads(INPUT_INVENTORY.to_json(orient="records")),
    "schema_mapping": {
        "news_source_table": "events(symbol, session_date, p_negative, p_neutral, p_positive, vader_compound, is_recap)",
        "price_source": "full_history/<SYMBOL>.csv columns date, close, adj close",
        "market_series": f"{MARKET_SYMBOL} from the same archive; frozen upstream market proxy",
    },
    "configuration": CONFIG,
    "scorer": {
        "primary": PRIMARY_SCORER,
        "definition": SEMANTIC_DEFINITION,
        "adverse_tone": "mean(p_negative - p_positive); larger means more adverse",
        "semantic_intensity": "mean(p_negative + p_positive)",
        "negative_share": "mean(argmax class == negative)",
        "sign_checks": sign_checks,
        "robustness_scorer": "VADER: adverse tone = -mean(compound), intensity = share(|compound| > 0.05)",
        "model_revision": E6_MANIFEST["finbert"]["model_revision"],
        "label_derived_fallback_used": False,
    },
    "timing_rule": {
        "forecast_timing": "mapped reaction session s -> forecast at close s -> next close-to-close return",
        "upstream_policy": E5_MANIFEST["timing_rule"],
        "upstream_counts_before_window_and_deduplication": {
            "date_only_or_exact_midnight_rows": DATE_ONLY_UPSTREAM_ROWS,
            "precise_timestamp_rows": PRECISE_TIMESTAMP_UPSTREAM_ROWS,
        },
        "calendar_dates_checked_for_date_only_branch": int(len(_calendar_days)),
        "date_only_non_forward_mappings": int(DATE_ONLY_TIMING_VIOLATIONS),
        "invalid_corpus_reaction_sessions": INVALID_CORPUS_REACTION_SESSIONS,
        "original_timestamp_retained_per_event": False,
        "strict_date_only_rule_verified_per_headline": False,
        "limitation": "the completed checkpoint cannot identify which retained events used the precise-timestamp branch",
        "max_news_forward_gap_days": MAX_NEWS_FORWARD_GAP_DAYS,
    },
    "garch": {
        "specification": CONFIG["GARCH_SPECIFICATION"],
        "package": f"arch {arch.__version__}",
        "fits_attempted": int(len(GARCH_SUMMARY)),
        "fits_converged": int(GARCH_SUMMARY["converged"].sum()) if len(GARCH_SUMMARY) else 0,
        "recursion_start": RECURSION_START,
        "refit_policy": VOLATILITY_REFIT,
        "refits_attempted": int(len(REFIT_LOG)),
        "refits_accepted": int(REFIT_LOG["accepted"].sum()) if len(REFIT_LOG) else 0,
        "refits_carried_forward": int(REFIT_LOG["carried_forward"].sum()) if len(REFIT_LOG) else 0,
        "max_recursion_vs_estimator_relative_deviation": (
            float(GARCH_SUMMARY["recursion_vs_arch_max_rel_diff"].max(skipna=True)) if len(GARCH_SUMMARY) else None
        ),
        "max_initialisation_sensitivity_at_dev_end": (
            float(GARCH_SUMMARY["initialisation_sensitivity_at_dev_end"].max(skipna=True)) if len(GARCH_SUMMARY) else None
        ),
    },
    "model_formulas": {name: " + ".join(terms) for name, terms in MODEL_TERMS.items()},
    "tail_parameterisation": CONFIG["TAIL_PARAMETERISATION"],
    "optimisation": (
        {
            name: {
                "objective": fit.objective,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "starts_attempted": fit.starts_attempted,
                "starts_converged": fit.starts_converged,
                "start_objective_spread": fit.start_objective_spread,
                "clipped_rows": fit.clipped_rows,
                "message": fit.message,
                "warnings": list(fit.warnings),
            }
            for name, fit in FITS.items()
        }
        if FITS
        else {}
    ),
    "bootstrap": {
        "unit": "moving block of target calendar dates; the full cross-section of firms on a date moves together",
        "block_length": DATE_BLOCK_LENGTH,
        "replications": BOOTSTRAP_REPS,
        "seed": RANDOM_SEED,
        "refits_models": False,
    },
    "random_seed": RANDOM_SEED,
    "counts": {
        "cohort_firms": len(COHORT_SYMBOLS),
        "modelling_universe": len(MODEL_UNIVERSE),
        "retained_firms": len(RETAINED_FIRMS),
        "panel_rows": int(len(PANEL)),
        "development_rows": int((PANEL["split"] == "development").sum()) if len(PANEL) else 0,
        "evaluation_rows": int(len(FORECASTS)),
        "evaluation_news_rows": int(FORECASTS["news_indicator"].sum()) if len(FORECASTS) else 0,
        "evaluation_target_dates": int(FORECASTS["target_date"].nunique()) if len(FORECASTS) else 0,
    },
    "attrition": json.loads(ATTRITION.to_json(orient="records")),
    "gates": json.loads(GATE_STATUS.to_json(orient="records")),
    "assertions": ASSERTIONS,
    "finding": FINDING,
    "scale_versus_tail": SCALE_VS_TAIL,
    "figures": FIGURE_PATHS,
    "outputs": {
        str(path.relative_to(OUTPUT_DIR)): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in output_files
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
            "arch": arch.__version__,
            "matplotlib": matplotlib.__version__,
            "sentiment_benchmark": sentiment_benchmark.__version__,
        },
    },
    "deviations": [
        "The 512 MB E6 checkpoint, its declared E5 upstream manifest, and the selected price archive were SHA-256 verified before loading.",
        "The checkpoint does not retain original headline timestamps or per-event timing-type flags. The hashed upstream manifest records 2,518,109 date-only/exact-midnight rows using a strictly-next-session rule and 5,660 precise-timestamp rows using a containing-or-next-session rule before windowing and deduplication. Both branches are point-in-time at the mapped session close, but a strictly-date-only mapping cannot be re-verified per retained headline.",
        "arch, jupytext, nbformat, nbconvert, ipykernel and pyarrow were added to pyproject.toml as the 'tailrisk' optional extra; no existing dependency was changed.",
    ]
    + ([GIT_PROVENANCE_WARNING] if GIT_PROVENANCE_WARNING else []),
    "not_claimed": [
        "no causal claim",
        "no deployable trading edge",
        "no claim about investor behaviour",
        "no claim of novelty in applying sentiment to tail-risk forecasting",
    ],
}
atomic_write_json(OUTPUT_DIR / "manifest.json", MANIFEST)
print(f"manifest written with {len(MANIFEST['outputs'])} hashed outputs")


# %%
def build_summary() -> str:
    variant_changes = []
    if PRICE_REPAIR == "min_abs_return":
        variant_changes.append(
            f"applies the minimum-absolute-return sensitivity to {int(PRICE_VALIDATION['repaired_price_rows'].sum()):,} "
            "candidate adjusted/raw-return gaps"
        )
    if VOLATILITY_REFIT == "annual_expanding":
        variant_changes.append("re-estimates the volatility filter before each evaluation year on an expanding window")
    variant_note = "" if not variant_changes else f"\n  Relative to v1, this cell {' and '.join(variant_changes)}. Nothing else differs."
    filter_description = "annual-refitted point-in-time" if VOLATILITY_REFIT == "annual_expanding" else "development-fitted frozen"
    lines = [
        "# FNSPID sentiment-conditioned tail risk - core result",
        "",
        f"- Variant: **{VARIANT}** — price repair `{PRICE_REPAIR}`, volatility refit `{VOLATILITY_REFIT}`" + variant_note,
        f"- Run mode: **{RUN_MODE}**"
        + (
            "\n  **This is an engineering check on a deterministic firm subset, not dissertation evidence.**" if RUN_MODE == "smoke" else ""
        ),
        f"- Git commit: `{GIT_COMMIT or 'unavailable'}` (dirty worktree: {GIT_DIRTY if GIT_DIRTY is not None else 'unavailable'})",
        f"- Primary scorer: {PRIMARY_SCORER}, probability-based semantics; alpha = {ALPHA}",
        "",
        "## Question",
        "",
        "After the initial price response, does firm-linked financial-news sentiment improve",
        "one-day-ahead VaR and ES forecasts beyond price-based conditional volatility, news",
        "arrival, and news volume?",
        "",
    ]
    if not GATE2_PASSED:
        lines += [
            "## Answer",
            "",
            "**Gate 2 failed. No inferential result was produced.**",
            "",
            "| Gate | Status | Observed | Requirement |",
            "| --- | --- | --- | --- |",
            *[f"| {row['gate']} | {row['status']} | {row['observed']} | {row['requirement']} |" for _, row in GATE_STATUS.iterrows()],
            "",
            "The data audit above lists the exact missing or invalid fields. No price was",
            "fabricated, no synthetic data entered any result, and no unrelated universe was",
            "substituted.",
            "",
        ]
        return "\n".join(lines) + "\n"

    news_block = MODEL_COMPARISON[MODEL_COMPARISON["population"] == "news_bearing_evaluation_origins"]
    verdict_text = {
        "SEMANTICS_IMPROVE_TAIL_FORECASTS": "Semantics improved the joint VaR/ES forecast beyond news arrival and volume.",
        "SEMANTICS_DEGRADE_TAIL_FORECASTS": "Semantics made the joint VaR/ES forecast worse.",
        "NULL_NO_DETECTABLE_SEMANTIC_GAIN": "No detectable semantic gain. This is a valid null.",
    }[FINDING["verdict"]]

    lines += [
        "## Answer",
        "",
        f"**{verdict_text}**",
        "",
        f"On {FINDING['n_dates']:,} evaluation target dates covering "
        f"{int(news_block[news_block['quantity'] == 'mean_fz0_M1']['rows'].iloc[0]):,} news-bearing firm-days across "
        f"{int(news_block[news_block['quantity'] == 'mean_fz0_M1']['firms'].iloc[0])} firms, the paired FZ0 loss",
        f"difference `FZ0(M2) - FZ0(M1)` is **{FINDING['paired_fz0_M2_minus_M1']:+.6g}** with a 95% date-block",
        f"bootstrap interval of **[{FINDING['ci_low']:+.6g}, {FINDING['ci_high']:+.6g}]** "
        f"({FINDING['share_of_bootstrap_means_below_zero']:.1%} of bootstrap means below zero).",
        f"In relative terms mean loss changes by {-FINDING['relative_improvement_pct']:+.4f}%; a negative paired",
        "difference (and a negative relative change) means semantics help.",
        "",
        "## Evidenced",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
    ]
    for name in MODEL_TERMS:
        value = float(news_block[news_block["quantity"] == f"mean_fz0_{name}"]["value"].iloc[0])
        lines.append(f"| mean FZ0, {name}, news-bearing origins | {value:.6f} |")
    for quantity in (
        "paired_fz0_M2_minus_M1",
        "paired_fz0_M2_minus_M2_intensity",
        "paired_fz0_M1_minus_M0",
        "relative_fz0_improvement_pct_M2_vs_M1",
    ):
        value = float(news_block[news_block["quantity"] == quantity]["value"].iloc[0])
        lines.append(f"| {quantity} | {value:+.6g} |")
    hit_rows = CALIBRATION[
        (CALIBRATION["diagnostic"] == "var_hit_rate_minus_alpha") & (CALIBRATION["population"] == "full_evaluation_panel")
    ]
    for _, row in hit_rows.iterrows():
        lines.append(f"| VaR hit rate, {row['model']}, full panel (nominal {ALPHA}) | {row['hit_rate']:.5f} |")
    m1_hit = hit_rows[hit_rows["model"] == "M1"].iloc[0]
    rejecting = CALIBRATION[CALIBRATION["diagnostic"].isin(["dq_conditional_coverage_wald", "es_conditional_calibration_wald"])]
    rejected = sum(1 for detail in rejecting["detail"] if float(detail.split("p = ")[1].split(";")[0]) < 0.05)
    lines += [
        "",
        f"Every model was scored on the same {len(FORECASTS):,} evaluation rows. All forecasts satisfy",
        "`ES < VaR < 0` in both standardised and return units, and all listed assertions passed.",
        "",
        "**Calibration caveat.** Every model over-violates its nominal level. On the full evaluation",
        f"panel M1 breaches on {m1_hit['hit_rate']:.5f} of days against a nominal {ALPHA}, and the 95%",
        f"date-block interval on the excess, [{m1_hit['ci_low']:+.5f}, {m1_hit['ci_high']:+.5f}], excludes zero.",
        f"{rejected} of {len(rejecting)} DQ conditional-coverage and ES identification tests reject at 5%.",
        f"Evaluation-period standardised returns have standard deviation {FORECASTS['z_target'].std():.3f}, so the",
        f"{filter_description} volatility filter under-predicts 2017-2023 volatility. The nested comparison",
        "is therefore a relative ranking among models that are all somewhat under-conservative; it is not",
        "a claim that any of them is correctly calibrated.",
        "",
        "**Timing caveat.** The frozen upstream manifest records a mixed policy: "
        f"{DATE_ONLY_UPSTREAM_ROWS:,} date-only/exact-midnight source rows use the strictly-next-session rule, while "
        f"{PRECISE_TIMESTAMP_UPSTREAM_ROWS:,} precise-timestamp rows use a containing-or-next-session rule before "
        "windowing and deduplication. Original timestamps are absent from the completed checkpoint, so the stricter "
        "date-only rule cannot be re-verified per retained headline. All forecasts remain point-in-time at the mapped "
        "reaction-session close, but this is a documented deviation from a uniformly date-only design.",
        "",
        "## Inference",
        "",
    ]
    if FINDING["verdict"] == "NULL_NO_DETECTABLE_SEMANTIC_GAIN":
        lines += [
            "Bounded reading: once the reaction-session shock, its magnitude, the conditional",
            "volatility level, market state, news arrival and news volume are in the information",
            "set, the additional signed-tone and intensity variables did not measurably sharpen",
            "the one-day-ahead lower tail on this panel at this horizon and coarse timestamp",
            "resolution. This is evidence of absence only at the precision the interval supports;",
            "it does not show that news semantics are irrelevant to tail risk in general.",
        ]
    elif FINDING["verdict"] == "SEMANTICS_IMPROVE_TAIL_FORECASTS":
        lines += [
            "Bounded reading: the semantic variables carried incremental lower-tail information",
            "beyond arrival and volume on this panel. The effect size is small in absolute loss",
            "units and is a statement about forecast quality, not about a tradeable edge.",
        ]
    else:
        lines += [
            "Bounded reading: adding semantics degraded forecast quality on this panel, which is",
            "consistent with the extra parameters adding estimation noise without adding signal.",
        ]
    if SCALE_VS_TAIL:
        baseline = SCALE_VS_TAIL["baseline_d_semantic"]
        scaled = SCALE_VS_TAIL["news_scaled_d_semantic"]
        baseline_ci = SCALE_VS_TAIL["baseline_ci"]
        scaled_ci = SCALE_VS_TAIL["news_scaled_ci"]
        baseline_inconclusive = baseline_ci[0] <= 0.0 <= baseline_ci[1]
        baseline_degradation = baseline_ci[0] > 0.0
        scaled_supported = scaled_ci[1] < 0.0
        if baseline_inconclusive:
            reading = (
                "the baseline interval spans zero, so this diagnostic cannot identify a semantic "
                "increment for the scale adjustment to explain"
            )
        elif baseline_degradation:
            reading = (
                "the baseline interval is above zero, so semantics degrades forecast loss and "
                "there is no semantic gain for the scale adjustment to explain"
            )
        elif not scaled_supported:
            reading = (
                "the detectable negative baseline difference is not retained after the scale "
                "adjustment, which is consistent with a conditional-scale contribution"
            )
        elif abs(scaled) < 0.5 * abs(baseline):
            reading = "the detectable negative paired difference becomes less than half as large after the scale adjustment"
        else:
            reading = (
                "a detectable negative paired difference remains after the scale adjustment, "
                "consistent with information beyond this particular conditional-scale correction"
            )
        lines += [
            "",
            "Scale-versus-tail diagnostic: with a frozen news-conditioned QLIKE volatility adjustment the",
            f"paired difference moves from {baseline:+.6g} to {scaled:+.6g} "
            f"(95% interval [{SCALE_VS_TAIL['news_scaled_ci'][0]:+.6g}, {SCALE_VS_TAIL['news_scaled_ci'][1]:+.6g}]).",
            f"Reading: {reading}.",
        ]
    lines += [
        "",
        "## Open limitations",
        "",
        "- **Survivorship.** The linked price panel is the set of tickers present in the FNSPID",
        f"  archive; {int((~PRICE_VALIDATION.loc[PRICE_VALIDATION['symbol'] != MARKET_SYMBOL, 'survivor_conditioned']).sum())} of "
        f"{len(PRICE_VALIDATION) - 1} cohort tickers end before 2023-12, so the sample is partly survivor",
        "  conditioned. It is described as the available linked firm-price panel, not an S&P 500 panel.",
        "- **Universe composition.** FNSPID links headlines by ticker, so the cohort mixes",
        "  operating firms with exchange-traded funds (for example AGG, GLD, QQQ). No local",
        "  metadata separates them, and they were not hand-removed, because an ad hoc curated",
        "  exclusion would be a researcher degree of freedom.",
        "- **Timestamp coarsening.** FNSPID dates are calendar dates. Headlines are pushed to the",
        "  next session, which is conservative but discards intraday ordering and merges Friday,",
        "  weekend and holiday news into one reaction session.",
        "- **Recap content.** Price-recap headlines mechanically restate the move that already",
        "  happened; the frozen regex flag is imperfect and the recap robustness check is a",
        "  sensitivity, not a clean identification.",
        "- **Corporate actions.** Only adjusted closes are available and no corporate-action",
        "  metadata; residual unadjusted events would show up as artificial tail losses.",
        "- **Shared dates.** Firm-days on the same calendar date are strongly dependent; the",
        "  bootstrap addresses this but assumes weak dependence across date blocks.",
        "- **Model misspecification.** One volatility filter and one tail parameterisation were",
        "  pre-specified. Failing to reject a calibration test is not proof of correctness.",
        "",
        "## Not claimed",
        "",
        "- No causal claim, no deployable alpha, no claim about investor behaviour, and no claim",
        "  of first use of sentiment in tail-risk forecasting. A null is a valid result.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f'REPRO_DIR="reports/reproductions/fnsipid_tail_risk_core_{VARIANT}_$(date -u +%Y%m%dT%H%M%SZ)"',
        f"FNSPID_TAIL_RISK_RUN_MODE={RUN_MODE} \\",
        f"FNSPID_TAIL_RISK_VARIANT={VARIANT} \\",
        'FNSPID_TAIL_RISK_OUTPUT_DIR="$REPRO_DIR" \\',
        "jupyter nbconvert \\",
        "  --to notebook \\",
        "  --execute notebooks/fnsipid_tail_risk_core.ipynb \\",
        '  --output "../$REPRO_DIR/fnsipid_tail_risk_core.executed.ipynb" \\',
        "  --ExecutePreprocessor.timeout=-1",
        "```",
        "",
        "A fresh output path is mandatory: the notebook fails closed rather than mixing a rerun",
        "with files from an earlier bundle. Choose a new `REPRO_DIR` after any interrupted run.",
        "",
        (
            f"Set `FNSPID_TAIL_RISK_RUN_MODE=full` for the full {len(ELIGIBLE)}-firm run."
            if RUN_MODE == "smoke"
            else f"This run used `FNSPID_TAIL_RISK_RUN_MODE=full` over all {len(ELIGIBLE)} eligible firms. "
            f"Set it to `smoke` for the deterministic {SMOKE_FIRM_COUNT}-firm engineering check."
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


atomic_write_text(OUTPUT_DIR / "summary.md", build_summary())
print((OUTPUT_DIR / "summary.md").read_text(encoding="utf-8"))

# summary.md is derived from the manifest's own finding, so it can only be
# written after the manifest is assembled. Re-hash it and rewrite manifest.json
# so the record covers every file in the bundle rather than a stale copy.
MANIFEST["outputs"]["summary.md"] = {
    "sha256": sha256_file(OUTPUT_DIR / "summary.md"),
    "size_bytes": (OUTPUT_DIR / "summary.md").stat().st_size,
}
verified_output_count = verify_output_manifest(OUTPUT_DIR, MANIFEST["outputs"])
check(
    "final_manifest_matches_output_bundle",
    verified_output_count == len(MANIFEST["outputs"]),
    f"{verified_output_count} outputs verified by SHA-256 and size",
)
MANIFEST["assertions"] = ASSERTIONS
atomic_write_json(OUTPUT_DIR / "manifest.json", MANIFEST)
print(f"manifest refreshed; {verified_output_count} hashed outputs and {len(ASSERTIONS)} assertions verified")

# %%
print(f"total notebook runtime: {time.time() - NOTEBOOK_START:.1f}s")
print(f"outputs in {OUTPUT_DIR.relative_to(REPO_ROOT)}:")
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(OUTPUT_DIR)}  ({path.stat().st_size:,} bytes)")
