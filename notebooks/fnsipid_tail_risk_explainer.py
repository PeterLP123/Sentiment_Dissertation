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
# # What did we actually find? A visual read of the FNSPID tail-risk result
#
# This notebook **explains** the frozen experiment in
# `reports/fnsipid_tail_risk_core_v1/`. It reads that run's artifacts, verifies
# their hashes against the run manifest, and draws them. It **re-estimates
# nothing** — no GARCH fit, no tail model, no new specification. Where a number
# is recomputed (the bootstrap replicate distribution), it is asserted equal to
# the frozen artifact before it is plotted.
#
# So nothing here can quietly disagree with the result it is describing.
#
# ## The question
#
# > After the initial price response, does firm-linked financial-news sentiment
# > improve one-day-ahead Value-at-Risk and Expected Shortfall forecasts beyond
# > price-based conditional volatility, news arrival, and news volume?
#
# ## The short answer, in one line
#
# **News *arrival* helps a lot. News *sentiment*, on top of arrival, does not
# measurably help at all.** The rest of this notebook shows why that is a
# credible null rather than a failure to look properly.

# %% [markdown]
# ## 1. Setup and provenance

# %%
# Long prose strings and annotation text intentionally stay on one line.
# ruff: noqa: E501
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "pyproject.toml").exists():
    raise RuntimeError("run this notebook from the repository root or from notebooks/")
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sentiment_benchmark.artifact_io import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from sentiment_benchmark.tail_risk import (  # noqa: E402
    block_bootstrap_indices,
    date_block_bootstrap_mean,
    verify_input_file,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - keeps the .py runnable outside a kernel

    def display(obj: Any) -> None:
        return None


CORE_DIR = Path(os.environ.get("FNSPID_TAIL_RISK_CORE_DIR", "") or REPO_ROOT / "reports" / "fnsipid_tail_risk_core_v1").expanduser()
OUTPUT_DIR = Path(
    os.environ.get("FNSPID_TAIL_RISK_EXPLAINER_OUTPUT_DIR", "") or REPO_ROOT / "reports" / "fnsipid_tail_risk_explainer_v1"
).expanduser()
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

if not (CORE_DIR / "manifest.json").exists():
    raise FileNotFoundError(
        "the frozen core run is missing. Produce it first with:\n"
        "  FNSPID_TAIL_RISK_RUN_MODE=full jupyter nbconvert --to notebook --execute "
        "notebooks/fnsipid_tail_risk_core.ipynb "
        "--output ../reports/fnsipid_tail_risk_core_v1/fnsipid_tail_risk_core.executed.ipynb "
        "--ExecutePreprocessor.timeout=-1"
    )
CORE = json.loads((CORE_DIR / "manifest.json").read_text(encoding="utf-8"))
if (
    CORE.get("status") != "completed"
    or CORE.get("run_mode") != "full"
    or CORE.get("configuration", {}).get("RUN_MODE") != "full"
    or bool(CORE.get("smoke_run_is_engineering_check_only"))
):
    raise RuntimeError("the explainer requires a completed full core run; smoke and failed bundles are rejected")
if not CORE.get("gates") or any(row.get("status") != "PASS" for row in CORE["gates"]):
    raise RuntimeError("the explainer refuses a core run with missing or failed data gates")
if not CORE.get("assertions") or any(not bool(row.get("passed")) for row in CORE["assertions"]):
    raise RuntimeError("the explainer refuses a core run with missing or failed assertions")
CORE_TIMING = CORE.get("timing_rule", {})
UPSTREAM_TIMING_COUNTS = CORE_TIMING.get("upstream_counts_before_window_and_deduplication", {})
DATE_ONLY_UPSTREAM_ROWS = int(UPSTREAM_TIMING_COUNTS.get("date_only_or_exact_midnight_rows", 0))
PRECISE_TIMESTAMP_UPSTREAM_ROWS = int(UPSTREAM_TIMING_COUNTS.get("precise_timestamp_rows", 0))
if DATE_ONLY_UPSTREAM_ROWS <= 0 or PRECISE_TIMESTAMP_UPSTREAM_ROWS <= 0:
    raise RuntimeError("the explainer requires the core run's verified mixed-timing audit")
CORE_GIT_COMMIT = CORE.get("git", {}).get("commit")
CORE_GIT_LABEL = CORE_GIT_COMMIT[:12] if isinstance(CORE_GIT_COMMIT, str) and CORE_GIT_COMMIT else "unavailable"

# Clear figures from any earlier pass so the index can never list a stale file.
stale = sorted(FIGURE_DIR.glob("*.png"))
for path in stale:
    path.unlink()
if stale:
    print(f"cleared {len(stale)} figure(s) from a previous pass")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
print(f"core run      : {CORE['experiment']} ({CORE['status']}, mode={CORE['run_mode']})")
print(f"generated     : {CORE['created_at']}")
print(f"git commit    : {CORE_GIT_LABEL} (dirty={CORE.get('git', {}).get('dirty')})")
print(f"runtime       : {CORE['runtime_seconds'] / 60:.1f} min")
print(f"counts        : {json.dumps(CORE['counts'])}")

# %% [markdown]
# ### Integrity gate
#
# Every artifact this notebook plots is hashed and compared with the hash the
# core run recorded for it. If the frozen bundle has drifted, this cell stops
# the notebook rather than drawing a stale picture.

# %%
READS = [
    "forecasts.parquet",
    "model_comparison.csv",
    "bootstrap.csv",
    "calibration.csv",
    "robustness.csv",
    "garch_fit_summary.csv",
    "price_validation.csv",
    "attrition.csv",
]

integrity_rows = []
for name in READS:
    path = CORE_DIR / name
    recorded = CORE["outputs"].get(name, {}).get("sha256")
    observed = sha256_file(path)
    integrity_rows.append(
        {
            "artifact": name,
            "size_mb": round(path.stat().st_size / 1_048_576, 2),
            "sha256_matches_manifest": recorded == observed,
            "sha256": observed[:16] + "...",
        }
    )
INTEGRITY = pd.DataFrame(integrity_rows)
display(INTEGRITY)
if not INTEGRITY["sha256_matches_manifest"].all():
    raise RuntimeError("a frozen artifact no longer matches the hash recorded in the core manifest")
print("\nAll frozen artifacts match the core run manifest. Everything below describes that exact run.")

# %% [markdown]
# ### Load
#
# `forecasts.parquet` is 233 MB and 46 columns wide; only the columns these
# figures need are read.

# %%
FORECAST_COLUMNS = [
    "symbol",
    "forecast_date",
    "target_date",
    "target_return",
    "z_target",
    "sigma_forecast",
    "news_indicator",
    "news_count",
    "semantic_intensity",
    "adverse_tone",
    "recap_share",
    "var_z_M1",
    "var_z_M2",
    "var_return_M1",
    "var_return_M2",
    "es_return_M2",
    "fz0_M0",
    "fz0_M1",
    "fz0_M2",
    "fz0_M2_intensity",
    "hit_M0",
    "hit_M1",
    "hit_M2",
    "hit_M2_intensity",
    "d_semantic",
    "d_tone_given_intensity",
    "d_news_vs_price",
]
FORECASTS = pd.read_parquet(CORE_DIR / "forecasts.parquet", columns=FORECAST_COLUMNS)
NEWS = FORECASTS[FORECASTS["news_indicator"] == 1]

COMPARISON = pd.read_csv(CORE_DIR / "model_comparison.csv")
BOOTSTRAP = pd.read_csv(CORE_DIR / "bootstrap.csv")
CALIBRATION = pd.read_csv(CORE_DIR / "calibration.csv")
ROBUSTNESS = pd.read_csv(CORE_DIR / "robustness.csv")
GARCH = pd.read_csv(CORE_DIR / "garch_fit_summary.csv")

ALPHA = CORE["configuration"]["ALPHA"]
BLOCK = CORE["bootstrap"]["block_length"]
REPS = CORE["bootstrap"]["replications"]
SEED = CORE["bootstrap"]["seed"]
MODELS = ["M0", "M1", "M2", "M2_intensity"]
MODEL_LABELS = {
    "M0": "M0\nprice state only",
    "M1": "M1\n+ news arrival & volume",
    "M2": "M2\n+ semantics",
    "M2_intensity": "M2$_{int}$\n+ intensity only",
}

print(f"evaluation rows {len(FORECASTS):,} | news-bearing {len(NEWS):,}")
print(f"firms {FORECASTS['symbol'].nunique()} | target dates {FORECASTS['target_date'].nunique():,}")
print(f"alpha {ALPHA} | bootstrap: {REPS} reps, block {BLOCK} dates, seed {SEED}")

# %% [markdown]
# ### House style
#
# One palette, one figure frame. Darker means "more information in the model",
# so the nesting `M0 ⊂ M1 ⊂ M2` reads straight off the colour ramp. Crimson
# always means "no effect / nominal level".

# %%
INK = "#1F2933"
MUTED = "#7B8794"
GRIDC = "#DCE3EA"
PALETTE = {
    "M0": "#BCCBD6",
    "M1": "#4F92AD",
    "M2": "#13405C",
    "M2_intensity": "#8FBBCE",
    "zero": "#C0392B",
    "good": "#2E7D32",
    "news": "#D98C33",
    "quiet": "#A7B0BA",
    "accent": "#7E4B8C",
}

# DejaVu Sans has no "semibold" face, so matplotlib logs a fallback to bold for
# every such label. The fallback is what we want; the log line is just noise.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.titlecolor": MUTED,
        "axes.titlepad": 11,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "grid.color": GRIDC,
        "grid.linewidth": 0.8,
        "figure.dpi": 110,
    }
)

FIGURE_INDEX: list[dict[str, str]] = []
SAMPLE_LINE = (
    f"Full run · {CORE['counts']['retained_firms']} firms · development 2011-2016, evaluation 2017-2023 · "
    f"{CORE['counts']['evaluation_rows']:,} evaluation firm-days"
)


def tidy(axis: plt.Axes, *, grid: str = "y") -> plt.Axes:
    """Strip chartjunk: no top/right spine, one light grid direction."""

    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRIDC)
    if grid in {"y", "both"}:
        axis.yaxis.grid(True, zorder=0)
    if grid in {"x", "both"}:
        axis.xaxis.grid(True, zorder=0)
    axis.set_axisbelow(True)
    return axis


def finish(fig: plt.Figure, name: str, title: str, subtitle: str, takeaway: str, source: str) -> None:
    """Title / subtitle / source footer laid out in absolute inches, then save and show.

    Reserving space in inches rather than figure fractions keeps multi-line
    subtitles from colliding with the first axes title on tall figures.
    """

    height = fig.get_size_inches()[1]
    subtitle_lines = subtitle.count("\n") + 1
    source_lines = source.count("\n") + 1
    top_pad = 0.62 + 0.20 * subtitle_lines + 0.30
    bottom_pad = 0.16 + 0.15 * source_lines + 0.16

    fig.suptitle(title, x=0.010, y=1 - 0.28 / height, ha="left", va="top", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.010, 1 - 0.62 / height, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    fig.text(0.010, 0.20 / height, source, ha="left", va="bottom", fontsize=8, color=MUTED, style="italic")
    # tight_layout silently gives up on some composite layouts; when it warns,
    # fall back to an explicit margin so the axes title cannot ride up into the
    # subtitle block.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.tight_layout(rect=(0.0, bottom_pad / height, 1.0, 1 - top_pad / height))
    if any("tight_layout" in str(entry.message) for entry in caught):
        fig.subplots_adjust(
            left=0.085,
            right=0.975,
            top=1 - (top_pad + 0.36) / height,
            bottom=bottom_pad / height + 0.11,
            hspace=0.45,
        )

    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    FIGURE_INDEX.append({"file": f"figures/{name}.png", "title": title, "takeaway": takeaway})
    display(fig)
    plt.close(fig)
    print(f"saved {path.relative_to(REPO_ROOT)}\n-> {takeaway}")


# %% [markdown]
# ## 2. Where the data came from
#
# Before any result, it is worth seeing how 28.6 million raw news rows become
# 958 thousand scored forecasts, and what was dropped on the way.

# %%
E6_MANIFEST = json.loads((REPO_ROOT / "reports/loop_moment2_finbert_20260719/manifest.json").read_text(encoding="utf-8"))
E6 = E6_MANIFEST["counts"]

text_stages = [
    ("Raw FNSPID rows scanned", E6["physical_rows_scanned"], ""),
    ("Rows for the 574 coherent firms", E6["rows_for_coherent_symbols"], "other tickers dropped upstream"),
    ("Deduplicated scored headlines", E6["deduplicated_coherent_events"], "URL / headline duplicates removed"),
    ("Firm reaction-sessions with news", E6["aggregated_symbol_sessions"], "headlines folded onto sessions"),
]
firm_stages = [
    ("Coherent FNSPID cohort", 574, ""),
    ("Has a price file", 571, "SEE, SKYW, SOL missing"),
    ("Price not frozen / stale", 570, "YNDX: 462 frozen sessions"),
    ("Meets support thresholds", CORE["counts"]["modelling_universe"], "< 500 dev or < 250 eval origins"),
    ("GARCH fit stationary", CORE["counts"]["retained_firms"], "6 near-unit-root fits excluded"),
]
day_stages = [
    ("Panel firm-days built", 1_923_584, ""),
    ("Inside dev / eval windows", CORE["counts"]["panel_rows"], "pre-2010 warm-up origins"),
    ("Evaluation firm-days", CORE["counts"]["evaluation_rows"], "2017-2023 forecast origins"),
    ("...of which news-bearing", CORE["counts"]["evaluation_news_rows"], "the primary population"),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
for axis, (stages, heading, colour) in zip(
    axes,
    [
        (text_stages, "Text", PALETTE["news"]),
        (firm_stages, "Firms", PALETTE["M1"]),
        (day_stages, "Forecast rows", PALETTE["M2"]),
    ],
    strict=True,
):
    labels = [stage[0] for stage in stages]
    values = [stage[1] for stage in stages]
    positions = np.arange(len(stages))
    widths = np.array(values, dtype=float) / max(values)
    axis.barh(positions, widths, height=0.46, color=colour, alpha=0.92, zorder=3)
    for position, (_label, value, note) in enumerate(stages):
        axis.text(widths[position] + 0.025, position, f"{value:,}", va="center", fontsize=10, fontweight="semibold", color=INK)
        if note:
            # The note explains why this stage is smaller than the one above it,
            # so it sits just below its own bar.
            axis.text(0.012, position + 0.40, f"\u2193  {note}", va="center", fontsize=8, color=MUTED, style="italic")
    axis.set_yticks(positions, labels, fontsize=9.5)
    axis.set_ylim(len(stages) - 0.35, -0.6)
    axis.set_xlim(0, 1.45)
    axis.set_xticks([])
    axis.set_title(heading, fontsize=12, fontweight="bold", color=INK)
    tidy(axis, grid="none")
    axis.spines["bottom"].set_visible(False)

finish(
    fig,
    "e01_funnel",
    "From 28.6 million raw news rows to 416 thousand scored news days",
    "Bar length is relative to the first stage in each panel. Nothing was rescored or re-downloaded: the corpus, its\n"
    "leak-safe date mapping and the FinBERT scores are reused from the completed E5/E6 runs.",
    "Only 13 of 574 cohort firms are lost, and every exclusion has one deterministic reason.",
    f"Source: reports/fnsipid_tail_risk_core_v1/attrition.csv and reports/loop_moment2_finbert_20260719/manifest.json · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 3. The timing rule — the part that is easiest to get wrong
#
# FNSPID headlines carry a **calendar date**, not a usable timestamp. So the
# experiment cannot ask "what happens in the minutes after the story?". It
# deliberately asks the harder, later question: *after the market has had a full
# session to react, is there anything left in the text?*
#
# The figure below is the whole timing contract. If you only look at one
# picture in this notebook, look at this one.

# %%
fig, axis = plt.subplots(figsize=(14, 6.4))
axis.set_xlim(-0.6, 6.1)
axis.set_ylim(-2.55, 2.5)
axis.axis("off")

days = [("Thu", True), ("Fri", True), ("Sat", False), ("Sun", False), ("Mon", True), ("Tue", True)]
BOX_Y, BOX_H = 0.0, 0.72
for index, (label, trading) in enumerate(days):
    face = "#EAF1F5" if trading else "#F4F5F6"
    edge = PALETTE["M1"] if trading else "#CBD2D9"
    axis.add_patch(Rectangle((index - 0.42, BOX_Y), 0.84, BOX_H, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3))
    axis.text(index, BOX_Y + BOX_H / 2 + 0.10, label, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    axis.text(
        index,
        BOX_Y + BOX_H / 2 - 0.17,
        "trading session" if trading else "market closed",
        ha="center",
        va="center",
        fontsize=7.5,
        color=MUTED,
    )

# Headlines dated Friday, Saturday and Sunday all react on Monday.
for source_day in (1, 2, 3):
    axis.plot([source_day], [1.62], marker="o", markersize=9, color=PALETTE["news"], zorder=5)
    axis.text(
        source_day,
        1.92,
        f"headline\ndated {days[source_day][0]}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=PALETTE["news"],
        fontweight="semibold",
    )
    axis.add_patch(
        FancyArrowPatch(
            (source_day, 1.50),
            (4.0, BOX_Y + BOX_H + 0.10),
            connectionstyle=f"arc3,rad={-0.30 + 0.10 * source_day}",
            arrowstyle="-|>,head_width=4,head_length=7",
            color=PALETTE["news"],
            linewidth=1.7,
            alpha=0.85,
            zorder=4,
        )
    )

# The reaction-session caption sits under its own box, clear of the arrows.
axis.text(4.0, BOX_Y - 0.10, "reaction session $s$", ha="center", va="top", fontsize=10.5, fontweight="bold", color=PALETTE["M2"])

ORIGIN = 4.42
axis.axvline(ORIGIN, ymin=0.10, ymax=0.82, color=PALETTE["M2"], linewidth=2.2, linestyle="--", zorder=6)
axis.add_patch(Rectangle((-0.6, -2.55), ORIGIN + 0.6, 5.05, facecolor=PALETTE["M1"], alpha=0.055, zorder=0))
axis.text(
    ORIGIN - 0.10,
    2.36,
    "FORECAST ORIGIN\nclose of session $s$",
    ha="right",
    va="top",
    fontsize=10.5,
    fontweight="bold",
    color=PALETTE["M2"],
)
axis.text(-0.55, 2.36, "known at the forecast origin", ha="left", va="top", fontsize=9.5, color=PALETTE["M1"], fontweight="semibold")
axis.text(
    ORIGIN + 0.14,
    2.36,
    "unknown — this is what we forecast",
    ha="left",
    va="top",
    fontsize=9.5,
    color=PALETTE["zero"],
    fontweight="semibold",
)


def bracket(x0: float, x1: float, y: float, text: str, colour: str) -> None:
    axis.plot([x0, x0, x1, x1], [y + 0.13, y, y, y + 0.13], color=colour, linewidth=1.8, zorder=5)
    axis.text((x0 + x1) / 2, y - 0.10, text, ha="center", va="top", fontsize=9.5, color=colour, fontweight="semibold")


bracket(1.0, 4.0, -0.80, "return realised DURING session $s$\n$\\rightarrow$ a PREDICTOR (reaction shock, its size)", PALETTE["M1"])
bracket(4.0, 5.0, -1.75, "close-to-close return $s \\rightarrow s+1$\n$\\rightarrow$ the TARGET we forecast", PALETTE["zero"])

axis.text(
    -0.5,
    1.52,
    "Saturday and Sunday news collapse onto the\nsame Monday session and are aggregated into\none firm-session feature row",
    fontsize=9,
    color=MUTED,
    style="italic",
    ha="left",
    va="top",
)

finish(
    fig,
    "e02_timing_rule",
    "The leak-safe timing rule: news reacts, then we forecast the day after",
    "A headline dated $d$ is mapped to the first trading session strictly after $d$. The return realised during that\n"
    "session is a predictor; the target is the NEXT close-to-close return. The initial price response is spent before\n"
    "the forecast is even made — that is what makes this a hard question rather than an event study.",
    "The exhaustive check over all 4,779 calendar dates in the window found 0 mappings that were not strictly forward.",
    f"Source: reports/fnsipid_tail_risk_core_v1/manifest.json (timing_rule) · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 4. The three models
#
# The models are strictly nested: each one keeps everything the previous one
# had and adds a block of variables. That is what makes the comparison a clean
# "does this block of information buy anything?" question.

# %%
fig, axis = plt.subplots(figsize=(13, 6.0))
axis.set_xlim(0, 10)
axis.set_ylim(0, 6.2)
axis.axis("off")

layers = [
    (
        0.25,
        9.5,
        5.4,
        PALETTE["M2"],
        "M2  ·  + semantics",
        ["semantic_intensity  =  mean($p_{neg} + p_{pos}$)", "adverse_tone  =  mean($p_{neg} - p_{pos}$)"],
    ),
    (
        0.85,
        8.3,
        4.2,
        PALETTE["M1"],
        "M1  ·  + news arrival & volume",
        ["news_indicator  (was there any news at all?)", "log1p(news_count)  (how much?)"],
    ),
    (
        1.45,
        7.1,
        3.0,
        PALETTE["M0"],
        "M0  ·  price state only",
        ["reaction shock $z$ and its magnitude", "log conditional volatility (GJR-GARCH)", "market return, |market return|, market vol"],
    ),
]
for depth, (x0, width, height, colour, heading, items) in enumerate(layers):
    axis.add_patch(
        FancyBboxPatch(
            (x0, 0.35 + (5.4 - height) / 2),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.22",
            facecolor=colour,
            edgecolor="white",
            linewidth=2.5,
            zorder=depth + 1,
        )
    )
    text_colour = "white" if colour != PALETTE["M0"] else INK
    top = 0.35 + (5.4 - height) / 2 + height
    axis.text(x0 + 0.35, top - 0.42, heading, fontsize=12.5, fontweight="bold", color=text_colour, va="center", zorder=10)
    for offset, item in enumerate(items):
        axis.text(x0 + 0.55, top - 0.95 - 0.42 * offset, "•  " + item, fontsize=10, color=text_colour, va="center", zorder=10)

axis.annotate(
    "",
    xy=(9.72, 3.05),
    xytext=(9.72, 4.95),
    arrowprops={"arrowstyle": "-|>", "color": PALETTE["zero"], "linewidth": 2.0},
)
axis.text(9.72, 5.05, "the headline test\nM2 vs M1", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=PALETTE["zero"])
axis.text(
    0.25,
    0.10,
    "All three are estimated on development rows only (2011-2016) and scored on the identical 958,461 evaluation rows.\n"
    "Each produces a joint VaR and Expected Shortfall forecast, parameterised so that ES < VaR < 0 always holds.",
    fontsize=9.5,
    color=MUTED,
    va="top",
)

finish(
    fig,
    "e03_nested_models",
    "Three strictly nested information sets",
    "M1 asks whether knowing that news happened, and how much, helps. M2 asks whether knowing what the news SAID\n"
    "helps on top of that. Because they are nested and share evaluation rows, the difference in loss is attributable.",
    "The comparison is M2 vs M1, not M2 vs nothing — the bar M2 has to clear is 'beyond arrival and volume'.",
    f"Source: reports/fnsipid_tail_risk_core_v1/manifest.json (model_formulas) · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 5. The headline result
#
# Forecasts are scored with the Fissler-Ziegel `FZ0` loss, the proper joint
# scoring function for VaR and ES together. **Lower is better.** A model cannot
# game it by being systematically too cautious or too aggressive.
#
# The absolute numbers look almost identical. That is exactly the point, and
# the next figure puts the differences on a scale where you can read them.

# %%
news_block = COMPARISON[COMPARISON["population"] == "news_bearing_evaluation_origins"]
panel_block = COMPARISON[COMPARISON["population"] == "full_evaluation_panel"]


def mean_fz0(block: pd.DataFrame, model: str) -> float:
    return float(block[block["quantity"] == f"mean_fz0_{model}"]["value"].iloc[0])


fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
for axis, (block, heading, count) in zip(
    axes,
    [
        (news_block, "News-bearing forecast origins (the primary population)", len(NEWS)),
        (panel_block, "Full evaluation panel, news and quiet days", len(FORECASTS)),
    ],
    strict=True,
):
    values = [mean_fz0(block, model) for model in MODELS]
    positions = np.arange(len(MODELS))
    axis.bar(positions, values, width=0.62, color=[PALETTE[m] for m in MODELS], zorder=3)
    low, high = min(values), max(values)
    span = high - low
    axis.set_ylim(low - span * 1.75, high + span * 0.55)
    for position, value in zip(positions, values, strict=True):
        axis.text(position, value + span * 0.05, f"{value:.6f}", ha="center", fontsize=10, fontweight="semibold", color=INK)
    # Significance-style brackets sit entirely below the bars, so nothing crosses a bar.
    for (i, j), depth, colour, label in (((0, 1), 0.55, PALETTE["good"], "arrival + volume"), ((1, 2), 1.15, PALETTE["zero"], "semantics")):
        y = low - span * depth
        axis.plot([i, i, j, j], [y + span * 0.10, y, y, y + span * 0.10], color=colour, linewidth=1.7, zorder=4)
        axis.text(
            (i + j) / 2,
            y - span * 0.05,
            f"{label}\n{values[j] - values[i]:+.3g}",
            ha="center",
            va="top",
            fontsize=9.5,
            color=colour,
            fontweight="semibold",
        )
    axis.set_xticks(positions, [MODEL_LABELS[m] for m in MODELS])
    axis.set_ylabel("mean FZ0 loss  (lower is better)")
    axis.set_title(f"{heading}\nn = {count:,}")
    tidy(axis)

finish(
    fig,
    "e04_loss_ladder",
    "Adding news arrival moves the loss. Adding sentiment barely moves it.",
    "Mean joint VaR/ES loss for each nested model on identical evaluation rows. The brackets show what each block of\n"
    "information bought. Note the axis: the whole visible range is about one part in a thousand of the loss level.",
    f"On news days, arrival + volume buys {mean_fz0(news_block, 'M1') - mean_fz0(news_block, 'M0'):+.3g} of loss; semantics buys {mean_fz0(news_block, 'M2') - mean_fz0(news_block, 'M1'):+.3g} — about 3% as much.",
    f"Source: reports/fnsipid_tail_risk_core_v1/model_comparison.csv · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 6. Effect sizes with honest uncertainty
#
# Firm-days on the same calendar date are strongly dependent — one bad market
# day moves every firm at once. So uncertainty is quantified with a **moving
# block bootstrap over target dates**: the paired loss difference is first
# averaged across the whole cross-section of firms on each date, then blocks of
# 20 consecutive dates are resampled, keeping every firm on a date together.
#
# This is the plot that decides the question.

# %%
CONTRASTS = {
    "d_news_vs_price": ("M1 $-$ M0\ndoes news ARRIVAL beat price state?", PALETTE["good"]),
    "d_semantic": ("M2 $-$ M1\ndoes SENTIMENT beat arrival?", PALETTE["zero"]),
    "d_tone_given_intensity": ("M2 $-$ M2$_{int}$\ndoes signed TONE beat intensity?", PALETTE["accent"]),
}
order = list(CONTRASTS)
primary = BOOTSTRAP[
    (BOOTSTRAP["population"] == "news_bearing_evaluation_origins")
    & (BOOTSTRAP["weighting"] == "observation_equal")
    & (BOOTSTRAP["block_length"] == BLOCK)
].set_index("contrast")

fig, axis = plt.subplots(figsize=(13, 5.6))
for position, contrast in enumerate(order):
    row = primary.loc[contrast]
    colour = CONTRASTS[contrast][1]
    axis.plot([row["ci_low"], row["ci_high"]], [position, position], color=colour, linewidth=3.4, solid_capstyle="round", zorder=3)
    for bound in ("ci_low", "ci_high"):
        axis.plot([row[bound]], [position], marker="|", markersize=14, markeredgewidth=2.4, color=colour, zorder=4)
    axis.plot(
        [row["point_estimate"]], [position], marker="o", markersize=12, color=colour, markeredgecolor="white", markeredgewidth=1.8, zorder=5
    )
    excludes = row["ci_high"] < 0 or row["ci_low"] > 0
    # Annotation sits above its own interval, so it never crosses the zero line.
    axis.text(
        row["ci_low"],
        position - 0.14,
        f"{row['point_estimate']:+.2e}   95% CI [{row['ci_low']:+.2e}, {row['ci_high']:+.2e}]\n"
        f"{'interval excludes zero' if excludes else 'interval covers zero — no detectable effect'}",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6, "alpha": 0.88},
        fontsize=9.5,
        color=colour if excludes else MUTED,
        fontweight="semibold" if excludes else "normal",
    )

axis.axvline(0.0, color=INK, linewidth=1.6, zorder=2)
axis.axvspan(-0.024, 0, color=PALETTE["good"], alpha=0.045, zorder=0)
axis.set_yticks(range(len(order)), [CONTRASTS[c][0] for c in order], fontsize=10.5)
axis.invert_yaxis()
axis.set_ylim(len(order) - 0.45, -0.95)
axis.set_xlim(-0.024, 0.014)
axis.set_xlabel("paired FZ0 loss difference   (negative = the richer model forecasts better)")
axis.text(0.0006, -0.90, "no effect", fontsize=9.5, color=INK, ha="left", va="top")
tidy(axis, grid="x")

finish(
    fig,
    "e05_effect_sizes",
    "News arrival clears zero decisively. Sentiment does not come close.",
    "Point estimate and 95% moving-block bootstrap interval for each nested contrast, on news-bearing evaluation\n"
    f"origins. {REPS:,} replications, blocks of {BLOCK} target dates, seed {SEED}. Models are not refitted inside the bootstrap:\n"
    "this is sampling uncertainty in the paired loss difference of fixed forecasts.",
    f"The semantic effect is {abs(primary.loc['d_semantic', 'point_estimate'] / primary.loc['d_news_vs_price', 'point_estimate']):.1%} the size of the arrival effect, and its interval spans zero.",
    f"Source: reports/fnsipid_tail_risk_core_v1/bootstrap.csv · {SAMPLE_LINE}",
)

# %% [markdown]
# ### Is this a null, or just a weak test?
#
# A null result is only interesting if the design could have detected an effect.
# It could: the *same* machinery, on the *same* rows, with the *same* bootstrap,
# detects the news-arrival effect at overwhelming confidence. The bootstrap
# distributions below make the contrast concrete.
#
# The replicate means are recomputed here and asserted equal to the frozen
# `bootstrap.csv` before plotting.


# %%
def date_level(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    grouped = frame.groupby("target_date", sort=True)[column].agg(["mean", "size"])
    return grouped["mean"].to_numpy(dtype=float), grouped["size"].to_numpy(dtype=float)


replicates: dict[str, np.ndarray] = {}
for contrast in CONTRASTS:
    values, weights = date_level(NEWS, contrast)
    check = date_block_bootstrap_mean(values, weights, block_length=BLOCK, replications=REPS, seed=SEED)
    frozen = primary.loc[contrast]
    for key in ("point_estimate", "ci_low", "ci_high", "share_below_zero"):
        assert abs(check[key] - float(frozen[key])) < 1e-12, f"{contrast}: recomputed {key} differs from bootstrap.csv"
    draws = block_bootstrap_indices(values.size, block_length=BLOCK, replications=REPS, seed=SEED)
    replicates[contrast] = np.sum(values[draws] * weights[draws], axis=1) / np.sum(weights[draws], axis=1)
print(f"recomputed bootstrap reproduces bootstrap.csv exactly for all {len(CONTRASTS)} contrasts")

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})

axis = axes[0]
for contrast in order:
    label, colour = CONTRASTS[contrast]
    sample = replicates[contrast]
    axis.hist(sample, bins=55, color=colour, alpha=0.55, zorder=3, label=label.replace("\n", " — "))
    axis.axvline(sample.mean(), color=colour, linewidth=2.0, zorder=4)
axis.axvline(0.0, color=INK, linewidth=1.8, linestyle="--", zorder=5)
axis.set_ylim(0, axis.get_ylim()[1] * 1.22)
axis.text(0.0004, axis.get_ylim()[1] * 0.99, "no effect", fontsize=9.5, color=INK, va="top", ha="left")
axis.set_xlabel("bootstrap replicate mean of the paired FZ0 difference")
axis.set_ylabel(f"replications (of {REPS:,})")
axis.legend(loc="upper left")
axis.set_title("All three contrasts on one scale")
tidy(axis)

axis = axes[1]
shares = [float(primary.loc[c, "share_below_zero"]) for c in order]
colours = [CONTRASTS[c][1] for c in order]
positions = np.arange(len(order))
axis.barh(positions, shares, height=0.5, color=colours, zorder=3)
axis.axvline(0.5, color=INK, linewidth=1.6, linestyle="--", zorder=4)
for position, share in zip(positions, shares, strict=True):
    axis.text(share + 0.02, position, f"{share:.1%}", va="center", fontsize=11, fontweight="semibold", color=INK)
axis.set_yticks(positions, [CONTRASTS[c][0].split("\n")[0] for c in order], fontsize=11)
axis.invert_yaxis()
axis.set_ylim(len(order) - 0.25, -0.6)
axis.set_xlim(0, 1.18)
axis.set_xlabel("share of bootstrap replicates below zero")
axis.text(0.5, len(order) - 0.32, "a coin flip", ha="center", va="top", fontsize=9, color=MUTED)
axis.text(1.0, len(order) - 0.32, "certain", ha="center", va="top", fontsize=9, color=MUTED)
axis.set_title("How confident is each verdict?")
tidy(axis, grid="x")

finish(
    fig,
    "e06_bootstrap_distributions",
    "The design can detect an effect — it detects one, just not from sentiment",
    "Bootstrap distributions of the three paired differences. The arrival effect sits entirely to the left of zero;\n"
    "the two semantic contrasts sit astride it. Same rows, same bootstrap, same seed — only the contrast changes.",
    f"Arrival: {shares[0]:.1%} of replicates below zero. Sentiment: {shares[1]:.1%}. Signed tone: {shares[2]:.1%}.",
    f"Source: recomputed from reports/fnsipid_tail_risk_core_v1/forecasts.parquet, verified against bootstrap.csv · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 7. Where does the (tiny) semantic difference come from?
#
# A mean can hide a story: a small average can be a steady drizzle or one
# thunderstorm. Cumulating the date-level contributions shows which it is. The
# curve below ends exactly at the point estimate.

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8.4), gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.45})
EVAL_DATES = pd.DatetimeIndex(sorted(NEWS["target_date"].unique()))

axis = axes[0]
paths = {}
for contrast in ("d_news_vs_price", "d_semantic"):
    values, weights = date_level(NEWS, contrast)
    contribution = np.cumsum(values * weights) / weights.sum()
    paths[contrast] = contribution
    label, colour = CONTRASTS[contrast]
    axis.plot(EVAL_DATES, contribution, color=colour, linewidth=2.2, zorder=3, label=label.replace("\n", " — "))
    axis.annotate(
        f"{contribution[-1]:+.2e}",
        xy=(EVAL_DATES[-1], contribution[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=10,
        fontweight="semibold",
        color=colour,
        va="center",
    )
axis.axhline(0.0, color=INK, linewidth=1.4, linestyle="--", zorder=2)
axis.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"), color=PALETTE["news"], alpha=0.16, zorder=1)
axis.text(pd.Timestamp("2020-05-12"), -0.0088, "COVID crash", ha="left", fontsize=9.5, color=PALETTE["news"], fontweight="semibold")
axis.set_ylabel("cumulative contribution to the\nmean paired FZ0 difference")
axis.legend(loc="lower left")
axis.set_title("Cumulative path — a steady slope means a persistent effect, a single step means one episode")
tidy(axis)

axis = axes[1]
yearly = COMPARISON[COMPARISON["population"].str.startswith("news_bearing_2") & (COMPARISON["quantity"] == "paired_fz0_M2_minus_M1")].copy()
yearly["year"] = yearly["population"].str[-4:].astype(int)
yearly = yearly.sort_values("year")
colours = [PALETTE["good"] if value < 0 else PALETTE["zero"] for value in yearly["value"]]
axis.bar(yearly["year"], yearly["value"], width=0.6, color=colours, zorder=3)
axis.axhline(0.0, color=INK, linewidth=1.4, zorder=4)
limit = float(np.abs(yearly["value"]).max())
axis.set_ylim(-limit * 1.5, limit * 1.1)
for year, value in zip(yearly["year"], yearly["value"], strict=True):
    axis.text(
        year,
        value + (limit * 0.06 if value > 0 else -limit * 0.06),
        f"{value:+.1e}",
        ha="center",
        va="bottom" if value > 0 else "top",
        fontsize=9,
        color=INK,
    )
axis.set_xticks(yearly["year"])
axis.set_ylabel("M2 $-$ M1 by year")
axis.set_xlabel("evaluation year")
axis.set_title(f"Negative means semantics helped that year — {int((yearly['value'] > 0).sum())} of {len(yearly)} years go the other way")
tidy(axis)

covid_step = float(paths["d_semantic"][EVAL_DATES.get_indexer([pd.Timestamp("2020-05-01")], method="nearest")[0]])
finish(
    fig,
    "e07_where_it_comes_from",
    "The semantic difference is one crisis, then a drift back toward zero",
    "Top: cumulative date-level contribution to the mean paired difference, ending exactly at the point estimate.\n"
    "Bottom: the same difference cut by evaluation year. Negative means semantics helped that year.",
    f"The semantic curve is flat until the 2020 crash drops it to {covid_step:+.1e}, then drifts back to {paths['d_semantic'][-1]:+.1e}. No persistent slope; the arrival curve, by contrast, declines throughout.",
    f"Source: reports/fnsipid_tail_risk_core_v1/forecasts.parquet and model_comparison.csv · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 8. The mechanical reason: M1 and M2 barely disagree
#
# The loss difference is small because the two models produce nearly the same
# forecast. Sentiment does move the tail in a sensible direction — more adverse
# tone pushes VaR deeper — but by a couple of percent of the forecast level, on
# a quantity whose day-to-day noise is far larger.

# %%
delta_z = (NEWS["var_z_M2"] - NEWS["var_z_M1"]).to_numpy()
delta_bp = ((NEWS["var_return_M2"] - NEWS["var_return_M1"]) * 10_000).to_numpy()
relative = np.abs(delta_bp) / (NEWS["var_return_M1"].abs().to_numpy() * 10_000) * 100
median_bp = float(np.median(np.abs(delta_bp)))
median_relative = float(np.median(relative))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

axis = axes[0]
limit = float(np.quantile(np.abs(delta_bp), 0.99))
axis.hist(np.clip(delta_bp, -limit, limit), bins=70, color=PALETTE["M2"], alpha=0.88, zorder=3)
axis.axvline(0.0, color=INK, linewidth=1.6, linestyle="--", zorder=4)
axis.set_xlabel(f"VaR$_{{M2}}$ $-$ VaR$_{{M1}}$   (basis points, clipped at the 99th pct, $\\pm${limit:.0f})")
axis.set_ylabel("news-bearing evaluation days")
axis.set_title(f"Median absolute disagreement: {median_bp:.1f} bp")
tidy(axis)

axis = axes[1]
upper = float(np.quantile(relative, 0.99))
axis.hist(np.clip(relative, 0, upper), bins=70, color=PALETTE["M1"], alpha=0.9, zorder=3)
axis.axvline(median_relative, color=PALETTE["zero"], linewidth=2.2, zorder=4)
axis.text(
    median_relative + upper * 0.03,
    axis.get_ylim()[1] * 0.92,
    f"median {median_relative:.2f}%",
    fontsize=10,
    color=PALETTE["zero"],
    fontweight="semibold",
)
axis.set_xlabel(f"|VaR$_{{M2}}$ $-$ VaR$_{{M1}}$| as a % of the VaR level (clipped at the 99th pct, {upper:.1f}%)")
axis.set_ylabel("news-bearing evaluation days")
axis.set_title("As a share of the forecast itself")
tidy(axis)

axis = axes[2]
low, high = np.quantile(delta_z, [0.005, 0.995])
inside = (delta_z >= low) & (delta_z <= high)
hexes = axis.hexbin(NEWS["adverse_tone"].to_numpy()[inside], delta_z[inside], gridsize=52, bins="log", cmap="BuPu", mincnt=1, zorder=3)
bins = pd.cut(NEWS["adverse_tone"], np.linspace(-1, 1, 21))
trend = pd.Series(delta_z, index=NEWS.index).groupby(bins, observed=True).mean()
axis.plot(
    [interval.mid for interval in trend.index],
    trend.to_numpy(),
    color=PALETTE["news"],
    linewidth=3.0,
    marker="o",
    markersize=4,
    zorder=5,
    label="mean within tone bin",
)
axis.axhline(0.0, color=INK, linewidth=1.2, linestyle="--", zorder=4)
axis.axvline(0.0, color=INK, linewidth=1.2, linestyle="--", zorder=4)
axis.set_ylim(low, high)
axis.set_xlabel("adverse tone  =  mean($p_{neg} - p_{pos}$)   $\\rightarrow$ more adverse")
axis.set_ylabel("VaR$_{M2}$ $-$ VaR$_{M1}$  (standardised units)")
axis.legend(loc="upper right")
axis.set_title("Confident text of EITHER sign deepens VaR")
fig.colorbar(hexes, ax=axis, label="days (log scale)", fraction=0.046, pad=0.02)
tidy(axis, grid="none")

# The curve is close to symmetric in tone: what deepens the tail is how far the
# text is from neutral, not which way it points.  Quantify the asymmetry.
very_adverse = float(delta_z[NEWS["adverse_tone"].to_numpy() >= 0.8].mean())
very_positive = float(delta_z[NEWS["adverse_tone"].to_numpy() <= -0.8].mean())
neutral_text = float(delta_z[np.abs(NEWS["adverse_tone"].to_numpy()) <= 0.1].mean())
axis.annotate(
    f"very positive text: {very_positive:+.3f}\nneutral text: {neutral_text:+.3f}\nvery adverse text: {very_adverse:+.3f}",
    xy=(0.03, 0.05),
    xycoords="axes fraction",
    fontsize=8.5,
    color=INK,
    va="bottom",
    bbox={"facecolor": "white", "edgecolor": GRIDC, "boxstyle": "round,pad=0.35"},
)

finish(
    fig,
    "e08_how_much_do_models_disagree",
    "Sentiment moves the tail through intensity, not direction — and only by a couple of percent",
    "Difference between the M2 and M1 one-day-ahead 2.5% VaR forecasts on news-bearing days. The right panel is close\n"
    "to symmetric in tone: relative to M1, M2 makes the tail SHALLOWER on neutral text and DEEPER on confident text of\n"
    "either sign. That is the same story the M2 vs M2$_{int}$ ablation tells, seen at the level of the forecast itself.",
    f"Median absolute disagreement is {median_bp:.1f} bp, or {median_relative:.2f}% of the VaR level. Against M1, M2 shifts VaR by {neutral_text:+.3f} standardised units on neutral text, {very_adverse:+.3f} on very adverse text and {very_positive:+.3f} on very positive text — so it is the confidence of the text that deepens the tail, not its direction, and the residual asymmetry runs slightly the other way.",
    f"Source: reports/fnsipid_tail_risk_core_v1/forecasts.parquet · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 9. A data-quality problem worth knowing about
#
# While drawing a worked example this notebook hit a bad day in the underlying
# price archive. It is worth surfacing.
#
# In a correctly adjusted price series, the adjusted-close return and the raw
# close return generally diverge around distributions and splits. A large
# unexplained difference can also indicate an inconsistent adjustment factor
# and an artificial return. The local archive has no authoritative
# corporate-action metadata, so the threshold below flags candidates rather
# than proving that every row is defective.
#
# **These flags do not overturn the headline result, but common targets do not
# cancel from a nonlinear paired FZ0 loss.** The declared 2 x 2 repair
# diagnostic moved the M2-minus-M1 point estimate, while its 95% interval still
# spanned zero in both repaired cells. Candidate discontinuities may also
# inflate the measured tail.

# %%
DECLARED_PRICE_INPUT = next(row for row in CORE["inputs"] if row["role"] == "adjusted_daily_prices")
PRICE_ARCHIVE = Path(os.environ.get("FNSPID_TAIL_RISK_PRICE_ARCHIVE", "") or DECLARED_PRICE_INPUT["path"]).expanduser()
verify_input_file(
    PRICE_ARCHIVE,
    expected_sha256=DECLARED_PRICE_INPUT["sha256"],
    expected_size=int(DECLARED_PRICE_INPUT["size_bytes"]),
)
RETAINED = sorted(GARCH.loc[GARCH["firm_attrition_reason"] == "retained", "symbol"])
WINDOW = (pd.Timestamp(CORE["configuration"]["DEV_START"]), pd.Timestamp(CORE["configuration"]["EVAL_END"]))
GAP_THRESHOLD = 0.05

gap_records: list[pd.DataFrame] = []
per_firm: list[dict[str, Any]] = []
with zipfile.ZipFile(PRICE_ARCHIVE) as archive:
    members = {
        Path(name).stem.upper(): name
        for name in archive.namelist()
        if name.startswith("full_history/") and name.casefold().endswith(".csv")
    }
    for symbol in RETAINED:
        frame = pd.read_csv(archive.open(members[symbol]))
        frame.columns = [str(column).strip().casefold() for column in frame.columns]
        frame["session_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["session_date", "close", "adj close"])
        frame = (
            frame[(frame["close"] > 0) & (frame["adj close"] > 0)].drop_duplicates("session_date", keep="last").sort_values("session_date")
        )
        frame = frame[frame["session_date"].between(*WINDOW)]
        if len(frame) < 10:
            continue
        adjusted = np.diff(np.log(frame["adj close"].to_numpy(dtype=float)))
        raw = np.diff(np.log(frame["close"].to_numpy(dtype=float)))
        gap = adjusted - raw
        per_firm.append(
            {
                "symbol": symbol,
                "days": gap.size,
                "flagged": int((np.abs(gap) > GAP_THRESHOLD).sum()),
                "max_abs_gap": float(np.abs(gap).max()),
            }
        )
        flagged = np.abs(gap) > GAP_THRESHOLD
        if flagged.any():
            gap_records.append(
                pd.DataFrame({"symbol": symbol, "target_date": frame["session_date"].to_numpy()[1:][flagged], "gap": gap[flagged]})
            )

PER_FIRM_GAPS = pd.DataFrame(per_firm)
FLAGGED = pd.concat(gap_records, ignore_index=True)
FLAGGED_KEYS = set(zip(FLAGGED["symbol"], FLAGGED["target_date"], strict=True))

marked = FORECASTS.assign(flagged=[key in FLAGGED_KEYS for key in zip(FORECASTS["symbol"], FORECASTS["target_date"], strict=True)])
flagged_rows = marked[marked["flagged"]]
clean_rows = marked[~marked["flagged"]]
flagged_breach = float(flagged_rows["hit_M2"].mean())
clean_breach = float(clean_rows["hit_M2"].mean())

print(f"firm-days scanned            : {int(PER_FIRM_GAPS['days'].sum()):,}")
print(f"adjustment gaps > {GAP_THRESHOLD:.0%}        : {len(FLAGGED):,} ({len(FLAGGED) / PER_FIRM_GAPS['days'].sum():.4%})")
print(f"firms with at least one      : {int((PER_FIRM_GAPS['flagged'] > 0).sum())} of {len(PER_FIRM_GAPS)}")
print(f"breach rate on flagged days  : {flagged_breach:.1%}  vs  {clean_breach:.2%} on clean days")

# %%
fig = plt.figure(figsize=(16, 5.4))
grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

# Panel A: the smoking gun.
axis = fig.add_subplot(grid[0, 0])
with zipfile.ZipFile(PRICE_ARCHIVE) as archive:
    case = pd.read_csv(archive.open(members["MRK"]))
case.columns = [str(column).strip().casefold() for column in case.columns]
case["session_date"] = pd.to_datetime(case["date"])
case = case[case["session_date"].between("2020-06-15", "2020-07-24")].sort_values("session_date")
axis.plot(case["session_date"], case["close"], color=INK, linewidth=2.0, marker="o", markersize=3.5, label="raw close")
axis.plot(case["session_date"], case["adj close"], color=PALETTE["zero"], linewidth=2.0, marker="o", markersize=3.5, label="adjusted close")
axis.axvline(pd.Timestamp("2020-07-06"), color=PALETTE["news"], linewidth=1.8, linestyle="--", zorder=2)
axis.annotate(
    "adjustment factor steps here:\nraw close $-$3.7%, adjusted close $-$14.7%",
    xy=(pd.Timestamp("2020-07-06"), 71.5),
    xytext=(pd.Timestamp("2020-06-26"), 83.0),
    fontsize=8.5,
    color=PALETTE["news"],
    fontweight="semibold",
    arrowprops={"arrowstyle": "-|>", "color": PALETTE["news"], "linewidth": 1.5, "connectionstyle": "arc3,rad=-0.2"},
)
axis.set_ylim(64.0, 86.0)
axis.set_ylabel("price (USD)")
axis.legend(loc="lower left")
axis.set_title("Case study: MRK, July 2020")
axis.tick_params(axis="x", rotation=25)
tidy(axis)

# Panel B: how common.
axis = fig.add_subplot(grid[0, 1])
magnitudes = np.abs(FLAGGED["gap"].to_numpy()) * 100
axis.hist(np.clip(magnitudes, 5, 60), bins=40, color=PALETTE["accent"], alpha=0.9, zorder=3)
axis.set_xlabel("size of the adjustment discontinuity (%, clipped at 60)")
axis.set_ylabel("firm-days")
axis.set_title(f"{len(FLAGGED):,} flagged firm-days in {int((PER_FIRM_GAPS['flagged'] > 0).sum())} of {len(PER_FIRM_GAPS)} firms")
tidy(axis)

# Panel C: do they land in the tail?
axis = fig.add_subplot(grid[0, 2])
bars = [clean_breach, flagged_breach]
labels = [f"clean days\nn = {len(clean_rows):,}", f"flagged days\nn = {len(flagged_rows):,}"]
axis.bar([0, 1], bars, width=0.5, color=[PALETTE["M1"], PALETTE["zero"]], zorder=3)
axis.axhline(ALPHA, color=INK, linewidth=1.8, linestyle="--", zorder=4)
axis.text(1.32, ALPHA + max(bars) * 0.035, f"nominal {ALPHA:.1%}", va="bottom", ha="left", fontsize=9.5, color=INK)
for position, value in enumerate(bars):
    axis.text(position, value + max(bars) * 0.03, f"{value:.1%}", ha="center", fontsize=11, fontweight="semibold", color=INK)
axis.set_xticks([0, 1], labels)
axis.set_xlim(-0.6, 1.9)
axis.set_ylabel("M2 VaR breach rate")
axis.set_title("Flagged days land in the tail far too often")
tidy(axis)

finish(
    fig,
    "e09_price_adjustment_integrity",
    "The price archive has candidate adjustment discontinuities that may amplify tail losses",
    "Adjusted-close and raw-close returns can diverge legitimately around distributions and splits. A gap above 5%\n"
    "flags a candidate adjustment discontinuity, not a proven data error, because authoritative corporate-action\n"
    "metadata are unavailable. These days are rare but concentrated among extreme measured returns.",
    f"{len(FLAGGED):,} of {int(PER_FIRM_GAPS['days'].sum()):,} firm-days ({len(FLAGGED) / PER_FIRM_GAPS['days'].sum():.3%}) are flagged, and they breach at {flagged_breach:.0%} against {clean_breach:.1%} on unflagged days. Common targets enter each model's nonlinear FZ0 loss differently, so they can move M2-vs-M1; the declared repair check moved the estimate but its interval still spanned zero.",
    f"Source: computed from the declared input archive full_history.zip · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 10. An illustrative company
#
# What the forecasts actually look like on one ticker, through 2020. The example
# must be (a) an operating company rather than an index, sector or commodity
# fund, and (b) free of the adjusted-price discontinuities found above, so the
# picture shows forecasting rather than a data artifact.
#
# There is no local metadata that separates funds from companies, so rather than
# maintain a guess-list of fund tickers this uses a short ordered preference of
# tickers verified by hand to be operating companies in the retained cohort. It
# is a **display choice for this one picture**: no statistic anywhere in the
# experiment excludes funds, because filtering the universe after seeing results
# would be a researcher degree of freedom.

# %%
ILLUSTRATION_PREFERENCE = ("V", "NKE", "CRM", "ADBE", "MU", "AMGN", "COST")
affected = set(FLAGGED.loc[FLAGGED["target_date"] >= pd.Timestamp(CORE["configuration"]["EVAL_START"]), "symbol"])
coverage = NEWS.groupby("symbol").size().sort_values(ascending=False)
eligible_examples = [symbol for symbol in ILLUSTRATION_PREFERENCE if symbol in coverage.index and symbol not in affected]
if not eligible_examples:
    raise RuntimeError("no preferred illustration ticker survives the cleanliness filter")
EXAMPLE = eligible_examples[0]
print(f"illustration ticker: {EXAMPLE} — {int(coverage.loc[EXAMPLE]):,} news-bearing evaluation days, no flagged adjustment day")
print(f"preference order tried: {ILLUSTRATION_PREFERENCE} | eligible: {eligible_examples}")
block = FORECASTS[(FORECASTS["symbol"] == EXAMPLE) & (FORECASTS["target_date"].dt.year == 2020)].sort_values("target_date")

fig, axis = plt.subplots(figsize=(14, 6.0))
axis.fill_between(
    block["target_date"],
    block["es_return_M2"],
    block["var_return_M2"],
    color=PALETTE["M2"],
    alpha=0.16,
    zorder=2,
    label="M2 tail zone (VaR to ES)",
)
axis.plot(block["target_date"], block["target_return"], color="#5A6672", linewidth=0.9, zorder=3, label="realised return")
axis.plot(block["target_date"], block["var_return_M1"], color=PALETTE["M1"], linewidth=2.0, zorder=4, label="VaR 2.5%, M1")
axis.plot(
    block["target_date"], block["var_return_M2"], color=PALETTE["M2"], linewidth=2.0, linestyle=(0, (4, 2)), zorder=5, label="VaR 2.5%, M2"
)
axis.plot(
    block["target_date"], block["es_return_M2"], color=PALETTE["M2"], linewidth=1.2, linestyle=":", zorder=4, label="Expected Shortfall, M2"
)
breaches = block[block["hit_M2"] == 1]
axis.scatter(
    breaches["target_date"],
    breaches["target_return"],
    s=36,
    color=PALETTE["zero"],
    zorder=6,
    label=f"M2 breach ({len(breaches)} in {len(block)} days)",
    edgecolor="white",
    linewidth=0.6,
)
axis.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
axis.set_ylim(block["es_return_M2"].min() * 1.45, max(block["target_return"].max(), 0.0) * 1.35)
axis.set_ylabel("daily log return")
axis.set_xlabel("target date")
axis.legend(loc="lower right", ncol=2)
tidy(axis)

finish(
    fig,
    "e10_illustrative_firm",
    f"One company's tail forecasts through 2020: {EXAMPLE}",
    "The shaded band is the region between VaR and Expected Shortfall — how bad it gets when it goes bad. The M1 and\n"
    "M2 VaR lines are drawn on top of each other and are hard to tell apart, which is the whole finding in miniature.",
    f"{EXAMPLE} is the most news-covered retained operating company with a clean adjusted-price series. Both models widen sharply into the March 2020 crash, driven by the volatility filter rather than by text.",
    f"Source: reports/fnsipid_tail_risk_core_v1/forecasts.parquet · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 11. The honest caveat: all four models are miscalibrated
#
# Proper loss ranks the models. Calibration asks a different question — are the
# forecasts *right* in absolute terms? Here the answer is no, for all of them
# equally. A 2.5% VaR should be breached on 2.5% of days; these are breached on
# about 3.06%.
#
# This does not affect the ranking, which is a paired comparison on identical
# rows. It does bound what the result can be used for.

# %%
hit_rows = CALIBRATION[CALIBRATION["diagnostic"] == "var_hit_rate_minus_alpha"]
es_rows = CALIBRATION[CALIBRATION["diagnostic"] == "mean_es_identification_residual"]
panel_hit = float(hit_rows[(hit_rows["model"] == "M2") & (hit_rows["population"] == "full_evaluation_panel")]["hit_rate"].iloc[0])

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

axis = axes[0]
populations = [
    ("news_bearing", "news days", PALETTE["news"]),
    ("no_news", "quiet days", PALETTE["quiet"]),
    ("full_evaluation_panel", "all days", PALETTE["M2"]),
]
width = 0.26
for offset, (population, label, colour) in zip((-width, 0.0, width), populations, strict=True):
    block = hit_rows[hit_rows["population"] == population].set_index("model").reindex(MODELS)
    positions = np.arange(len(MODELS)) + offset
    axis.bar(positions, block["hit_rate"], width=width, color=colour, zorder=3, label=label)
    axis.errorbar(
        positions,
        block["hit_rate"],
        yerr=[block["hit_rate"] - (ALPHA + block["ci_low"]), (ALPHA + block["ci_high"]) - block["hit_rate"]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )
axis.axhline(ALPHA, color=PALETTE["zero"], linewidth=2.0, linestyle="--", zorder=5)
axis.text(-0.62, ALPHA - 0.0007, f"nominal {ALPHA:.1%}", ha="left", va="top", fontsize=9.5, color=PALETTE["zero"], fontweight="semibold")
axis.set_xticks(range(len(MODELS)), [m.replace("M2_intensity", "M2$_{int}$") for m in MODELS])
axis.set_ylabel("VaR breach rate")
axis.set_ylim(0.020, 0.041)
axis.legend(loc="upper center", ncol=3, fontsize=8.5)
axis.set_title("Every model breaches too often")
tidy(axis)

axis = axes[1]
per_firm_hit = FORECASTS.groupby("symbol")["hit_M2"].mean()
axis.hist(np.clip(per_firm_hit, 0, 0.09), bins=50, color=PALETTE["M2"], alpha=0.85, zorder=3)
axis.axvline(ALPHA, color=PALETTE["zero"], linewidth=2.0, linestyle="--", zorder=4)
axis.axvline(per_firm_hit.median(), color=PALETTE["good"], linewidth=2.0, zorder=5)
axis.set_ylim(0, axis.get_ylim()[1] * 1.22)
axis.text(
    ALPHA - 0.0012,
    axis.get_ylim()[1] * 0.99,
    f"nominal {ALPHA:.1%}",
    ha="right",
    va="top",
    fontsize=9.5,
    color=PALETTE["zero"],
    fontweight="semibold",
)
axis.text(
    per_firm_hit.median() + 0.0012,
    axis.get_ylim()[1] * 0.99,
    f"median firm {per_firm_hit.median():.2%}",
    ha="left",
    va="top",
    fontsize=9.5,
    color=PALETTE["good"],
    fontweight="semibold",
)
axis.set_xlabel("per-firm M2 breach rate")
axis.set_ylabel("firms")
axis.set_title(f"Spread across {len(per_firm_hit)} firms")
tidy(axis)

axis = axes[2]
block = es_rows[es_rows["population"] == "full_evaluation_panel"].set_index("model").reindex(MODELS)
positions = np.arange(len(MODELS))
axis.errorbar(
    positions,
    block["value"],
    yerr=[block["value"] - block["ci_low"], block["ci_high"] - block["value"]],
    fmt="o",
    markersize=9,
    color=PALETTE["M2"],
    ecolor=PALETTE["M2"],
    elinewidth=2.0,
    capsize=5,
    zorder=4,
)
axis.axhline(0.0, color=PALETTE["zero"], linewidth=2.0, linestyle="--", zorder=3)
axis.set_ylim(-0.0022, float(block["ci_high"].max()) * 1.20)
axis.text(
    -0.55, -0.0004, "a correct ES would sit on this line", ha="left", va="top", fontsize=9.5, color=PALETTE["zero"], fontweight="semibold"
)
axis.set_xticks(positions, [m.replace("M2_intensity", "M2$_{int}$") for m in MODELS])
axis.set_ylabel("mean ES identification residual")
axis.set_xlim(-0.6, len(MODELS) - 0.4)
axis.set_title("Expected Shortfall is not deep enough either")
tidy(axis)

finish(
    fig,
    "e11_calibration",
    "All four models are equally under-conservative — the ranking survives, absolute accuracy does not",
    "VaR breach rates against the nominal 2.5% level with 95% date-block intervals, the spread across firms, and the\n"
    "Expected Shortfall identification residual (zero if ES is right). Every interval misses its target in the same\n"
    "direction, for every model.",
    f"Breach rate is about {panel_hit:.2%} against a nominal {ALPHA:.1%}. The M2-vs-M1 comparison is paired on common rows, but nonlinear FZ0 losses need not cancel a shared scale bias; this is a relative ranking, not a deployable risk system.",
    f"Source: reports/fnsipid_tail_risk_core_v1/calibration.csv and forecasts.parquet · {SAMPLE_LINE}",
)

# %% [markdown]
# ### Why they under-cover
#
# The volatility filter is fitted on 2011-2016 and frozen. If it were correctly
# specified, standardised evaluation returns would have standard deviation 1.
# They do not — 2017-2023 was more volatile than the filter was taught to
# expect, and 2020 in particular.

# %%
by_year = FORECASTS.assign(year=FORECASTS["target_date"].dt.year).groupby("year")
year_stats = pd.DataFrame({"std_z": by_year["z_target"].std(), "breach_rate": by_year["hit_M2"].mean(), "days": by_year.size()})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

axis = axes[0]
colours = [PALETTE["zero"] if value > 1.05 else PALETTE["M1"] for value in year_stats["std_z"]]
axis.bar(year_stats.index, year_stats["std_z"], width=0.6, color=colours, zorder=3)
axis.axhline(1.0, color=INK, linewidth=1.8, linestyle="--", zorder=4)
axis.set_ylim(0.9, float(year_stats["std_z"].max()) * 1.16)
axis.text(float(year_stats.index[0]) - 0.42, 1.008, "a correctly scaled filter gives 1.0", fontsize=9.5, color=INK, va="bottom")
for year, value in year_stats["std_z"].items():
    axis.text(year, value + 0.012, f"{value:.2f}", ha="center", fontsize=9.5, color=INK)
axis.set_xticks(year_stats.index)
axis.set_ylabel("std. dev. of standardised return $z$")
axis.set_xlabel("evaluation year")
axis.set_title("The frozen filter under-predicts evaluation volatility")
tidy(axis)

axis = axes[1]
fitted = GARCH[GARCH["converged"]]
axis.scatter(fitted["persistence"], fitted["nu"].clip(upper=25), s=15, color=PALETTE["M1"], alpha=0.55, zorder=3)
axis.axvline(1.0, color=PALETTE["zero"], linewidth=1.6, linestyle="--", zorder=4)
axis.text(0.9975, 24.4, "unit root", ha="right", va="top", fontsize=9, color=PALETTE["zero"], rotation=90)
axis.set_xlim(0.6, 1.005)
axis.set_xlabel("GJR persistence   $\\alpha + \\gamma/2 + \\beta$")
axis.set_ylabel("Student-$t$ degrees of freedom (clipped at 25)")
axis.set_title(
    f"{len(fitted)} firm fits: median persistence {fitted['persistence'].median():.3f}, median $\\nu$ {fitted['nu'].median():.1f}"
)
tidy(axis, grid="both")

finish(
    fig,
    "e12_volatility_filter",
    "The volatility filter, not the text, is doing the heavy lifting — and it is stretched",
    "Left: standardised evaluation returns by year; 1.0 would mean the frozen GJR-GARCH filter got the scale right.\n"
    "Right: the fitted filters themselves. Persistence clusters just under a unit root and innovations are very fat\n"
    "tailed (median 5 degrees of freedom), which is normal for daily equity returns.",
    f"Overall std(z) is {FORECASTS['z_target'].std():.3f}, and 2020 reaches {year_stats.loc[2020, 'std_z']:.2f} — that is the main source of the over-breaching in the previous figure.",
    f"Source: reports/fnsipid_tail_risk_core_v1/forecasts.parquet and garch_fit_summary.csv · {SAMPLE_LINE}",
)

# %% [markdown]
# ## 12. Does the null hold up under the frozen robustness checks?
#
# Every check below was declared before the primary result was computed and run
# afterwards. None of them is allowed to become the headline. They are here to
# show the null is not an artifact of one arbitrary choice.

# %%
primary_row = primary.loc["d_semantic"]
forest = pd.concat(
    [
        pd.DataFrame(
            [
                {
                    "check": "PRIMARY  ·  FinBERT, GJR-GARCH, alpha = 0.025",
                    "value": primary_row["point_estimate"],
                    "ci_low": primary_row["ci_low"],
                    "ci_high": primary_row["ci_high"],
                }
            ]
        ),
        ROBUSTNESS[["check", "value", "ci_low", "ci_high"]],
    ],
    ignore_index=True,
).dropna(subset=["value"])

PRETTY = {
    "scale_versus_tail": "news-conditioned volatility scale first",
    "vader_scorer": "VADER instead of FinBERT",
    "alpha_0.01": "deeper tail, alpha = 0.01",
    "ewma_price_filter": "EWMA instead of GJR-GARCH",
    "exclude_recap_heavy_sessions": "drop price-recap-heavy news days",
    "control_recap_share": "control for recap share",
    "firm_equal_aggregation": "firm-equal, not observation-equal *",
}
forest["label"] = forest["check"].map(PRETTY).fillna(forest["check"])
covering = int(sum((row["ci_low"] <= 0 <= row["ci_high"]) for _, row in forest.iterrows()))

fig, axis = plt.subplots(figsize=(13.5, 6.4))
for position, row in forest.iterrows():
    is_primary = position == 0
    excludes = row["ci_high"] < 0 or row["ci_low"] > 0
    colour = PALETTE["M2"] if is_primary else (PALETTE["good"] if excludes else MUTED)
    axis.plot(
        [row["ci_low"], row["ci_high"]],
        [position, position],
        color=colour,
        linewidth=3.2 if is_primary else 2.2,
        solid_capstyle="round",
        zorder=3,
    )
    axis.plot(
        [row["value"]],
        [position],
        marker="D" if is_primary else "o",
        markersize=10 if is_primary else 8,
        color=colour,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=4,
    )
    axis.text(
        0.0088,
        position,
        f"{row['value']:+.2e}",
        va="center",
        fontsize=9.5,
        color=colour,
        fontweight="bold" if is_primary else "normal",
        family="monospace",
    )

axis.axvline(0.0, color=INK, linewidth=1.8, zorder=2)
axis.axvspan(-0.009, 0, color=PALETTE["good"], alpha=0.05, zorder=0)
axis.set_yticks(range(len(forest)), forest["label"], fontsize=10.5)
axis.get_yticklabels()[0].set_fontweight("bold")
axis.invert_yaxis()
axis.set_ylim(len(forest) - 0.4, -0.6)
axis.set_xlim(-0.009, 0.0125)
axis.set_xlabel("paired FZ0 difference, M2 $-$ M1   (negative = sentiment helps)")
axis.set_title("Every declared robustness check, with its 95% interval")
tidy(axis, grid="x")

finish(
    fig,
    "e13_robustness_forest",
    "The null is not an artifact of one arbitrary choice",
    "Swapping the sentiment scorer, the tail level, the volatility filter, or the treatment of price-recap headlines\n"
    "leaves the conclusion where it was. Each interval is the same date-block bootstrap as the primary.",
    f"{covering} of {len(forest)} rows cover zero. The only one that does not is the aggregation variant that discards cross-firm date dependence by construction.",
    f"Source: reports/fnsipid_tail_risk_core_v1/robustness.csv and bootstrap.csv · {SAMPLE_LINE}\n"
    "* firm-equal resampling treats firms as independent and so ignores the shared-date dependence the primary bootstrap\n"
    "  is built to respect. Its narrower interval is not stronger evidence.",
)

# %% [markdown]
# ## 13. Read-out
#
# What a reader should take away, in order of how well the evidence supports it.

# %%
d_arrival = primary.loc["d_news_vs_price"]
d_sem = primary.loc["d_semantic"]
d_tone = primary.loc["d_tone_given_intensity"]

READOUT = f"""
EVIDENCED  (what the executed run found)

  1. News ARRIVAL and VOLUME improve one-day-ahead VaR/ES forecasts beyond price
     state alone.  M1 - M0 = {d_arrival["point_estimate"]:+.3e}, 95% CI
     [{d_arrival["ci_low"]:+.3e}, {d_arrival["ci_high"]:+.3e}], {d_arrival["share_below_zero"]:.1%} of bootstrap
     replicates below zero.

  2. News SENTIMENT adds nothing measurable on top of that.
     M2 - M1 = {d_sem["point_estimate"]:+.3e}, 95% CI
     [{d_sem["ci_low"]:+.3e}, {d_sem["ci_high"]:+.3e}] - the interval covers zero.
     The effect is {abs(d_sem["point_estimate"] / d_arrival["point_estimate"]):.1%} the size of the arrival effect.

  3. Signed TONE adds nothing beyond non-neutral intensity.
     M2 - M2_intensity = {d_tone["point_estimate"]:+.3e}, CI [{d_tone["ci_low"]:+.3e}, {d_tone["ci_high"]:+.3e}].

  4. The two models barely disagree.  Median absolute VaR difference is
     {median_bp:.1f} basis points, {median_relative:.2f}% of the forecast level.  Text moves
     the tail mostly through INTENSITY, not direction.  Against M1, M2 shifts VaR
     by {neutral_text:+.3f} standardised units on neutral text but {very_adverse:+.3f} on very
     adverse text and {very_positive:+.3f} on very positive text: confident text of either
     sign deepens the tail.  That is the same story the M2 vs M2_intensity
     ablation tells, seen at the level of the forecast itself.

  5. All four models breach too often ({panel_hit:.2%} against a nominal {ALPHA:.1%}) and fail
     their conditional-coverage tests, equally.  Standardised evaluation returns
     have std {FORECASTS["z_target"].std():.3f}, so the frozen 2011-2016 volatility filter
     under-predicts 2017-2023 volatility.

  6. DATA QUALITY.  {len(FLAGGED):,} firm-days ({len(FLAGGED) / PER_FIRM_GAPS["days"].sum():.3%}) carry a candidate
     adjusted/raw-return gap larger than {GAP_THRESHOLD:.0%}.  Without authoritative corporate-action
     metadata these are flags, not proof that every row is defective.  They breach at
     {flagged_breach:.0%} versus {clean_breach:.1%} on unflagged days.  Common targets do
     not cancel from the nonlinear paired loss: the declared repair changed the M2-vs-M1
     estimate, but its 95% interval still spanned zero.  Candidate discontinuities may
     also inflate the measured tail.

INFERENCE  (bounded interpretation)

  Once you know the size of the price shock, the current volatility level, the
  market state, that news arrived and how much of it arrived, the tone of that
  news carries no further information about tomorrow's lower tail - at this
  horizon, on this panel, at calendar-date timestamp resolution.

  The design is not blind: the same rows, the same bootstrap and the same loss
  detect the arrival effect at overwhelming confidence.  Attention is
  informative about tail risk; sentiment, conditional on attention, is not.

OPEN LIMITATIONS

  - Timestamp policy.  The hashed upstream manifest records {DATE_ONLY_UPSTREAM_ROWS:,}
    date-only/exact-midnight rows under a strictly-next-session rule and
    {PRECISE_TIMESTAMP_UPSTREAM_ROWS:,} precise-timestamp rows under a containing-or-next-session rule
    before windowing and deduplication.  Original timestamps are absent from the
    completed checkpoint, so a uniformly date-only rule cannot be verified per
    retained headline.  A null here says nothing about intraday effects.
  - Survivorship and universe.  37 of 574 tickers end before 2023-12, and the
    ticker-linked cohort mixes operating firms with ETFs.
  - Calibration.  No model here is correctly calibrated, so this is a relative
    ranking, not a validated risk system.
  - Price adjustment.  See item 6; authoritative corporate-action metadata and
    a validated adjusted-price source are the highest-value upgrades to this panel.
  - One specification.  One volatility filter, one tail parameterisation.

NOT CLAIMED

  No causality.  No tradeable edge.  No claim about investor behaviour.  No
  claim of novelty.  A null is a valid result.
"""
print(READOUT)

# %% [markdown]
# ## 14. Save the figure index

# %%
index_lines = [
    "# FNSPID tail-risk result — visual explainer",
    "",
    f"Generated from the frozen run `reports/fnsipid_tail_risk_core_v1/` "
    f"(git `{CORE_GIT_LABEL}`, {CORE['run_mode']} mode, {CORE['counts']['retained_firms']} firms).",
    "",
    "This bundle re-estimates nothing. Every artifact it reads was hash-verified against the core run manifest, and the",
    "one recomputed quantity (the bootstrap replicate distribution) is asserted equal to `bootstrap.csv` before plotting.",
    "",
    "| # | Figure | What it shows |",
    "| --: | --- | --- |",
]
for position, entry in enumerate(FIGURE_INDEX, start=1):
    index_lines.append(f"| {position} | **{entry['title']}**<br>[`{entry['file']}`]({entry['file']}) | {entry['takeaway']} |")
index_lines += ["", "## Read-out", "", "```text", READOUT.strip(), "```", ""]
atomic_write_text(OUTPUT_DIR / "README.md", "\n".join(index_lines).rstrip() + "\n")

PRICE_INTEGRITY = PER_FIRM_GAPS.sort_values("max_abs_gap", ascending=False)
PRICE_INTEGRITY.to_csv(OUTPUT_DIR / "price_adjustment_integrity.csv", index=False)
FLAGGED.sort_values(["symbol", "target_date"]).to_csv(OUTPUT_DIR / "price_adjustment_flagged_days.csv", index=False)

atomic_write_json(
    OUTPUT_DIR / "manifest.json",
    {
        "schema_version": 1,
        "experiment": "fnsipid_tail_risk_explainer_v1",
        "purpose": "visual explanation of the frozen core run; no model is estimated here",
        "source_run": {
            "path": "reports/fnsipid_tail_risk_core_v1",
            "experiment": CORE["experiment"],
            "run_mode": CORE["run_mode"],
            "git_commit": CORE["git"]["commit"],
            "created_at": CORE["created_at"],
        },
        "inputs_verified": json.loads(INTEGRITY.to_json(orient="records")),
        "recomputation": {
            "bootstrap_replicates": "recomputed with the frozen block length, replications and seed, then asserted equal to bootstrap.csv to 1e-12 before plotting",
            "price_adjustment_scan": {
                "description": "read-only diagnostic over the declared input price archive; compares adjusted-close and raw-close log returns",
                "classification": "candidate adjusted/raw-return gaps only; no authoritative corporate-action metadata are available",
                "archive": str(PRICE_ARCHIVE),
                "threshold": GAP_THRESHOLD,
                "window": [str(WINDOW[0].date()), str(WINDOW[1].date())],
                "firm_days_scanned": int(PER_FIRM_GAPS["days"].sum()),
                "flagged_firm_days": int(len(FLAGGED)),
                "firms_affected": int((PER_FIRM_GAPS["flagged"] > 0).sum()),
                "breach_rate_flagged": flagged_breach,
                "breach_rate_clean": clean_breach,
                "affects_model_comparison": True,
                "reason": (
                    "common contaminated targets enter each model's nonlinear FZ0 loss differently; "
                    "the declared 2 x 2 repair moved M2-minus-M1, while repaired-cell intervals still spanned zero"
                ),
            },
        },
        "notebook_source": {
            "path": "notebooks/fnsipid_tail_risk_explainer.py",
            "sha256": sha256_file(REPO_ROOT / "notebooks" / "fnsipid_tail_risk_explainer.py"),
        },
        "figures": FIGURE_INDEX,
        "outputs": {
            str(path.relative_to(OUTPUT_DIR)): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(OUTPUT_DIR.rglob("*"))
            # manifest.json is this file. Executed-notebook and HTML renders are
            # produced by nbconvert after the kernel exits, so they can never be
            # hashed from in here; excluding them keeps the record verifiable.
            if path.is_file() and path.name != "manifest.json" and path.suffix not in {".ipynb", ".html"}
        },
    },
)
print(f"{len(FIGURE_INDEX)} figures written to {FIGURE_DIR.relative_to(REPO_ROOT)}")
print(f"index: {(OUTPUT_DIR / 'README.md').relative_to(REPO_ROOT)}")
