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
# # 01 — FNSPID firm-day panel (S1)
#
# Builds one news-bearing firm-day panel for the FNSPID coherent cohort:
# scored headlines (Moment-2 checkpoint) ∩ adjusted open ∩ SPY, with the
# LSEG earnings calendar left-joined.
#
# **No modelling.** Attrition table + coverage figures only. Licensed
# headline text is not written into the panel file.
#
# Timing: news `session_date` is inherited from the Gate-1 revised
# next-session mapping already applied in the FinBERT checkpoint. Earnings
# use BMO → same session / AMC → next session.

# %%
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "pyproject.toml").exists():
    raise RuntimeError("run from repo root or final_experiments/")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.panel import (  # noqa: E402
    PanelPaths,
    build_fnspid_firm_day_panel,
    coverage_breaks_by_month,
    write_panel_outputs,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover

    def display(obj: object) -> None:
        print(obj)


plt.rcParams.update(
    {
        "figure.figsize": (9.5, 4.2),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    }
)

OUTPUT_DIR = REPO_ROOT / "final_experiments" / "outputs" / "01_panel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("output:", OUTPUT_DIR)

# %% [markdown]
# ## 1. Build

# %%
paths = PanelPaths(output_dir=OUTPUT_DIR)
panel, attrition, meta = build_fnspid_firm_day_panel(paths)
written = write_panel_outputs(panel, attrition, meta, OUTPUT_DIR)

print(
    f"panel rows={meta['n_rows']:,}  symbols={meta['n_symbols']:,}  "
    f"sessions={meta['n_sessions']:,}  earnings share={meta['earnings_session_share']:.2%}"
)
display(attrition)
display(pd.Series({k: meta[k] for k in (
    "n_rows",
    "n_symbols",
    "n_sessions",
    "earnings_session_share",
    "missing_price_symbols",
    "primary_spine",
    "frozen_chronological_split",
)}, name="manifest_highlights"))

# %% [markdown]
# ## 2. Attrition read
#
# The executable panel is the **intersection** of cohort news days and a
# tradable adjusted open (plus SPY on that session). That selection is a
# collider — stated here, not solved here.

# %%
attrition = attrition.copy()
attrition["lost"] = attrition["surviving"].shift(1) - attrition["surviving"]
display(attrition)

# %% [markdown]
# ## 3. Coverage

# %%
monthly = coverage_breaks_by_month(panel)
breaks = monthly.loc[monthly["break_gt_fold"].fillna(False)]
print(f"months with >3× consecutive article-count change: {len(breaks)}")
display(breaks.head(12))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].plot(monthly["month"], monthly["article_count"], lw=1.3)
axes[0].set_title("FNSPID panel — articles per month")
axes[0].set_ylabel("Article count (sum over firm-days)")

firms_per_session = panel.groupby("session_date")["symbol"].nunique()
axes[1].plot(firms_per_session.index, firms_per_session.values, lw=0.8, alpha=0.85)
axes[1].set_title("Firms with news ∩ price per session")
axes[1].set_ylabel("Firms")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "coverage_monthly_and_firms.png", dpi=140)
plt.show()

# Firm-month heatmap mass
firm_month = (
    panel.assign(month=panel["session_date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["symbol", "month"], observed=True)
    .size()
    .rename("n_days")
    .reset_index()
)
per_month_firms = firm_month.groupby("month")["symbol"].nunique()
fig, ax = plt.subplots()
ax.plot(per_month_firms.index, per_month_firms.values, marker="o", ms=3)
ax.set_title("Distinct firms with ≥1 panel day per month")
ax.set_ylabel("Firms")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "coverage_firms_per_month.png", dpi=140)
plt.show()

# Article-count distribution (the low-n vs high-n mass question)
bins = [0, 1, 5, 20, 75, 10_000]
labels = ["1", "2-5", "6-20", "21-75", "76+"]
panel = panel.copy()
panel["n_bin"] = pd.cut(panel["article_count"], bins=bins, labels=labels)
n_tab = (
    panel.groupby("n_bin", observed=True)
    .agg(firm_days=("article_count", "size"), article_mass=("article_count", "sum"))
    .assign(
        share_firm_days=lambda d: d["firm_days"] / d["firm_days"].sum(),
        share_article_mass=lambda d: d["article_mass"] / d["article_mass"].sum(),
    )
)
display(n_tab)

fig, ax = plt.subplots()
ax.bar(n_tab.index.astype(str), n_tab["share_firm_days"], width=0.4, label="share of firm-days")
ax.bar(
    [i + 0.4 for i in range(len(n_tab))],
    n_tab["share_article_mass"],
    width=0.4,
    label="share of article mass",
)
ax.set_xticks([i + 0.2 for i in range(len(n_tab))], n_tab.index.astype(str))
ax.set_ylabel("Share")
ax.set_xlabel("Articles per firm-day")
ax.set_title("Where the FNSPID panel mass sits")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "firmday_n_bins.png", dpi=140)
plt.show()

# %% [markdown]
# ## 4. Earnings overlap and split labels

# %%
earn_share = (
    panel.groupby(panel["session_date"].dt.to_period("Y"))["is_earnings_session"]
    .mean()
    .rename("earnings_session_share")
)
earn_share.index = earn_share.index.to_timestamp()
fig, ax = plt.subplots()
ax.plot(earn_share.index, earn_share.values, marker="o")
ax.set_title("Share of panel firm-days flagged as earnings sessions")
ax.set_ylabel("Share")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "earnings_session_share.png", dpi=140)
plt.show()

display(panel["split"].value_counts().rename("firm_days").to_frame())
print("Chronological split frozen 2026-07-30 (chronological evaluation block).")
print(json.dumps(meta["frozen_chronological_split"], indent=2))

# %% [markdown]
# ## 5. Schema and caveats

# %%
schema = pd.DataFrame(
    {
        "column": panel.columns,
        "dtype": [str(panel[c].dtype) for c in panel.columns],
        "non_null_share": [float(panel[c].notna().mean()) for c in panel.columns],
    }
)
display(schema)
schema.to_csv(OUTPUT_DIR / "panel_schema.csv", index=False)

print("Written:")
for key, path in written.items():
    print(f"  {key}: {path}")

# %% [markdown]
# ### Caveats (carry forward)
#
# 1. **Publisher / story_family_id / availability_timestamp** are null — the
#    scored checkpoint does not carry them. A zstd rescan would be needed.
# 2. **News timing** is Gate-1 next-session (mostly date-only), not LSEG's
#    +15‑minute open rule.
# 3. **Prices** are split-adjusted opens, not dividend-adjusted.
# 4. **Selection:** rows require news and a price — collider risk for later
#    inference.
# 5. **Chronological split** is frozen (`dev≤2019-12-31`, `eval≥2020-01-01`);
#    call it a chronological evaluation block, not a pristine holdout.
