"""Shared figure style for experiments notebooks.

One place for the house rcParams and the colour roles, so figures promoted into
the dissertation read as one system instead of drifting per notebook.

Colour is assigned **by the job it does**, not by taste:

- *categorical* (identity: which signal): fixed slot order, never cycled. Charts
  needing more than about five series use small multiples instead of more hues.
- *ordinal / sequential* (magnitude: quantile 1..5, story counts): one hue,
  light to dark, so rank is readable without a legend.
- *diverging* (polarity: Sharpe, returns, anything signed): two hues with a
  **neutral grey** midpoint. Never a colour at zero — a yellow or green midpoint
  makes "no effect" look like a value.
- *status* (good/bad verdicts): reserved, never reused as a series colour.

The categorical order and the diverging pair were validated for colour-vision
deficiency separation before use. Three light-mode slots sit under 3:1 contrast
on the chart surface, so series that use them carry direct labels rather than
relying on the legend swatch alone.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#: Fixed categorical order. Assign slot 1 first and never re-order between
#: charts: colour follows the entity, not its rank in the current sort.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Ordinal steps for ranked groups (quantile 1 = lowest signal). Light to dark,
#: none lighter than the 2:1 floor against the light surface.
ORDINAL_BLUE: tuple[str, ...] = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")

#: Sequential ramp endpoints for continuous magnitude.
SEQUENTIAL_LIGHT = "#cde2fb"
SEQUENTIAL_DARK = "#0d366b"

#: Diverging poles and the neutral midpoint.
DIVERGING_LOW = "#0d366b"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#8f1f1f"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#8a8984",
    "grid": "#d8d7d2",
    "surface": "#fcfcfb",
    "reference": "#52514e",
}


# Reader-facing labels for the compact identifiers used in tables and files.
# Keep identifiers in data artifacts for reproducibility; translate them only at
# the figure boundary so dissertation plots do not read like debug output.
DISPLAY_LABELS: dict[str, str] = {
    "mean_hard_label": "Mean hard label",
    "mean_continuous": "Mean continuous score",
    "median_continuous": "Median continuous score",
    "trimmed_mean": "Trimmed mean",
    "negative_share": "Negative-story share",
    "dispersion": "Score dispersion",
    "strongest_event": "Strongest event",
    "attention_log_n": "Log-count weighted mean",
    "decayed_state": "Decayed sentiment state",
    "M_control_only": "Prior-return control",
    "M_level": "Sentiment level",
    "M_firm_only": "Firm-demeaned sentiment",
    "M_surprise": "Firm + market demeaned",
    "M_both": "Level + full surprise",
    "firm_baseline": "Firm baseline",
    "market_wide_cs": "Market-wide mean",
    "idiosyncratic": "Idiosyncratic residual",
    "fixed_band": "Fixed band",
    "logistic": "Logistic gate",
    "gradient_boosted": "Gradient-boosted gate",
    "mlp": "MLP gate",
    "mlp_label_shuffle": "Shuffled-label MLP",
    "earnings_guidance": "Earnings / guidance",
    "analyst_rating": "Analyst rating",
    "mna_strategy": "M&A / strategy",
    "legal_regulatory": "Legal / regulatory",
    "product_technology": "Product / technology",
    "management_governance": "Management / governance",
    "macro_sector": "Macro / sector",
    "commodity_rates_fx": "Commodities / rates / FX",
    "labor_esg": "Labour / ESG",
    "market_price_technical": "Market-price / technical",
    "generic_low_information": "Generic / low information",
    "other": "Other",
    "mean_sentiment_score": "All-story mean",
    "mean_reuters_sentiment_score": "Reuters-only mean",
    "mean_non_reuters_sentiment_score": "Non-Reuters mean",
    "reuters_2x_weighted_score": "Reuters 2x weighted",
    "mean_actionable_sentiment_score": "Actionable-only mean",
}


def display_label(value: Any) -> str:
    """Translate an internal identifier into a compact figure label."""
    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("_", " ").strip().capitalize())


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue → neutral grey → red, for signed quantities."""
    return LinearSegmentedColormap.from_list(
        "fe_diverging", [DIVERGING_LOW, "#6da7ec", DIVERGING_MID, "#d98080", DIVERGING_HIGH]
    )


def sequential_cmap() -> LinearSegmentedColormap:
    """One hue, light → dark, for unsigned magnitude."""
    return LinearSegmentedColormap.from_list(
        "fe_sequential", [SEQUENTIAL_LIGHT, "#5598e7", SEQUENTIAL_DARK]
    )


def apply_house_style() -> None:
    """Recessive chrome: thin marks, muted grid, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.figsize": (9.5, 4.2),
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK["muted"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.grid": True,
            "grid.color": INK["grid"],
            "grid.alpha": 0.7,
            "grid.linewidth": 0.7,
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "text.color": INK["primary"],
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "savefig.facecolor": INK["surface"],
            # No global savefig.bbox: combining a global "tight" with per-figure
            # tight_layout and a raised suptitle can collapse the bbox. Pass
            # bbox_inches="tight" explicitly on the figures that need it.
        }
    )


def label_line_end(ax: Any, x: Any, y: Any, text: str, colour: str, **kwargs: Any) -> None:
    """Direct-label a series at its right end.

    Required relief for the lower-contrast slots, and it removes the
    legend-to-line lookup on any multi-series chart.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        color=INK["secondary"],
        **kwargs,
    )


def label_line_ends(
    ax: Any,
    entries: list[tuple[float, float, str, str]],
    *,
    min_gap_frac: float = 0.055,
) -> None:
    """Direct-label several series at their right ends without collisions.

    ``entries`` is ``(x, y, text, colour)``. Labels are nudged apart vertically
    to at least ``min_gap_frac`` of the axis range, so a legend lookup is never
    needed even when two series converge.
    """
    if not entries:
        return
    low, high = ax.get_ylim()
    gap = abs(high - low) * min_gap_frac
    ordered = sorted(entries, key=lambda e: e[1])

    # Sweep upward pushing each label clear of the one below it, then shift the
    # whole stack back so the nudging stays centred on the true values.
    placed: list[float] = []
    for _, y, _, _ in ordered:
        placed.append(y if not placed else max(y, placed[-1] + gap))
    drift = sum(placed) / len(placed) - sum(e[1] for e in ordered) / len(ordered)
    placed = [p - drift for p in placed]

    for (x, y, text, colour), label_y in zip(ordered, placed, strict=True):
        ax.annotate(
            text,
            xy=(x, label_y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=colour,
            annotation_clip=False,
        )
        if abs(label_y - y) > gap * 0.35:
            # Leader so a nudged label still points at its own line.
            ax.plot([x, x], [y, label_y], color=colour, lw=0.7, alpha=0.55, zorder=2)


def zero_line(ax: Any, *, axis: str = "y") -> None:
    """Recessive reference line at zero — the baseline every signed chart needs."""
    if axis == "y":
        ax.axhline(0.0, color=INK["reference"], lw=1.0, zorder=1)
    else:
        ax.axvline(0.0, color=INK["reference"], lw=1.0, zorder=1)


def annotate_source(fig: Any, text: str) -> None:
    """One-line provenance under a figure, so a promoted plot carries its own caption."""
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.5, color=INK["muted"])
