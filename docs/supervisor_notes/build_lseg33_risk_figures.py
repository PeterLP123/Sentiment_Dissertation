"""Build the licence-safe LSEG-33 figures used in the 2026-08-12 note.

The script only reads aggregate state and inference outputs from notebooks 72
and 73.  It does not read or reproduce licensed Reuters text.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep Matplotlib's cache out of the research workspace.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "sentiment_dissertation_matplotlib"),
)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECENT_DIR = (
    ROOT / "final_experiments/outputs/72_lseg_negative_pressure_har_transfer"
)
EARLIER_DIR = (
    ROOT
    / "final_experiments/outputs/73_lseg_negative_pressure_har_backward_transfer"
)
HYBRID_DIR = (
    ROOT
    / "final_experiments/outputs/36_lseg_gemma_finbert_hybrid_alpha_audit"
)
LONG_CONTROL_DIR = (
    ROOT
    / "final_experiments/outputs/60_lseg_gemma_hybrid_long_leg_exposure_control"
)
RESIDUAL_DIR = (
    ROOT
    / "final_experiments/outputs/64_lseg_gemma_hybrid_long_leg_residual_portfolios"
)
JOINED_HYBRID_DIR = (
    ROOT
    / "final_experiments/outputs/67_lseg_gemma_finbert_joined_long_window"
)
FIGURE_DIR = ROOT / "dissertation/figures"

BLUE = "#2369A8"
AMBER = "#E69F00"
RED = "#C23B22"
GREY = "#A7ADB4"
INK = "#202124"


def read_state(path: Path) -> pd.DataFrame:
    """Read a saved hysteresis state with typed dates and booleans."""

    frame = pd.read_csv(path, parse_dates=["session_date"])
    if frame["risk_off"].dtype != bool:
        frame["risk_off"] = frame["risk_off"].astype(str).str.lower().eq("true")
    return frame


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DCE1", linewidth=0.8, alpha=0.8)
    axis.tick_params(colors="#4D5156")
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def read_earlier_strategies() -> dict[str, pd.DataFrame]:
    """Read and align the three saved LSEG-33 daily strategy paths."""

    files = {
        "HAR": "frozen_har_daily.parquet",
        "News-timed overlay": "har_negative_pressure_overlay_daily.parquet",
        "Constant lower exposure": "har_constant_matched_multiplier_daily.parquet",
    }
    strategies: dict[str, pd.DataFrame] = {}
    reference_dates: pd.Series | None = None
    for label, filename in files.items():
        frame = pd.read_parquet(EARLIER_DIR / filename).copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"])
        frame = frame.sort_values("session_date").reset_index(drop=True)
        if reference_dates is None:
            reference_dates = frame["session_date"]
        elif not reference_dates.equals(frame["session_date"]):
            raise ValueError(f"Strategy dates do not align for {label}")
        strategies[label] = frame
    return strategies


def pounds_axis(value: float, _position: float) -> str:
    """Format a pounds axis compactly in thousands."""

    sign = "-" if value < 0 else ""
    return f"{sign}£{abs(value) / 1_000:.0f}k"


def plot_activation_windows() -> None:
    earlier = read_state(EARLIER_DIR / "backward_hysteresis_state.csv")
    recent = read_state(RECENT_DIR / "recent_hysteresis_state.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.8), sharey=True)
    panels = [
        (
            axes[0],
            earlier,
            "Earlier LSEG-33 window: the rule activates",
            "53 of 456 sessions risk-off (18 entries)",
        ),
        (
            axes[1],
            recent,
            "Recent LSEG-33 window: the same rule stays inactive",
            "0 of 167 sessions risk-off; maximum z = 1.35",
        ),
    ]

    for axis, frame, title, note in panels:
        visible = frame.loc[frame["negative_pressure_z"].notna()].copy()
        axis.plot(
            visible["session_date"],
            visible["negative_pressure_z"],
            color=BLUE,
            linewidth=1.55,
            label="Negative-news pressure",
        )
        axis.fill_between(
            frame["session_date"],
            0,
            1,
            where=frame["risk_off"],
            transform=axis.get_xaxis_transform(),
            color=AMBER,
            alpha=0.20,
            linewidth=0,
            label="Reduced-exposure state",
        )
        axis.axhline(1.5, color=RED, linestyle="--", linewidth=1.3)
        axis.axhline(0.5, color="#6F747A", linestyle=":", linewidth=1.0)
        axis.text(
            0.99,
            0.93,
            note,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        axis.set_ylabel("Pressure z-score")
        axis.set_ylim(-2.55, 4.9)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        style_axis(axis)

    axes[0].text(
        1.0,
        1.03,
        "Enter risk-off at 1.5; leave at 0.5",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#5F6368",
    )
    axes[1].set_xlabel("Session date")
    axes[1].legend(
        handles=[
            plt.Line2D([0], [0], color=BLUE, linewidth=1.8),
            plt.Rectangle((0, 0), 1, 1, color=AMBER, alpha=0.25),
        ],
        labels=["Negative-news pressure", "Reduced-exposure state"],
        loc="lower left",
        frameon=False,
        ncols=2,
        fontsize=9.5,
    )
    fig.suptitle(
        "The unchanged risk rule behaves differently across the two LSEG-33 windows",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        "The earlier z-score begins after the fixed 126-session warm-up. Shading marks sessions with reduced exposure.",
        fontsize=9,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=1.45)
    fig.savefig(
        FIGURE_DIR / "fig_lseg33_risk_activation.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_timing_placebo() -> None:
    null = pd.read_csv(EARLIER_DIR / "circular_shift_null.csv")
    tests = pd.read_csv(EARLIER_DIR / "timing_tests.csv")
    shifted = null.loc[null["shift"] != 0]
    observed = null.loc[null["shift"] == 0].iloc[0]

    return_q = float(
        tests.loc[
            tests["estimand"].eq("overlay minus HAR net return"),
            "q_bh_two_test",
        ].iloc[0]
    )
    downside_q = float(
        tests.loc[
            tests["estimand"].eq("HAR minus overlay downside squared return"),
            "q_bh_two_test",
        ].iloc[0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    specifications = [
        (
            axes[0],
            shifted["mean_net_difference"] * 10_000,
            float(observed["mean_net_difference"]) * 10_000,
            "Return: no reliable improvement",
            "Net return change (bps per session)",
            f"BH q = {return_q:.3f}",
        ),
        (
            axes[1],
            shifted["downside_reduction"] * 100_000_000,
            float(observed["downside_reduction"]) * 100_000_000,
            "Downside timing: better than shifted dates",
            "Downside-loss reduction (bps² per session)",
            f"BH q = {downside_q:.3f}",
        ),
    ]

    for axis, values, actual, title, xlabel, q_label in specifications:
        axis.hist(values, bins=24, color=GREY, edgecolor="white", linewidth=0.6)
        axis.axvline(actual, color=BLUE, linewidth=2.6, label="Actual schedule")
        axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Shifted schedules")
        axis.text(
            0.98,
            0.91,
            q_label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color=INK,
        )
        axis.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), frameon=False)
        style_axis(axis)

    fig.suptitle(
        "Earlier LSEG-33 window: compare the real schedule with every shifted schedule",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.text(
        0.01,
        -0.02,
        "The blue line is the observed timing; grey bars are all 455 non-zero circular date shifts.",
        fontsize=9,
        color="#5F6368",
    )
    fig.tight_layout(w_pad=2.5)
    fig.savefig(
        FIGURE_DIR / "fig_lseg33_timing_placebo.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_economic_waterfall() -> None:
    """Decompose the overlay's return difference on a £1 million notional."""

    strategies = read_earlier_strategies()
    har = strategies["HAR"]
    overlay = strategies["News-timed overlay"]
    notional = 1_000_000.0

    gross_difference = overlay["gross_return"] - har["gross_return"]
    underlying_down = har["forward_return"] < 0
    components = np.array(
        [
            gross_difference.where(underlying_down, 0.0).sum(),
            gross_difference.where(~underlying_down, 0.0).sum(),
            -(overlay["cost"] - har["cost"]).sum(),
        ]
    ) * notional
    labels = ["Losses avoided\non down days", "Upside given up\non up days", "Extra\ntrading costs"]
    cumulative = np.r_[0.0, np.cumsum(components)]
    total = float(components.sum())

    fig, axis = plt.subplots(figsize=(10.8, 5.5))
    x = np.arange(4)
    for index, value in enumerate(components):
        bottom = cumulative[index] if value >= 0 else cumulative[index + 1]
        axis.bar(
            x[index],
            abs(value),
            bottom=bottom,
            width=0.62,
            color=BLUE if value >= 0 else AMBER,
            edgecolor=INK,
            linewidth=0.65,
        )
        axis.text(
            x[index],
            bottom + abs(value) / 2,
            f"{value / 1_000:+.1f}k",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if abs(value) > 25_000 else INK,
        )
        if index < len(components) - 1:
            axis.plot(
                [x[index] + 0.31, x[index + 1] - 0.31],
                [cumulative[index + 1], cumulative[index + 1]],
                color="#72777D",
                linewidth=0.9,
            )

    axis.bar(
        x[3],
        abs(total),
        bottom=min(0.0, total),
        width=0.62,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.9,
    )
    axis.text(
        x[3],
        total + 5_000,
        f"+£{total / 1_000:.1f}k",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.set_xticks(x, labels + ["Net arithmetic\ndifference"])
    axis.set_ylabel("Contribution per £1 million")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(pounds_axis))
    axis.set_title(
        "Economic decomposition of the LSEG-33 risk overlay",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=28,
    )
    axis.text(
        0.0,
        1.012,
        "Earlier 456-session window; daily gross-return effects plus incremental costs",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#5F6368",
    )
    style_axis(axis)
    axis.grid(axis="x", visible=False)
    fig.text(
        0.01,
        0.005,
        "The compounded ending-wealth difference is +£10.9k per £1m. The mean return difference is not reliable: "
        "95% block-bootstrap interval −1.30 to +1.69 bps/session (p = 0.849).",
        fontsize=9.2,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(
        FIGURE_DIR / "fig_lseg33_economic_waterfall.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_cumulative_economic_value() -> None:
    """Show when protection was earned and when its cost was paid."""

    strategies = read_earlier_strategies()
    har = strategies["HAR"]
    overlay = strategies["News-timed overlay"]
    notional = 1_000_000.0

    gross_difference = overlay["gross_return"] - har["gross_return"]
    underlying_down = har["forward_return"] < 0
    loss_saving = gross_difference.where(underlying_down, 0.0) * notional
    sacrifice = (
        gross_difference.where(~underlying_down, 0.0)
        - (overlay["cost"] - har["cost"])
    ) * notional
    net = (overlay["net_return"] - har["net_return"]) * notional

    fig, axis = plt.subplots(figsize=(11.2, 5.3))
    dates = har["session_date"]
    axis.plot(
        dates,
        loss_saving.cumsum(),
        color=BLUE,
        linewidth=2.1,
        label="Losses avoided on down days",
    )
    axis.plot(
        dates,
        sacrifice.cumsum(),
        color=AMBER,
        linewidth=2.0,
        linestyle="--",
        label="Upside and costs sacrificed",
    )
    axis.plot(
        dates,
        net.cumsum(),
        color=INK,
        linewidth=2.4,
        label="Net arithmetic difference",
    )
    midpoint = dates.iloc[len(dates) // 2 - 1]
    axis.axvline(midpoint, color="#7A7F85", linewidth=1.0, linestyle=":")
    axis.text(
        midpoint,
        axis.get_ylim()[1],
        "  Halfway point",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5F6368",
    )
    axis.axhline(0, color="#70757A", linewidth=0.8)
    axis.set_ylabel("Cumulative contribution per £1 million")
    axis.set_xlabel("Session date")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(pounds_axis))
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=9))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.set_title(
        "Cumulative economic contributions through the earlier LSEG-33 window",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=28,
    )
    axis.text(
        0.0,
        1.015,
        "Uncompounded daily differences on a fixed £1m notional; 53 risk-off sessions across 18 episodes",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#5F6368",
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.985),
        frameon=False,
        ncols=3,
        fontsize=9.2,
    )
    style_axis(axis)
    fig.text(
        0.01,
        0.005,
        "The net contribution was about +£19.6k in the first half and −£13.0k in the second half. "
        "This time instability is why the return claim remains null.",
        fontsize=9.2,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(
        FIGURE_DIR / "fig_lseg33_cumulative_economic_value.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_economic_risk_metrics() -> None:
    """Compare downside deviation and drawdown on an intuitive scale."""

    strategies = read_earlier_strategies()
    labels = list(strategies)
    colors = [GREY, BLUE, AMBER]
    downside_deviation = [
        np.sqrt(np.mean(np.minimum(frame["net_return"], 0.0) ** 2) * 252) * 100
        for frame in strategies.values()
    ]
    maximum_drawdown = []
    for frame in strategies.values():
        wealth = (1.0 + frame["net_return"]).cumprod()
        maximum_drawdown.append(abs(float((wealth / wealth.cummax() - 1.0).min())) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    panels = [
        (
            axes[0],
            downside_deviation,
            "Annualised downside deviation",
            "Lower means smaller negative-return variability",
        ),
        (
            axes[1],
            maximum_drawdown,
            "Maximum peak-to-trough drawdown",
            "Lower means less capital lost at the worst point",
        ),
    ]
    y = np.arange(len(labels))
    for axis, values, title, subtitle in panels:
        axis.barh(y, values, color=colors, edgecolor=INK, linewidth=0.6, height=0.58)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.22)
        for row, value in enumerate(values):
            axis.text(
                value + max(values) * 0.025,
                row,
                f"{value:.2f}%",
                va="center",
                ha="left",
                fontsize=10.5,
                fontweight="bold" if labels[row] == "News-timed overlay" else "normal",
                color=INK,
            )
        axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=24)
        axis.text(
            0.0,
            1.01,
            subtitle,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#5F6368",
        )
        axis.set_xlabel("Percent")
        style_axis(axis)
        axis.grid(axis="y", visible=False)

    fig.suptitle(
        "Risk measures in the earlier LSEG-33 window",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.text(
        0.01,
        -0.025,
        "The news-timed overlay has the lowest downside deviation. The constant lower-exposure control has the smallest "
        "maximum drawdown, so timing is not best on every risk measure.",
        fontsize=9.2,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.92), w_pad=2.5)
    fig.savefig(
        FIGURE_DIR / "fig_lseg33_economic_risk_metrics.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_lseg_economic_case_and_limit() -> None:
    """Put the strongest LSEG economics beside the historical failure."""

    hybrid = pd.read_parquet(HYBRID_DIR / "capped_hybrid_daily.parquet").copy()
    residual = pd.read_parquet(RESIDUAL_DIR / "market_residual_daily.parquet").copy()
    controls = pd.read_csv(LONG_CONTROL_DIR / "control_inference.csv")
    windows = pd.read_csv(JOINED_HYBRID_DIR / "window_results.csv")
    for frame in (hybrid, residual):
        frame["return_end_date"] = pd.to_datetime(frame["return_end_date"])

    strategies = {
        "Recent hybrid": (hybrid, BLUE),
        "Lower-risk market residual": (residual, AMBER),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))

    for label, (frame, color) in strategies.items():
        wealth = (1.0 + frame["net_return"]).cumprod()
        axes[0, 0].plot(
            frame["return_end_date"],
            wealth,
            color=color,
            linewidth=2.1,
            label=label,
        )
    axes[0, 0].axhline(1.0, color="#777777", linewidth=0.8)
    axes[0, 0].set_title("Recent 167-session after-cost paths", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Growth of £1")
    axes[0, 0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0, 0].legend(frameon=False, fontsize=9)
    style_axis(axes[0, 0])

    scorecard_rows = []
    for label, (frame, _color) in strategies.items():
        wealth = (1.0 + frame["net_return"]).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        scorecard_rows.append(
            {
                "label": label,
                "profit": float(wealth.iloc[-1] - 1.0) * 1_000_000,
                "drawdown": abs(float(drawdown.min())) * 1_000_000,
            }
        )
    scorecard = pd.DataFrame(scorecard_rows)
    x = np.arange(len(scorecard))
    width = 0.34
    axes[0, 1].bar(
        x - width / 2,
        scorecard["profit"],
        width,
        color=BLUE,
        label="Ending profit",
    )
    axes[0, 1].bar(
        x + width / 2,
        scorecard["drawdown"],
        width,
        color=AMBER,
        label="Worst drawdown",
    )
    axes[0, 1].set_xticks(x, ["Hybrid", "Market residual"])
    axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(pounds_axis))
    axes[0, 1].set_ylabel("£ per £1 million")
    axes[0, 1].set_title("Economic scale at 10 bps per side", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=9)
    for container in axes[0, 1].containers:
        axes[0, 1].bar_label(
            container,
            labels=[f"£{value / 1_000:.1f}k" for value in container.datavalues],
            padding=3,
            fontsize=9,
        )
    style_axis(axes[0, 1])
    axes[0, 1].grid(axis="x", visible=False)

    comparison_order = [
        "selected_long_minus_equal_weight_market",
        "selected_long_minus_same_sector_equal_weight",
    ]
    comparison_labels = ["Same-day market", "Same-sector stocks"]
    selected = controls.set_index("comparison").loc[comparison_order]
    estimates = selected["mean_difference_bps_session"].to_numpy(dtype=float)
    lower = selected["ci_low_bps_session"].to_numpy(dtype=float)
    upper = selected["ci_high_bps_session"].to_numpy(dtype=float)
    y = np.arange(len(selected))
    axes[1, 0].errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
        linewidth=1.8,
        markersize=7,
    )
    axes[1, 0].axvline(0, color="#777777", linewidth=0.8)
    axes[1, 0].set_yticks(y, comparison_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Selected-long advantage (gross bps/session)")
    axes[1, 0].set_title("Selected names beat exposure controls", loc="left", fontweight="bold")
    for row, q_value in enumerate(selected["bh_q_value"]):
        axes[1, 0].text(
            upper[row] + 0.25,
            row,
            f"BH q={q_value:.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    style_axis(axes[1, 0])

    window_order = ["backward_2024_2025", "opened_2025_2026"]
    window_labels = ["Earlier unchanged replay", "Recent opened window"]
    window_rows = windows.set_index("window").loc[window_order]
    values = window_rows["total_return_net"].to_numpy(dtype=float) * 1_000_000
    bars = axes[1, 1].bar(
        window_labels,
        values,
        color=[RED, BLUE],
        edgecolor=INK,
        linewidth=0.6,
        width=0.62,
    )
    axes[1, 1].axhline(0, color="#666666", linewidth=0.8)
    axes[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(pounds_axis))
    axes[1, 1].set_ylabel("Ending profit per £1 million")
    axes[1, 1].set_title("The economic result is not stable through time", loc="left", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=8)
    axes[1, 1].set_ylim(min(-50_000.0, float(values.min()) * 1.25), float(values.max()) * 1.16)
    axes[1, 1].bar_label(
        bars,
        labels=[f"{value / 1_000:+.1f}k" for value in values],
        padding=4,
        fontsize=10,
        fontweight="bold",
    )
    style_axis(axes[1, 1])
    axes[1, 1].grid(axis="x", visible=False)

    fig.suptitle(
        "The strongest LSEG economic result is material, but not historically stable",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=1.01,
    )
    fig.text(
        0.01,
        -0.012,
        "Recent hybrid: Gemma event timing plus FinBERT name ranking on the fixed 33-company set. "
        "The unchanged earlier replay is the binding robustness failure; recent-window results remain retrospective.",
        fontsize=9,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=2.2, w_pad=2.0)
    fig.savefig(
        FIGURE_DIR / "fig_lseg33_economic_case_and_limit.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_activation_windows()
    plot_timing_placebo()
    plot_economic_waterfall()
    plot_cumulative_economic_value()
    plot_economic_risk_metrics()
    plot_lseg_economic_case_and_limit()


if __name__ == "__main__":
    main()
