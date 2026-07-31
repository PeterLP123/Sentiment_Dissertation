"""Focused tests for final_experiments evaluation / turnover helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from final_experiments.lib.evaluate import (
    TradeConfig,
    block_bootstrap_mean,
    breakeven_bps_per_side,
    build_daily_portfolio,
    cross_sectional_rank_scores,
    signal_to_position,
    summarize_daily_portfolio,
)


def test_signal_to_position_band_and_orient() -> None:
    assert signal_to_position(0.5, threshold=0.2) == 1
    assert signal_to_position(-0.5, threshold=0.2) == -1
    assert signal_to_position(0.1, threshold=0.2) == 0
    assert signal_to_position(0.5, threshold=0.0, orient=-1.0) == -1


def test_daily_portfolio_turnover_and_breakeven() -> None:
    # Day 1: long A, short B. Day 2: flip both → high turnover.
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "session_date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]
            ),
            "split": ["development"] * 4,
            "signal": [1.0, -1.0, -1.0, 1.0],
            "ar_open_h1": [0.02, -0.02, 0.01, -0.01],
        }
    )
    cfg = TradeConfig(threshold=0.0, cost_bps_per_side=10.0, min_names=2)
    daily = build_daily_portfolio(frame, "signal", config=cfg, orient=1.0, split="development")
    assert len(daily) == 2
    # Day 1 weights: A=+0.5, B=-0.5; gross = 0.5*0.02 + (-0.5)*(-0.02) = 0.02
    assert abs(daily.loc[0, "gross_return"] - 0.02) < 1e-12
    # Day 1 vs flat: turnover 0.5*(|0.5|+|-0.5|) = 0.5
    assert abs(daily.loc[0, "turnover"] - 0.5) < 1e-12
    # Day 2 flip: each weight changes by 1.0 → turnover 0.5*(1+1)=1.0
    assert abs(daily.loc[1, "turnover"] - 1.0) < 1e-12

    be = breakeven_bps_per_side(daily)
    assert be is not None
    assert abs(
        be - 10_000.0 * daily["gross_return"].mean() / (2.0 * daily["turnover"].mean())
    ) < 1e-9

    summary = summarize_daily_portfolio(daily, config=cfg)
    assert summary["n_sessions"] == 2
    assert summary["breakeven_bps_per_side"] == be
    # net = gross - 2 * half-L1 turnover * 10bps/side
    expected_net0 = 0.02 - 2.0 * 0.5 * 0.001
    assert abs(daily.loc[0, "net_return"] - expected_net0) < 1e-12


def test_flat_day_charges_exit_turnover_once() -> None:
    # Day 1 trades A and B; day 2 has a single name, below min_names, so the book
    # goes flat. Exiting a gross-1 book costs exactly 0.5 one-sided turnover --
    # counting the departed names twice would report 1.0.
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
            "split": ["development"] * 3,
            "signal": [1.0, -1.0, 1.0],
            "ar_open_h1": [0.02, -0.02, 0.03],
        }
    )
    cfg = TradeConfig(threshold=0.0, cost_bps_per_side=10.0, min_names=2)
    daily = build_daily_portfolio(frame, "signal", config=cfg, orient=1.0, split="development")
    assert abs(daily.loc[1, "turnover"] - 0.5) < 1e-12
    assert daily.loc[1, "gross_return"] == 0.0


def test_cs_rank_scores_are_centred_and_span_the_session() -> None:
    z = cross_sectional_rank_scores(np.array([10.0, 20.0, 30.0]))
    assert z.tolist() == [-1.0, 0.0, 1.0]
    # A constant session has no cross-sectional information: no trade.
    assert cross_sectional_rank_scores(np.array([5.0, 5.0, 5.0])).tolist() == [0.0, 0.0, 0.0]


def test_cs_rank_trades_one_sided_signals_both_ways() -> None:
    # dispersion-like signal: non-negative, so sign() can only ever short it.
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "session_date": pd.to_datetime(["2020-01-02"] * 4),
            "split": ["development"] * 4,
            "signal": [0.1, 0.2, 0.3, 0.4],
            "ar_open_h1": [0.01, 0.0, 0.0, -0.01],
        }
    )
    sign_daily = build_daily_portfolio(
        frame, "signal", config=TradeConfig(position_mode="sign"), orient=-1.0, split="development"
    )
    assert sign_daily.loc[0, "n_long"] == 0
    assert abs(sign_daily.loc[0, "net_exposure"] + 1.0) < 1e-12

    rank_daily = build_daily_portfolio(
        frame, "signal", config=TradeConfig(position_mode="cs_rank"), orient=-1.0, split="development"
    )
    assert rank_daily.loc[0, "n_long"] > 0
    assert rank_daily.loc[0, "n_short"] > 0
    assert abs(rank_daily.loc[0, "net_exposure"]) < 1e-12
    # orient=-1 means low dispersion is the long leg, so A (0.1) is long.
    assert rank_daily.loc[0, "gross_return"] > 0


def test_block_bootstrap_is_seeded() -> None:
    x = np.linspace(-0.01, 0.01, 40)
    a = block_bootstrap_mean(x, block_length=5, replications=200, seed=7)
    b = block_bootstrap_mean(x, block_length=5, replications=200, seed=7)
    assert a["mean"] == b["mean"]
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]
