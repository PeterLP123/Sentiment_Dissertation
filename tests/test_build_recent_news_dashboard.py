from __future__ import annotations

import math
import runpy
from pathlib import Path

_DASHBOARD = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "build_recent_news_dashboard.py"))
_compound = _DASHBOARD["_compound"]
_cost_sensitivity = _DASHBOARD["_cost_sensitivity"]
_maximum_drawdown = _DASHBOARD["_maximum_drawdown"]
_monthly_activity = _DASHBOARD["_monthly_activity"]


def test_return_helpers_compound_and_measure_peak_to_trough_loss() -> None:
    returns = [0.10, -0.20, 0.05]

    assert math.isclose(_compound(returns), -0.076)
    assert math.isclose(_maximum_drawdown(returns), -0.20)


def test_activity_and_cost_views_preserve_exit_only_sessions() -> None:
    rows = [
        {
            "session": "2026-01-02",
            "active_names": 2,
            "gross_return": 0.02,
            "turnover": 1.0,
        },
        {
            "session": "2026-01-05",
            "active_names": 0,
            "gross_return": 0.0,
            "turnover": 1.0,
        },
        {
            "session": "2026-01-06",
            "active_names": 0,
            "gross_return": 0.0,
            "turnover": 0.0,
        },
    ]

    activity = {row["status"]: row["sessions"] for row in _monthly_activity(rows)}
    costs = {row["cost_bps"]: row["evaluation_net_return"] for row in _cost_sensitivity(rows)}

    assert activity == {"Invested": 1, "Exit only": 1, "Cash": 1}
    assert math.isclose(costs[0], 0.02)
    assert math.isclose(costs[10], (1.019 * 0.999) - 1.0)
