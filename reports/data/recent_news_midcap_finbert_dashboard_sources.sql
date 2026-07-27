-- Canonical DuckDB source map for the plot-first recent-news FinBERT dashboard.
-- The checked-in Python builder executes equivalent transformations with the
-- standard library and fails if the reconstructed headline metrics disagree
-- with the frozen replay. These queries make the chart-level source path
-- explicit for portable-dashboard provenance and independent audit.

CREATE OR REPLACE TEMP VIEW finbert_daily AS
SELECT
    CAST(session AS DATE) AS session,
    CAST(active_names AS INTEGER) AS active_names,
    CAST(gross_return AS DOUBLE) AS gross_return,
    CAST(net_return AS DOUBLE) AS net_return,
    CAST(turnover AS DOUBLE) AS turnover
FROM read_ndjson_auto(
    'results/strategy_research/baselines/'
    'recent-news-midcap-finbert-event-v1-dfdd88113a5f/daily_pnl.jsonl'
)
WHERE scorer = 'finbert'
  AND NOT final_liquidation;

-- Headline cards and the period-return chart.
WITH payload AS (
    SELECT finbert
    FROM read_json_auto(
        'results/strategy_research/baselines/'
        'recent-news-midcap-finbert-event-v1-dfdd88113a5f/metrics.json'
    )
), periods AS (
    SELECT 'Development' AS period, finbert.development AS metrics FROM payload
    UNION ALL
    SELECT 'Evaluation', finbert.evaluation FROM payload
    UNION ALL
    SELECT 'Combined', finbert.all FROM payload
)
SELECT
    period,
    metrics.cumulative_gross_return AS gross_return,
    metrics.cumulative_net_return AS net_return,
    metrics.after_cost_sharpe AS net_sharpe,
    metrics.maximum_drawdown,
    metrics.active_day_count_excluding_liquidation AS active_sessions,
    metrics.observations AS total_sessions,
    metrics.breakeven_cost_bps_per_side_approx AS breakeven_cost_bps
FROM periods;

-- Cumulative gross/net return and cost-aware drawdown.
WITH compounded AS (
    SELECT
        session,
        active_names,
        turnover,
        gross_return,
        net_return,
        EXP(SUM(LN(1 + gross_return)) OVER (ORDER BY session)) AS gross_wealth,
        EXP(SUM(LN(1 + net_return)) OVER (ORDER BY session)) AS net_wealth
    FROM finbert_daily
), with_peak AS (
    SELECT
        *,
        MAX(net_wealth) OVER (ORDER BY session ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS net_peak
    FROM compounded
)
SELECT
    session,
    CASE WHEN session >= DATE '2026-01-02' THEN 'Evaluation' ELSE 'Development' END AS period,
    gross_wealth - 1 AS cumulative_gross_return,
    net_wealth - 1 AS cumulative_net_return,
    net_wealth / net_peak - 1 AS net_drawdown,
    gross_return AS daily_gross_return,
    net_return AS daily_net_return,
    active_names,
    turnover
FROM with_peak
ORDER BY session;

-- Monthly invested, exit-only, and cash composition.
SELECT
    DATE_TRUNC('month', session) AS month,
    CASE
        WHEN active_names > 0 THEN 'Invested'
        WHEN turnover > 0 THEN 'Exit only'
        ELSE 'Cash'
    END AS status,
    COUNT(*) AS sessions
FROM finbert_daily
GROUP BY month, status
ORDER BY month, status;

-- Evaluation cost sensitivity at the five displayed per-side cost levels.
WITH costs(cost_bps) AS (VALUES (0), (5), (10), (15), (20))
SELECT
    cost_bps,
    EXP(SUM(LN(1 + gross_return - turnover * cost_bps / 10000.0))) - 1
        AS evaluation_net_return,
    COUNT(*) AS evaluation_sessions
FROM finbert_daily
CROSS JOIN costs
WHERE session >= DATE '2026-01-02'
GROUP BY cost_bps
ORDER BY cost_bps;

-- Five largest positive and negative evaluation stock contributions.
WITH payload AS (
    SELECT finbert
    FROM read_json_auto(
        'results/strategy_research/baselines/'
        'recent-news-midcap-finbert-event-v1-dfdd88113a5f/stock_metrics.json'
    )
), stocks AS (
    SELECT UNNEST(finbert) AS stock
    FROM payload
), ranked AS (
    SELECT
        stock.symbol AS symbol,
        stock.net_pnl_usd AS net_pnl_usd,
        stock.gross_pnl_usd AS gross_pnl_usd,
        stock.transaction_cost_usd AS transaction_cost_usd,
        stock.active_days AS active_days,
        stock.maximum_weight AS maximum_weight,
        ROW_NUMBER() OVER (ORDER BY stock.net_pnl_usd ASC) AS loss_rank,
        ROW_NUMBER() OVER (ORDER BY stock.net_pnl_usd DESC) AS gain_rank
    FROM stocks
)
SELECT * EXCLUDE (loss_rank, gain_rank)
FROM ranked
WHERE loss_rank <= 5 OR gain_rank <= 5
ORDER BY net_pnl_usd;

-- Evaluation uncertainty summary used in the interpretation boundary.
WITH payload AS (
    SELECT primary_vs_cash
    FROM read_json_auto(
        'results/strategy_research/baselines/'
        'recent-news-midcap-finbert-event-v1-dfdd88113a5f/bootstrap.json'
    )
)
SELECT
    primary_vs_cash.mean_daily_difference AS mean_daily_net_return,
    primary_vs_cash.confidence_interval_low AS confidence_interval_low,
    primary_vs_cash.confidence_interval_high AS confidence_interval_high,
    primary_vs_cash.confidence_level AS confidence_level,
    primary_vs_cash.replications AS replications,
    primary_vs_cash.block_length AS block_length_sessions
FROM payload;
