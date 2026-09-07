# Historical Configurations

These TOML files record inputs and parameters for the benchmark application and the
pre-submission research programme. They are retained so historical runs remain
traceable; they do not supersede the frozen specifications in
[`submission/news-sentiment-beyond-mean/`](../submission/news-sentiment-beyond-mean/README.md).

| Area | Files |
| --- | --- |
| Benchmark prompts and providers | `default_prompts.toml`, `crossed_scoring.toml`, and example provider configurations |
| LSEG collections | `lseg_*.toml` files describing dated universes, windows, and workspace queries |
| Historical trading studies | `trading_*.toml` files, including preregistration and reviewed pilot variants |
| Strategy research | `strategy_research/`, with model, signal-quality, persistence, and baseline specifications |
| Tavily collection | `tavily_query_matrix.toml` and `tavily_ticker_query_matrix.toml` |

Some configurations require licensed local data, credentials, or artifacts that are
intentionally absent from a public clone. Treat dates, model IDs, samples, thresholds,
and costs as part of each recorded design. Create a new clearly named configuration
for a new experiment instead of rewriting a historical one.
