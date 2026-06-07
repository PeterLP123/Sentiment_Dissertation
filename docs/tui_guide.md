# How To Use The Terminal UI

The TUI is the interactive front end for the same benchmark, provider, results, and Tavily workflows exposed by the CLI.

![TUI screenshot](assets/tui-news-tab.svg)

## Prerequisites

- Project installed with `python -m pip install -e ".[dev]"`.
- `Data/data.csv` available.
- Provider credentials or Ollama host configured if you plan to run models.
- `TAVILY_API_KEY` configured if you plan to use the News tab.

## Open The TUI

```bash
sentiment-bench tui
```

The app stores local session preferences in `results/tui_session.json`.

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `1` | Dashboard |
| `2` | Models |
| `3` | Prompt |
| `4` | Run |
| `5` | Results |
| `6` | News |
| `r` | Refresh active data |
| `s` | Start a benchmark run |
| `c` | Cancel current run |
| `?` | Help |

## Tab Guide

| Tab | Purpose | Typical actions |
| --- | --- | --- |
| Dashboard | Shows dataset, provider, and run readiness. | Pick provider and endpoint, confirm credentials, review selected models. |
| Models | Fetches and filters provider models. | Search, select one or more models, confirm provider availability. |
| Prompt | Edits prompt configuration. | Pick prompt preset, review prompt text, save a custom TUI prompt. |
| Run | Configures experiment settings. | Choose pilot/full mode, seed, temperature, tokens, few-shot settings, and baselines. |
| Results | Loads stored run metrics. | Inspect metrics, confusion matrices, misclassifications, exports, and figures. |
| News | Sources Tavily article corpora. | Check API, fetch articles, review output path and extraction failures. |

## Run A Benchmark

1. Open Dashboard and select provider.
2. Enter the provider endpoint:

   ```text
   https://openrouter.ai/api/v1
   ```

   or:

   ```text
   http://desktop-pc:11434
   ```

3. Open Models and fetch models.
4. Select one or more model IDs.
5. Open Prompt and choose `default_label_only` or another prompt.
6. Open Run, keep `pilot` for the first pass, and start the run.
7. Open Results when the run finishes.

## Source News

1. Open News.
2. Enter a query such as:

   ```text
   bank earnings sentiment
   ```

3. Choose topic and time range.
4. Keep extraction enabled unless you only need snippets.
5. Click Check Tavily for a low-cost connectivity check.
6. Click Fetch News.
7. Read the log line for the saved output path and failed extraction count.

The News tab writes the same files as `sentiment-bench fetch-news`: `articles.jsonl`, `articles.csv`, and `manifest.json`.

## Export Results

In the Results tab:

1. Select a completed run.
2. Review primary and all-scope metrics.
3. Use Export Run to write `results/exports/run_<id>/`.
4. Use View Figures if the optional plotting extra is installed.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Models tab shows no models | Provider credential or endpoint problem. | Check `.env`, provider selector, endpoint, and firewall. |
| Run button refuses to start | Missing model selection or invalid settings. | Review Dashboard readiness and Run validation hints. |
| Ollama run fails from Mac | Desktop Ollama server not reachable. | Check `OLLAMA_HOST`, desktop firewall, and `ollama list` on the desktop. |
| News check fails | `TAVILY_API_KEY` missing or invalid. | Add the key to `.env`, restart the TUI, and retry Check Tavily. |
| Figures do not open | Plotting extra not installed or OS open failed. | Run `python -m pip install -e ".[figures]"` and export again. |
