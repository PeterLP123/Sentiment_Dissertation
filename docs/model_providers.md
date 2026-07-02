# How To Configure Model Providers

Use this guide to route benchmark calls to OpenRouter, Cerebras Inference, or an Ollama server running local models such as Gemma on a desktop PC.

```mermaid
flowchart LR
    CMD["run / run-trading-strategy / TUI"] --> P["Provider adapter<br/>providers.py"]
    P -->|"provider=openrouter"| OR["OpenRouter<br/>hosted API models"]
    P -->|"provider=cerebras"| CB["Cerebras Inference<br/>high-throughput hosted models"]
    P -->|"provider=ollama"| OL["Ollama server<br/>localhost or desktop PC"]
    OL -->|"-cloud tags"| OC["Ollama Cloud<br/>signed-in daemon"]
    OR --> S["Same response record:<br/>label, latency, tokens, cost"]
    CB --> S
    OL --> S
```

## Prerequisites

- Project installed with `python -m pip install -e ".[dev]"`.
- `sentiment-bench --help` works from the repository root.
- For OpenRouter: `OPENROUTER_API_KEY`.
- For Cerebras: `CEREBRAS_API_KEY`.
- For Ollama: the official `ollama` Python package is installed through project dependencies, and the Ollama server is reachable from this machine.
- For Ollama Cloud models (`-cloud` tags): the Ollama daemon is signed in (`ollama signin`) and has internet access. See [Use Ollama Cloud Models](#use-ollama-cloud-models).

## Provider Defaults

| Setting | Default | Used by |
| --- | --- | --- |
| `SENTIMENT_BENCH_PROVIDER` | `openrouter` | CLI and TUI provider selection. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter-compatible chat endpoint. |
| `CEREBRAS_BASE_URL` | `https://api.cerebras.ai/v1` | Cerebras OpenAI-compatible endpoint. |
| `CEREBRAS_MAX_RPM` | model-specific | Optional local ceiling; live API-key quota headers take precedence. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama Python client host. |
| `SENTIMENT_BENCH_AUTO_FETCH_MODELS` | `1` | TUI auto-fetches local Ollama models when provider is Ollama and host starts with `http://localhost`. |
| `OLLAMA_NUM_PARALLEL` | Ollama default | Optional Ollama server setting for parallel request handling; the TUI displays the value it can see. |
| `--provider` | `openrouter` | Overrides the env default for one command. |
| `--ollama-host` | `http://localhost:11434` | Overrides `OLLAMA_HOST` for one command. |

## Configure OpenRouter

Add this to `.env`:

```bash
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
SENTIMENT_BENCH_PROVIDER=openrouter
```

Check available models:

```bash
sentiment-bench list-models --provider openrouter --limit 20
```

Run a pilot:

```bash
sentiment-bench run --provider openrouter \
  --models openai/gpt-4o-mini --mode pilot
```

## Configure Cerebras

Add the API key and endpoint to `.env`:

```bash
CEREBRAS_API_KEY=your-cerebras-key
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
```

In the TUI, select **Cerebras**, click **Fetch Models**, and select one or more returned model IDs. The TUI sets concurrency to 64. The client admits requests through a per-model token bucket, retries transient failures, and honors `Retry-After` or Cerebras reset headers on HTTP 429.

The model request limits recorded for this dissertation project on 2 July 2026 are shown below. They are fallback planning values only: the client serializes one initial quota probe, then enforces the live minute/hour/day headers returned for the active API key before releasing concurrent requests.

| Model | RPM | RPH | RPD | TPM | TPH | TPD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma-4-31b` | 500 | 30,000 | 720,000 | 500,000 | 30,000,000 | 720,000,000 |
| `gpt-oss-120b` | 1,000 | 60,000 | 1,440,000 | 1,000,000 | 120,000,000 | 2,000,000,000 |
| `zai-glm-4.7` | 500 | 30,000 | 720,000 | 500,000 | 30,000,000 | 720,000,000 |

These short label-only requests normally reach the request limit before the token limit. Set `CEREBRAS_MAX_RPM` only when you want a ceiling below the live API quota. Keep `max_completion_tokens=64` for hard-label accuracy tests so token quota is not reserved unnecessarily.

```bash
sentiment-bench list-models --provider cerebras --limit 20
sentiment-bench run --provider cerebras \
  --models gemma-4-31b --mode pilot --concurrency 64
```

## Configure Ollama On A Desktop PC

The repository can run on the same machine as Ollama or on another machine while the model runs on a desktop PC. The important requirement is that the Ollama server listens on an address reachable from the benchmark process.

On the machine running Ollama:

```bash
ollama pull gemma4:12b
ollama list
```

Use the exact model tag from `ollama list` in benchmark commands. For example, `gemma4:latest` and `gemma4:12b` are different run contexts and should be recorded separately.

Start Ollama so it listens on the network interface you intend to use. On macOS or Linux, a common local-network setup is:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

On Windows PowerShell:

```powershell
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

Restrict this to a trusted private network and configure the desktop firewall deliberately. Do not expose the Ollama port to the public internet.

On the machine running this repository, add:

```bash
SENTIMENT_BENCH_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
```

If the model is hosted on another computer, use its hostname or private IP address:

```bash
# Hostname example:
OLLAMA_HOST=http://desktop-pc:11434

# Private IP example:
OLLAMA_HOST=http://192.168.1.25:11434
```

Check the model list from the benchmark machine:

```bash
sentiment-bench list-models --provider ollama \
  --ollama-host http://desktop-pc:11434
```

Run a pilot with the remote Gemma model:

```bash
sentiment-bench run --provider ollama \
  --ollama-host http://desktop-pc:11434 \
  --models gemma4:12b --mode pilot --prompt-id finance_calibrated_label_only
```

For local Ollama on the same Windows machine, use:

```powershell
.\.venv\Scripts\sentiment-bench.exe run --provider ollama `
  --ollama-host http://localhost:11434 `
  --models gemma4:12b --mode pilot --prompt-id finance_calibrated_label_only
```

## Use Ollama Cloud Models

Ollama Cloud runs large hosted models (for example `gpt-oss:120b-cloud`, `minimax-m3:cloud`, and `deepseek-v4-pro:cloud`) on Ollama's infrastructure while you talk to them through a **signed-in local daemon**. No separate provider or API key is configured in this repository — the daemon proxies the request, so cloud models use the existing `ollama` provider and the same chat API as local models.

Sign in once on the machine running the Ollama daemon:

```bash
ollama signin
```

Then reference the model by its cloud tag. Ollama uses tags such as `gpt-oss:120b-cloud` for explicit variants and `minimax-m3:cloud` for untagged base models. The daemon pulls the manifest and routes inference to the cloud:

```bash
sentiment-bench run --provider ollama \
  --ollama-host http://localhost:11434 \
  --models gpt-oss:120b-cloud --mode pilot --prompt-id finance_calibrated_label_only
```

In the TUI:

1. Keep Provider on `Ollama` and the endpoint on your daemon (for example `http://localhost:11434`).
2. On the Models tab, press **Cloud Catalog** to list cloud models even if you have not pulled or set any up — they appear in the table marked in the **Cloud** column and filtered to `cloud`. You can also add a cloud model by id, or fetch models and look for the **Cloud** column. Press Enter on a row to select.
3. The Selected models summary shows how many of your chosen models are cloud (for example `3 model(s) selected (1 cloud).`).

Notes specific to cloud models:

- **Cloud Catalog** fetches Ollama's live catalogue from the public `https://ollama.com/v1/models` endpoint (no API key) and maps each base id to its runnable cloud tag, so it stays current as Ollama adds or retires models. The result is cached to `results/ollama_cloud_cache.json`; if the fetch fails (offline), it falls back to that cache and then to a small built-in list. To pin a fixed list instead, set `OLLAMA_CLOUD_MODELS` (comma-separated tags, e.g. `OLLAMA_CLOUD_MODELS=gpt-oss:120b-cloud,minimax-m3:cloud`), which takes precedence over the live fetch. Confirm a tag works against your signed-in daemon before relying on it.
- They run remotely, so the **local GPU/VRAM monitor and `Loaded in Ollama` (`/api/ps`) do not reflect them** — those panels describe only models in your local VRAM.
- They require internet access and a signed-in daemon; an unauthenticated daemon returns an error for a cloud tag.
- Record the exact cloud tag (and that the run used cloud routing) for reproducibility, just as you would a local tag.
- Reasoning-capable cloud models such as `gpt-oss` can spend a short completion budget on thinking; keep `Disable Ollama thinking` enabled for short label-only classification, the same as for local thinking models.

## Gemma 4 Notes

Gemma 4 models can spend a short completion budget on internal thinking before emitting a label. In the TUI, keep `Disable Ollama thinking` enabled for `gemma4:*` and other thinking-capable local models. The TUI highlights this recommendation when such a model is selected and stores the setting with the run request metadata.

For CLI runs, use the finance-calibrated label-only prompt and keep `--max-completion-tokens` at the default `64` or higher. If a CLI Gemma 4 pilot produces invalid outputs, retry through the TUI with `Disable Ollama thinking` enabled, then compare the run metadata.

## LSEG Trading Runs With Local Ollama

Use `configs/lseg_ollama_trading_example.toml` as the contract. Trading runs require exact tags containing a tag separator (for example `gemma3:12b`), record the digest returned by Ollama, set temperature 0 and concurrency 1, disable thinking, keep models resident with configurable `ollama_keep_alive = "30m"`, and constrain label-only responses with a positive/negative/neutral JSON enum. Score reuse also requires matching content revision/hash, model tag/digest, prompt hash, and request settings. See [LSEG to Ollama pipeline](lseg_ollama_pipeline.md).

## Use The TUI Provider Controls

Open:

```bash
sentiment-bench tui
```

In the Dashboard tab:

1. Set Provider to `OpenRouter`, `Cerebras`, or `Ollama`.
2. Set the endpoint field. Cerebras defaults to `https://api.cerebras.ai/v1`; for Ollama, use `http://desktop-pc:11434`.
3. Open the Models tab and fetch models. For local Ollama at `http://localhost:11434`, the TUI also auto-fetches installed models on startup unless `SENTIMENT_BENCH_AUTO_FETCH_MODELS=0`.
4. Select one or more models.
5. For Gemma 4, keep `Disable Ollama thinking` enabled on the Run tab.
6. Continue through Prompt and Run.

![TUI provider controls screenshot](assets/tui-dashboard.svg)

## Local Resource Monitoring

The TUI Dashboard and Run tabs include a local resource monitor. It shows:

- Provider and active endpoint.
- Selected model count and fetched provider model count.
- TUI concurrency and max completion tokens.
- `OLLAMA_NUM_PARALLEL` as visible to the TUI process.
- Ollama thinking state.
- NVIDIA GPU utilization, VRAM, power, and temperature when `nvidia-smi` is available.

Low GPU utilization during a run is not always a benchmark bug. Local model throughput can be limited by Ollama scheduling, prompt/token generation speed, CPU work, or a single active request. Increase concurrency cautiously after a successful pilot, and watch invalid-output counts as well as speed.

## Verification

OpenRouter verification:

```bash
sentiment-bench list-models --provider openrouter --limit 5
```

Ollama verification:

```bash
sentiment-bench list-models --provider ollama \
  --ollama-host http://desktop-pc:11434 --limit 5
```

A successful check prints a table of model IDs. A failed check usually means credentials, host reachability, firewall rules, or a missing model on the Ollama host need attention.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `OPENROUTER_API_KEY is required` | `.env` missing or key not exported. | Add `OPENROUTER_API_KEY` to `.env` or your shell. |
| Ollama connection refused | Ollama is not running or only bound to localhost on the model host. | Start Ollama on the host and make `OLLAMA_HOST` reachable from the benchmark machine. |
| Model not found | The model was not pulled on the Ollama host or the tag differs. | Run `ollama list`, pull the exact tag such as `ollama pull gemma4:12b`, and use that tag in the benchmark. |
| `-cloud` model fails or asks to sign in | The daemon is not signed in to Ollama Cloud or has no internet access. | Run `ollama signin` on the daemon machine, confirm connectivity, then retry with the exact `-cloud` tag. |
| Gemma 4 pilot has many invalid outputs | The model spent the short answer budget on thinking or ignored the strict label-only format. | In the TUI, keep `Disable Ollama thinking` enabled and use `finance_calibrated_label_only`. |
| Very slow local runs | Desktop model inference is slower than API routing, a single request is active, or Ollama scheduling is limiting throughput. | Use pilot mode first, review the TUI resource monitor, then increase TUI/CLI concurrency and `OLLAMA_NUM_PARALLEL` cautiously. |
| Different results across machines | Local model version, prompt settings, sampling settings, or hardware/runtime environment differ. | Set `SENTIMENT_BENCH_MACHINE_LABEL` per machine and compare model tag, provider route, prompt hash, seed, temperature, git commit, Python version, and export metadata. |
