# How To Configure Model Providers

Use this guide to route benchmark calls to OpenRouter or to an Ollama server running local models such as Gemma on a desktop PC.

## Prerequisites

- Project installed with `python -m pip install -e ".[dev]"`.
- `sentiment-bench --help` works from the repository root.
- For OpenRouter: `OPENROUTER_API_KEY`.
- For Ollama: the official `ollama` Python package is installed through project dependencies, and the Ollama server is reachable from this machine.

## Provider Defaults

| Setting | Default | Used by |
| --- | --- | --- |
| `SENTIMENT_BENCH_PROVIDER` | `openrouter` | CLI and TUI provider selection. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter-compatible chat endpoint. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama Python client host. |
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

## Configure Ollama On A Desktop PC

The repository can run on your Mac while the model runs on a desktop PC. The important requirement is that the desktop's Ollama server listens on a network address reachable from the Mac.

On the desktop PC:

```bash
ollama pull gemma3
```

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

On the Mac running this repository, add:

```bash
SENTIMENT_BENCH_PROVIDER=ollama
OLLAMA_HOST=http://desktop-pc:11434
```

If the hostname is not resolvable, use the desktop PC's private IP address:

```bash
OLLAMA_HOST=http://192.168.1.25:11434
```

Check the model list from the Mac:

```bash
sentiment-bench list-models --provider ollama \
  --ollama-host http://desktop-pc:11434
```

Run a pilot with the remote Gemma model:

```bash
sentiment-bench run --provider ollama \
  --ollama-host http://desktop-pc:11434 \
  --models gemma3 --mode pilot
```

## Use The TUI Provider Controls

Open:

```bash
sentiment-bench tui
```

In the Dashboard tab:

1. Set Provider to `OpenRouter` or `Ollama`.
2. Set the endpoint field. For Ollama, use `http://desktop-pc:11434`.
3. Open the Models tab and fetch models.
4. Select one or more models.
5. Continue through Prompt and Run.

![TUI provider controls screenshot](assets/tui-dashboard.svg)

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
| Ollama connection refused | Ollama is not running or only bound to localhost on the desktop. | Start Ollama on the desktop and make `OLLAMA_HOST` reachable from the Mac. |
| Model not found | The model was not pulled on the Ollama host. | Run `ollama pull gemma3` on the desktop PC. |
| Very slow local runs | Desktop model inference is slower than API routing or the network is saturated. | Use pilot mode first, keep `--concurrency 1`, and increase only after checking stability. |
| Different results across machines | Local model version, prompt settings, or sampling settings differ. | Record model tag, provider route, prompt hash, seed, temperature, and export metadata. |
