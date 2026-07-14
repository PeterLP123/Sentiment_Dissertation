# UCL GPU Labelling

Use the current reserved UCL GPU host through the Mac SSH alias. The hostname may change between reservations; all persistent state stays under the auto-mounted project store:

```text
/cs/student/project_msc/2025/cf/pprender/
```

The bootstrap layout keeps repositories, environments, licensed input data, run outputs, and model artifacts separate. None of the data, environments, model weights, or run outputs belong in Git.

## Connect and check the GPU

```bash
ssh rachet
bash
nvidia-smi
```

Start work only when `nvidia-smi` shows no existing compute process. Long-running services and labelling commands use `tmux`.

## Persistent paths

```text
repos/Sentiment_Dissertation/   Git clone
envs/sentiment/                 Python 3.12 environment
envs/ollama/                    User-space Ollama runtime
data/raw/                       Licensed immutable inputs
runs/labels/                    Append-only score checkpoints
artifacts/models/ollama/        Ignored model blobs
artifacts/ollama-home/          Ollama identity and user state
artifacts/logs/                 Runtime logs
```

The Ollama service listens only on `127.0.0.1:11434`. Start or check it with:

```bash
cd /cs/student/project_msc/2025/cf/pprender/repos/Sentiment_Dissertation
scripts/ucl_ollama_server.sh
tmux attach -t ollama-ucl
```

## Run the frozen 100-headline pilot

The default launcher scores the same hash-ordered population used by the mid-cap headline study, writes an append-only CSV, and records a manifest beside it:

```bash
cd /cs/student/project_msc/2025/cf/pprender/repos/Sentiment_Dissertation
scripts/ucl_label_headlines.sh
tmux attach -t labels-gemma4_12b
```

Rerunning the launcher resumes from successful rows. To continue from the pilot through the complete frozen population, use an empty `LIMIT` while retaining the same output path:

```bash
LIMIT= scripts/ucl_label_headlines.sh
```

Override `MODEL`, `PROMPT_ID`, `COLLECTION_ROOT`, `OUTPUT_PATH`, or `CONCURRENCY` explicitly for a different registered experiment. Do not change those settings while reusing an existing score file.

## Git access

The Mac forwards its SSH agent for the current UCL alias, so the remote clone can pull and push without storing a GitHub private key on UCL:

```bash
git pull --ff-only origin main
git push origin main
```

Agent-backed Git access is available only while the SSH connection from the Mac is active.
