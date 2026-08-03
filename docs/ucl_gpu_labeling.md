# UCL GPU Labelling

Use the current reserved UCL GPU host through the Mac SSH alias. The hostname may change between reservations; all persistent state stays under the auto-mounted project store:

```text
/cs/student/project_msc/2025/cf/pprender/
```

The bootstrap layout keeps repositories, environments, licensed input data, run outputs, and model artifacts separate. None of the data, environments, model weights, or run outputs belong in Git. Set `PIP_CACHE_DIR`, `HF_HOME`, and other large caches under this project root; the 10 GB home volume is not a model or package cache.

## Connect and check the GPU

To scan every public lab GPU from the Mac without password prompts:

```bash
python scripts/ucl_gpu_status.py
```

The checker uses the hostnames published on the [UCL CS GPU page](https://tsg.cs.ucl.ac.uk/gpus/), checks them concurrently through the `ucl-knuckles` SSH alias, and reports `FREE`, `TAKEN`, `OFFLINE`, `AUTH`, `HOST_KEY`, or `ERROR`. `FREE` means that `nvidia-smi` returned successfully and listed no compute processes; ordinary driver/display memory by itself does not count as taken. Use `--lab lab105`, `--lab lab121`, `--host canada-l`, or `--json` to narrow or automate the check.

The published page currently describes 25 lab105 and 30 lab121 PCs, but lists 23 and 31 hostnames respectively. The dated inventory in `scripts/ucl_gpu_status.py` preserves all 54 names actually published so stale or offline entries remain visible rather than being silently omitted; it does not guess the two unlisted lab105 names.

The script sets SSH `BatchMode=yes`, so it never asks for or stores a password. Public-key authentication must already work. On macOS, use an SSH configuration like this (substitute your UCL CS username):

```sshconfig
Host ucl-knuckles
    HostName knuckles.cs.ucl.ac.uk
    User <ucl-cs-username>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C
```

Add the key to the macOS keychain once with `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`. If the public key has not yet been registered in the CS account's `~/.ssh/authorized_keys`, that server-side setup still requires one initial authenticated login (or help from TSG); the checker deliberately does not work around UCL authentication.

For a single reserved workstation, continue to use the existing alias:

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
tmux attach -t labels-gemma4_12b-t64
```

Rerunning the launcher resumes from successful rows. To continue from the pilot through the complete frozen population, set an explicitly empty `LIMIT` while retaining the same output path:

```bash
LIMIT= scripts/ucl_label_headlines.sh
```

The default 64-token completion budget leaves enough room for the structured three-probability response. The output filename records the model, prompt, and token budget so a changed request cannot silently append to an earlier score file.

Override `MODEL`, `PROMPT_ID`, `MAX_COMPLETION_TOKENS`, `COLLECTION_ROOT`, `OUTPUT_PATH`, or `CONCURRENCY` explicitly for a different registered experiment. Do not change those settings while reusing an existing score file.

## Expanded 44-company headline labelling

The expanded all-source corpus was completed with the one-time compatibility
launcher `scripts/ucl_run_lseg_baselines.sh` and shared project paths. That
launcher is now frozen behind an explicit legacy opt-in because its resumable
contract also invokes VADER; do not use it for a new collection.

The primary analysis consumes FinBERT only. The paired VADER rows left by the
legacy resumable executor are inert provenance and must not enter tables,
figures, LLM agreement metrics, or return analysis.

For the separate structured-output Gemma gate, an explicitly empty
`SOURCE_CODE` means all sources and `DIRECT_COMPANY_ONLY=0` retains contextual
and multi-company headlines. The first 200 rows are the deterministic
hash-ordered runtime/format pilot. The researcher-supplied wording is registered
as `investor_headline_soft_label_v1` with prompt hash
`81596538d99b29b8`:

```bash
COLLECTION_ROOT=/cs/student/project_msc/2025/cf/pprender/data/raw/lseg_us_sector_44_8m_headlines \
RUN_ROOT=/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/llm \
OUTPUT_PATH=/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/llm/headline_scores_gemma4_12b_investor_headline_soft_label_v1_t64.csv \
LABEL_TMUX_SESSION=lseg44-gemma4-pilot \
PROMPT_ID=investor_headline_soft_label_v1 \
SOURCE_CODE= \
DIRECT_COMPANY_ONLY=0 \
MAX_POPULATION=888155 \
LIMIT=200 \
scripts/ucl_label_headlines.sh
```

Do not present LLM–FinBERT agreement as accuracy. Expansion beyond the 200-row
gate follows the frozen sampling and human-audit design in
`final_experiments/12_lseg_44_labelling.ipynb`.

## Git access

The Mac forwards its SSH agent for the current UCL alias, so the remote clone can pull and push without storing a GitHub private key on UCL:

```bash
git pull --ff-only origin main
git push origin main
```

Agent-backed Git access is available only while the SSH connection from the Mac is active.
