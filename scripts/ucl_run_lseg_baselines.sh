#!/usr/bin/env bash
set -euo pipefail

project_root="${UCL_PROJECT_ROOT:-/cs/student/project_msc/2025/cf/pprender}"
repo_root="${UCL_REPO_ROOT:-${project_root}/repos/Sentiment_Dissertation}"
collection_root="${COLLECTION_ROOT:-${project_root}/data/raw/lseg_us_sector_44_8m_headlines}"
run_root="${RUN_ROOT:-${project_root}/runs/labels/lseg_us_sector_44_8m_headlines}"
output_path="${OUTPUT_PATH:-${run_root}/headline_scores_finbert4556_vader_baseline_v1_cacheonly_20260803.csv}"
log_path="${LABEL_LOG_PATH:-${project_root}/artifacts/logs/lseg44-finbert-20260803.log}"
record_path="${RUN_RECORD_PATH:-${run_root}/ucl_run_record_20260803.yaml}"
session_name="${LABEL_TMUX_SESSION:-lseg44-finbert-20260803}"
python_bin="${PYTHON_BIN:-${project_root}/envs/sentiment/bin/python3}"
finbert_revision="${FINBERT_REVISION:-4556d13015211d73dccd3fdd39d39232506f3e43}"
finbert_batch_size="${FINBERT_BATCH_SIZE:-64}"

if [[ "${ALLOW_LEGACY_VADER_EXECUTOR:-0}" != "1" ]]; then
  echo "This one-time compatibility executor is frozen because it also runs VADER." >&2
  echo "VADER was excluded from new work on 2026-08-03; use the completed artifact and FinBERT-only export." >&2
  exit 1
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${project_root}/artifacts/cache/huggingface"
export NLTK_DATA="${project_root}/nltk_data"
export PIP_CACHE_DIR="${project_root}/artifacts/cache/pip"
export XDG_CACHE_HOME="${project_root}/artifacts/cache/xdg"
export SENTIMENT_BENCH_MACHINE_LABEL="ucl-$(hostname -s)"

if [[ "${UCL_BASELINE_INNER:-0}" != "1" ]]; then
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session ${session_name} already exists; attach with: tmux attach -t ${session_name}" >&2
    exit 1
  fi
  compute_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${compute_processes}" ]]; then
    echo "Refusing to launch because the GPU has compute processes:" >&2
    echo "${compute_processes}" >&2
    exit 1
  fi
  mkdir -p "${run_root}" "$(dirname "${log_path}")" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  commit="$(git -C "${repo_root}" rev-parse HEAD)"
  cat >"${record_path}" <<EOF
run_id: lseg44-finbert-20260803
started_at_utc: ${started_at}
host: $(hostname -s)
gpu: ${gpu}
gpu_check_at_utc: ${started_at}
project_root: ${project_root}
repository: ${repo_root}
commit: ${commit}
environment: ${project_root}/envs/sentiment
python: 3.12
packages: torch=2.12.1+cu130 transformers=5.12.1 nltk=3.10.0
model: ProsusAI/finbert@${finbert_revision} plus VADER
dataset_version: headlines_sha256=7f48c46690f3293cbfe846ddecc701852daf3c35643f2747dcbc39f829927878 population=888155 inherited=568704 missing=319451
prompt_or_config: headline text only; score=P(positive)-P(negative)
parameters: finbert_batch_size=${finbert_batch_size} finbert_checkpoint_size=256 local_files_only=true
seed: null
tmux_session: ${session_name}
command: scripts/ucl_run_lseg_baselines.sh
output_path: ${output_path}
checkpoint_path: ${output_path}.manifest.json
log_path: ${log_path}
resume_from: 568704 inherited exact-hash FinBERT and VADER pairs
notes: licensed text remains on UCL project storage and outside Git
EOF
  tmux new-session -d -s "${session_name}" bash -lc \
    "cd '${repo_root}' && exec env UCL_BASELINE_INNER=1 UCL_PROJECT_ROOT='${project_root}' UCL_REPO_ROOT='${repo_root}' COLLECTION_ROOT='${collection_root}' RUN_ROOT='${run_root}' OUTPUT_PATH='${output_path}' LABEL_LOG_PATH='${log_path}' RUN_RECORD_PATH='${record_path}' LABEL_TMUX_SESSION='${session_name}' PYTHON_BIN='${python_bin}' FINBERT_REVISION='${finbert_revision}' FINBERT_BATCH_SIZE='${finbert_batch_size}' '${repo_root}/scripts/ucl_run_lseg_baselines.sh' >>'${log_path}' 2>&1"
  echo "Labelling started in tmux session ${session_name}."
  echo "Checkpointed output: ${output_path}"
  echo "Persistent log: ${log_path}"
  echo "Run record: ${record_path}"
  exit 0
fi

cd "${repo_root}"
exec "${python_bin}" scripts/run_lseg_headline_study.py score-baselines \
  --collection-root "${collection_root}" \
  --output "${output_path}" \
  --finbert-batch-size "${finbert_batch_size}" \
  --finbert-revision "${finbert_revision}"
