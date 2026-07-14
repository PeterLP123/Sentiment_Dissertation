#!/usr/bin/env bash
set -euo pipefail

project_root="${UCL_PROJECT_ROOT:-/cs/student/project_msc/2025/cf/pprender}"
repo_root="${UCL_REPO_ROOT:-${project_root}/repos/Sentiment_Dissertation}"
collection_root="${COLLECTION_ROOT:-${project_root}/data/raw/lseg_us_midcap_22_1y}"
run_root="${RUN_ROOT:-${project_root}/runs/labels/lseg_us_midcap_22_1y}"
model="${MODEL:-gemma4:12b}"
prompt_id="${PROMPT_ID:-financial_soft_label_base}"
limit="${LIMIT:-100}"
max_population="${MAX_POPULATION:-2282}"
concurrency="${CONCURRENCY:-1}"
api_url="${OLLAMA_API_URL:-http://127.0.0.1:11434}"
sentiment_bench="${project_root}/envs/sentiment/bin/sentiment-bench"
safe_model="${model//[\/:]/_}"
output_path="${OUTPUT_PATH:-${run_root}/headline_scores_${safe_model}.csv}"
session_name="${LABEL_TMUX_SESSION:-labels-${safe_model}}"

if [[ "${UCL_LABEL_INNER:-0}" != "1" ]]; then
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session ${session_name} already exists; attach with: tmux attach -t ${session_name}" >&2
    exit 1
  fi
  tmux new-session -d -s "${session_name}" bash -lc \
    "cd '${repo_root}' && exec env UCL_LABEL_INNER=1 UCL_PROJECT_ROOT='${project_root}' COLLECTION_ROOT='${collection_root}' RUN_ROOT='${run_root}' MODEL='${model}' PROMPT_ID='${prompt_id}' LIMIT='${limit}' MAX_POPULATION='${max_population}' CONCURRENCY='${concurrency}' OUTPUT_PATH='${output_path}' '${repo_root}/scripts/ucl_label_headlines.sh'"
  echo "Labelling started in tmux session ${session_name}."
  echo "Attach with: tmux attach -t ${session_name}"
  echo "Checkpointed output: ${output_path}"
  exit 0
fi

cd "${repo_root}"
mkdir -p "${run_root}" "${project_root}/artifacts/cache"
export OLLAMA_HOST="${api_url}"
export OLLAMA_MODELS="${project_root}/artifacts/models/ollama"
export XDG_CACHE_HOME="${project_root}/artifacts/cache/xdg"
export SENTIMENT_BENCH_MACHINE_LABEL="ucl-$(hostname -s)"

"${repo_root}/scripts/ucl_ollama_server.sh"

if ! "${project_root}/envs/ollama/bin/ollama" list | awk 'NR > 1 {print $1}' | grep -qxF "${model}"; then
  echo "Model ${model} is not installed in ${OLLAMA_MODELS}." >&2
  echo "Pull it first with: ${project_root}/envs/ollama/bin/ollama pull ${model}" >&2
  exit 1
fi

common_args=(
  --model "${model}"
  --provider ollama
  --ollama-host "${api_url}"
  --collection-root "${collection_root}"
  --prompt-id "${prompt_id}"
  --output "${output_path}"
  --temperature 0
  --max-completion-tokens 32
  --concurrency "${concurrency}"
  --source-code NS:RTRS
  --direct-company-only
  --max-population "${max_population}"
  --no-ollama-think
  --ollama-keep-alive -1
  --structured-output
)

"${sentiment_bench}" score-headlines "${common_args[@]}" --dry-run

run_args=("${common_args[@]}")
if [[ -n "${limit}" ]]; then
  run_args+=(--limit "${limit}")
fi
"${sentiment_bench}" score-headlines "${run_args[@]}"
