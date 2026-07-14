#!/usr/bin/env bash
set -euo pipefail

project_root="${UCL_PROJECT_ROOT:-/cs/student/project_msc/2025/cf/pprender}"
ollama_bin="${OLLAMA_BIN:-${project_root}/envs/ollama/bin/ollama}"
models_dir="${OLLAMA_MODELS:-${project_root}/artifacts/models/ollama}"
ollama_home="${OLLAMA_HOME:-${project_root}/artifacts/ollama-home}"
log_dir="${project_root}/artifacts/logs"
session_name="${OLLAMA_TMUX_SESSION:-ollama-ucl}"
api_url="${OLLAMA_API_URL:-http://127.0.0.1:11434}"

mkdir -p "${models_dir}" "${ollama_home}" "${log_dir}"

if [[ ! -x "${ollama_bin}" ]]; then
  echo "Missing Ollama executable: ${ollama_bin}" >&2
  exit 1
fi

if curl --fail --silent --max-time 2 "${api_url}/api/version" >/dev/null; then
  echo "Ollama is already available at ${api_url}."
  exit 0
fi

compute_processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${compute_processes}" ]]; then
  echo "Refusing to start Ollama because the GPU already has compute processes:" >&2
  echo "${compute_processes}" >&2
  exit 1
fi

if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session ${session_name} exists but ${api_url} is unavailable." >&2
  echo "Inspect it with: tmux attach -t ${session_name}" >&2
  exit 1
fi

log_path="${log_dir}/ollama-$(hostname -s).log"
tmux new-session -d -s "${session_name}" bash -lc \
  "exec env HOME='${ollama_home}' OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS='${models_dir}' OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 '${ollama_bin}' serve >>'${log_path}' 2>&1"

for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 2 "${api_url}/api/version" >/dev/null; then
    echo "Ollama started in tmux session ${session_name}; log: ${log_path}"
    exit 0
  fi
  sleep 1
done

echo "Ollama did not become ready; inspect ${log_path}." >&2
exit 1
