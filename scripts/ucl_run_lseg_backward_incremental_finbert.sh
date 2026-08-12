#!/usr/bin/env bash
set -euo pipefail

project_root="${UCL_PROJECT_ROOT:-/cs/student/project_msc/2025/cf/pprender}"
repo_root="${UCL_REPO_ROOT:-${project_root}/repos/Sentiment_Dissertation}"
snapshot_id="${SNAPSHOT_ID:-snapshot_20260808_quota_stop}"
collection_root="${COLLECTION_ROOT:-${project_root}/data/raw/lseg_us_sector_33_backward_incremental/${snapshot_id}}"
run_root="${RUN_ROOT:-${project_root}/runs/labels/lseg_us_sector_33_backward_20240101_20251026/incremental/${snapshot_id}}"
output_path="${OUTPUT_PATH:-${run_root}/headline_scores_finbert4556_vader_baseline_v1_cacheonly.csv}"
seed_output_path="${SEED_OUTPUT_PATH:-}"
allow_resume="${ALLOW_RESUME:-0}"
log_path="${LABEL_LOG_PATH:-${project_root}/artifacts/logs/lseg33-backward-finbert-${snapshot_id}.log}"
record_path="${RUN_RECORD_PATH:-${run_root}/ucl_run_record.yaml}"
session_name="${LABEL_TMUX_SESSION:-lseg33-backward-finbert-20260808}"
python_bin="${PYTHON_BIN:-${project_root}/envs/sentiment/bin/python3}"
finbert_revision="${FINBERT_REVISION:-4556d13015211d73dccd3fdd39d39232506f3e43}"
finbert_batch_size="${FINBERT_BATCH_SIZE:-64}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${project_root}/artifacts/cache/huggingface"
export NLTK_DATA="${project_root}/nltk_data"
export PIP_CACHE_DIR="${project_root}/artifacts/cache/pip"
export XDG_CACHE_HOME="${project_root}/artifacts/cache/xdg"
export SENTIMENT_BENCH_MACHINE_LABEL="ucl-$(hostname -s)"

if [[ "${UCL_INCREMENTAL_FINBERT_INNER:-0}" != "1" ]]; then
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session ${session_name} already exists" >&2
    exit 1
  fi
  compute_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${compute_processes}" ]]; then
    echo "Refusing to launch because the GPU has compute processes:" >&2
    echo "${compute_processes}" >&2
    exit 1
  fi
  [[ -x "${python_bin}" ]] || { echo "Python environment is missing: ${python_bin}" >&2; exit 1; }
  [[ -f "${collection_root}/headlines.jsonl" ]] || { echo "Snapshot headlines are missing" >&2; exit 1; }
  [[ -f "${collection_root}/manifest.json" ]] || { echo "Snapshot manifest is missing" >&2; exit 1; }
  resume_existing_output="false"
  if [[ -e "${output_path}" || -e "${output_path}.manifest.json" ]]; then
    [[ "${allow_resume}" == "1" ]] || {
      echo "Refusing to overwrite an existing score artifact without ALLOW_RESUME=1" >&2
      exit 1
    }
    [[ -f "${output_path}" && -f "${output_path}.manifest.json" ]] || {
      echo "Refusing an incomplete resume checkpoint pair" >&2
      exit 1
    }
    [[ -z "${seed_output_path}" ]] || {
      echo "SEED_OUTPUT_PATH must be empty when resuming the existing output" >&2
      exit 1
    }
    resume_existing_output="true"
  fi

  mkdir -p "${run_root}" "$(dirname "${log_path}")" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
  seed_output_sha256="null"
  if [[ -n "${seed_output_path}" ]]; then
    [[ -f "${seed_output_path}" ]] || { echo "FinBERT seed output is missing: ${seed_output_path}" >&2; exit 1; }
    cp -- "${seed_output_path}" "${output_path}"
    seed_output_sha256="$(sha256sum "${seed_output_path}" | awk '{print $1}')"
  fi
  starting_output_sha256="null"
  starting_manifest_sha256="null"
  if [[ "${resume_existing_output}" == "true" ]]; then
    starting_output_sha256="$(sha256sum "${output_path}" | awk '{print $1}')"
    starting_manifest_sha256="$(sha256sum "${output_path}.manifest.json" | awk '{print $1}')"
  fi
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  commit="$(git -C "${repo_root}" rev-parse HEAD)"
  dirty="false"
  [[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || dirty="true"
  headlines_sha256="$(sha256sum "${collection_root}/headlines.jsonl" | awk '{print $1}')"
  manifest_sha256="$(sha256sum "${collection_root}/manifest.json" | awk '{print $1}')"
  population="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"]["unique_scorable_headlines"])' "${collection_root}/manifest.json")"
  population_sha256="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"]["population_sha256"])' "${collection_root}/manifest.json")"

  cat >"${record_path}" <<EOF
run_id: ${session_name}
started_at_utc: ${started_at}
host: $(hostname -s)
gpu: ${gpu}
gpu_check_at_utc: ${started_at}
project_root: ${project_root}
repository: ${repo_root}
commit: ${commit}
repository_dirty: ${dirty}
environment: ${project_root}/envs/sentiment
model: ProsusAI/finbert@${finbert_revision} plus non-analytical VADER companion rows
dataset_snapshot: ${snapshot_id}
seed_score_artifact: ${seed_output_path:-null}
seed_score_artifact_sha256: ${seed_output_sha256}
resume_existing_output: ${resume_existing_output}
starting_output_sha256: ${starting_output_sha256}
starting_manifest_sha256: ${starting_manifest_sha256}
dataset_headlines_sha256: ${headlines_sha256}
dataset_manifest_sha256: ${manifest_sha256}
population_sha256: ${population_sha256}
population_unique_scorable_headlines: ${population}
score: P(positive)-P(negative)
parameters: finbert_batch_size=${finbert_batch_size} finbert_checkpoint_size=256 local_files_only=true
seed: null
tmux_session: ${session_name}
output_path: ${output_path}
checkpoint_path: ${output_path}.manifest.json
log_path: ${log_path}
notes: terminal-company-date immutable snapshot; licensed text remains on UCL shared storage; no prices or returns loaded
EOF

  tmux new-session -d -s "${session_name}" bash -lc \
    "cd '${repo_root}' && exec env UCL_INCREMENTAL_FINBERT_INNER=1 UCL_PROJECT_ROOT='${project_root}' UCL_REPO_ROOT='${repo_root}' SNAPSHOT_ID='${snapshot_id}' COLLECTION_ROOT='${collection_root}' RUN_ROOT='${run_root}' OUTPUT_PATH='${output_path}' LABEL_LOG_PATH='${log_path}' RUN_RECORD_PATH='${record_path}' LABEL_TMUX_SESSION='${session_name}' PYTHON_BIN='${python_bin}' FINBERT_REVISION='${finbert_revision}' FINBERT_BATCH_SIZE='${finbert_batch_size}' '${run_root}/ucl_run_lseg_backward_incremental_finbert.sh' >>'${log_path}' 2>&1"
  echo "FinBERT scoring started in tmux session ${session_name}."
  echo "Output: ${output_path}"
  echo "Log: ${log_path}"
  echo "Run record: ${record_path}"
  exit 0
fi

cd "${repo_root}"
exec "${python_bin}" scripts/run_lseg_headline_study.py score-baselines \
  --collection-root "${collection_root}" \
  --output "${output_path}" \
  --finbert-batch-size "${finbert_batch_size}" \
  --finbert-revision "${finbert_revision}"
