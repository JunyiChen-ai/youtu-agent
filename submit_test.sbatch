#!/bin/bash
#SBATCH --job-name=youtu-eval
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -euo pipefail

REPO_ROOT="/data/jehc223/youtu-agent"
cd "$REPO_ROOT"
mkdir -p logs

# Activate venv
source "${REPO_ROOT}/.venv/bin/activate"

# Resolve mode early for server launch behaviour
MODE=${MODE:-prompt}
MODE_LOWER=$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')

# vLLM server
if command -v python >/dev/null 2>&1; then
  VLLM_PORT=$(python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('127.0.0.1', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
PY
  )
else
  VLLM_PORT=${VLLM_PORT:-8000}
fi

VLLM_MODEL_ID=${VLLM_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
VLLM_SERVED_NAME=${VLLM_SERVED_NAME:-qwen2.5-7b}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.7}
UTU_DB_URL=${UTU_DB_URL:-sqlite:///utu.db}

export NO_PROXY=127.0.0.1,localhost; export no_proxy=127.0.0.1,localhost

VLLM_TOOL_FLAGS=()
if [ "${MODE_LOWER}" != "prompt" ]; then
  VLLM_TOOL_FLAGS+=(--enable-auto-tool-choice --tool-call-parser hermes)
fi

echo "[INFO] Launching vLLM server on port ${VLLM_PORT} with model ${VLLM_MODEL_ID}"
python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL_ID}" \
  --served-model-name "${VLLM_SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  "${VLLM_TOOL_FLAGS[@]}" \
  > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[INFO] Stopping vLLM server (PID ${VLLM_PID})"
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[INFO] Waiting for vLLM server to become ready..."
READY=false
for _ in $(seq 1 30); do
  if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] vLLM server is ready."
    READY=true
    break
  fi
  sleep 5
done

if [ "${READY}" != "true" ]; then
  echo "[ERROR] vLLM server did not start in time."
  exit 1
fi

# Configure UTU
export UTU_LLM_TYPE=chat.completions
export UTU_LLM_MODEL="${VLLM_SERVED_NAME}"
export UTU_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export UTU_LLM_API_KEY="local-token"
export DB_URL="${UTU_DB_URL}"

cat > "${REPO_ROOT}/.env" <<EOF
UTU_LLM_TYPE=${UTU_LLM_TYPE}
UTU_LLM_MODEL=${UTU_LLM_MODEL}
UTU_LLM_BASE_URL=${UTU_LLM_BASE_URL}
UTU_LLM_API_KEY=${UTU_LLM_API_KEY}
DB_URL=${UTU_DB_URL}
EOF

# Params
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
DOMAIN=${DOMAIN:-diff}
MODE=${MODE:-prompt}
DATASET=${DATASET:-D1}
PRE_N=${PRE_N:-1000}
PASS_K=${PASS_K:-1}
ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY:-16}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-16384}
TASK_TIMEOUT=${TASK_TIMEOUT:-1800}

# Resolve dataset (diff domain supports shorthand and ensures test split)
if [ "${DOMAIN}" = "diff" ]; then
  mkdir -p data/diff_awareness/benchmark_suite data/diff_awareness/dataset
  # Ensure PKL symlinks
  for f in /data/jehc223/difference_awareness/benchmark_suite/*.pkl; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    tgt="data/diff_awareness/benchmark_suite/$bn"
    if [ ! -L "$tgt" ] && [ ! -f "$tgt" ]; then
      ln -s "$f" "$tgt" || true
    fi
  done

  if [ -z "${DATASET##*_train}" ] || [ -z "${DATASET##*_test}" ]; then
    # already full name; keep
    DATASET_BASE="${DATASET%_*}"
    SPLIT_SUFFIX="${DATASET##*_}"
  else
    # shorthand e.g. D1 -> resolve PKL base (e.g., D1_1k), then map to D1_{PRE_N}_test
    CANDIDATES=(data/diff_awareness/benchmark_suite/${DATASET}_*.pkl)
    if [ ! -e "${CANDIDATES[0]}" ]; then
      echo "[ERROR] Could not resolve dataset base for '${DATASET}'." >&2
      exit 1
    fi
    RESOLVED_PKL=$(ls -1 ${CANDIDATES[@]} | sort | head -n1)
    DATASET_BASE=$(basename "${RESOLVED_PKL}" .pkl)
    SPLIT_SUFFIX="test"
    echo "[INFO] Resolved PKL base=$(basename "${RESOLVED_PKL}")"
  fi

  OUTPUT_BASE="${DATASET_BASE%%_*}"
  RUNTIME_DATASET="${OUTPUT_BASE}_${PRE_N}_${SPLIT_SUFFIX}"
  TRAIN_JSON="data/diff_awareness/dataset/${OUTPUT_BASE}_${PRE_N}_train.json"
  TEST_JSON="data/diff_awareness/dataset/${OUTPUT_BASE}_${PRE_N}_test.json"

  if [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${TEST_JSON}" ]; then
    echo "[INFO] Preprocessing ${DATASET_BASE} -> ${OUTPUT_BASE}_${PRE_N}_{train,test}.json ..."
    python scripts/diff_awareness/preprocess.py \
      --dataset_name "${DATASET_BASE}" \
      --n "${PRE_N}" \
      --source_dir "data/diff_awareness/benchmark_suite" \
      --out_dir "data/diff_awareness/dataset"
  else
    echo "[INFO] Found ${TRAIN_JSON} and ${TEST_JSON}"
  fi

  DATASET="${RUNTIME_DATASET}"
  echo "[INFO] Using runtime DATASET=${DATASET}"
fi

# Experience file (user can pass only the step dir; we resolve to experiences.json)
EXPERIENCE_STEP_DIR=${EXPERIENCE_STEP_DIR:-}
EXPERIENCE_FILE=${EXPERIENCE_FILE:-}
if [ -z "${EXPERIENCE_FILE}" ] && [ -n "${EXPERIENCE_STEP_DIR}" ]; then
  EXPERIENCE_FILE="${EXPERIENCE_STEP_DIR%/}/experiences.json"
fi
if [ -n "${EXPERIENCE_FILE}" ]; then
  echo "[INFO] Using experiences from ${EXPERIENCE_FILE}"
  EXPERIENCE_SUFFIX=$(basename "${EXPERIENCE_FILE}")
  EXPERIENCE_SUFFIX="${EXPERIENCE_SUFFIX%.*}"
  EXPERIENCE_SUFFIX=${EXPERIENCE_SUFFIX//[^A-Za-z0-9]/_}
  # trim to last two underscore-separated segments to avoid long suffixes
  IFS='_' read -ra EXP_SEG <<<"${EXPERIENCE_SUFFIX}"
  SEG_COUNT=${#EXP_SEG[@]}
  if [ "${SEG_COUNT}" -gt 2 ]; then
    EXPERIENCE_SUFFIX="${EXP_SEG[$((SEG_COUNT-2))]}_${EXP_SEG[$((SEG_COUNT-1))]}"
  fi
  if [ -n "${EXPERIENCE_SUFFIX}" ]; then
    EXPERIENCE_SUFFIX="_${EXPERIENCE_SUFFIX}"
  fi
else
  EXPERIENCE_SUFFIX=""
fi

# Auto experiment name: include final dataset and _eval suffix
if [ -z "${EXPERIMENT_NAME}" ]; then
  DATASET_BASE_GUESS="${DATASET%_*}"
  DOMAIN_L=$(echo "${DOMAIN}" | tr '[:upper:]' '[:lower:]')
  DATASET_L=$(echo "${DATASET_BASE_GUESS}" | tr '[:upper:]' '[:lower:]')
  MODE_L=$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')
  EXPERIMENT_NAME="${DOMAIN_L}_${DATASET_L}_${MODE_L}_eval"
  echo "[INFO] Auto EXPERIMENT_NAME=${EXPERIMENT_NAME}"
fi

if [ -n "${EXPERIENCE_SUFFIX}" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}${EXPERIENCE_SUFFIX}"
  echo "[INFO] Experiment name with experience suffix: ${EXPERIMENT_NAME}"
fi

# Run evaluation rollouts
echo "[INFO] Starting evaluation rollouts"
python -m training_free_grpo.main \
  --mode "${MODE}" \
  --domain "${DOMAIN}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --dataset "${DATASET}" \
  --rollout_concurrency "${ROLLOUT_CONCURRENCY}" \
  --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
  --pass_k "${PASS_K}" \
  ${EXPERIENCE_FILE:+--experience_file "${EXPERIENCE_FILE}"}

# Compute custom metrics for diff domain
if [ "${DOMAIN}" = "diff" ]; then
  ROLLOUT_FILE="data/${DOMAIN}/eval/${EXPERIMENT_NAME}.jsonl"
  if [ -f "${ROLLOUT_FILE}" ]; then
    echo "[INFO] Computing Difference/Context Awareness metrics from ${ROLLOUT_FILE}"
    python scripts/diff_awareness/evaluate_diff_metrics.py --rollout_file "${ROLLOUT_FILE}"
  else
    echo "[WARN] Rollout file not found: ${ROLLOUT_FILE}"
  fi
fi

echo "[INFO] Evaluation completed"
