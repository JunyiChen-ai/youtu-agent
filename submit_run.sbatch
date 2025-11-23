#!/bin/bash
#SBATCH --job-name=youtu-grpo
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

REPO_ROOT="/data/jehc223/youtu-agent"
cd "$REPO_ROOT"
mkdir -p logs

echo "[INFO] Testing HuggingFace connectivity..."
HF_STATUS=$(curl -s -o /tmp/hf_connectivity.txt -w "%{http_code}" https://huggingface.co)
if [ "${HF_STATUS}" != "200" ]; then
  echo "[ERROR] HuggingFace connectivity test failed with HTTP status ${HF_STATUS}." >&2
  echo "[ERROR] Response body:" >&2
  cat /tmp/hf_connectivity.txt >&2 || true
  exit 1
fi
echo "[INFO] HuggingFace connectivity verified."

# Activate project virtual environment
source "${REPO_ROOT}/.venv/bin/activate"

# Determine mode early for server launch configuration
MODE=${MODE:-agent}
MODE_LOWER=$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')

# LLM server settings
# Choose a free port for vLLM
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

# Ensure proxy variables do not interfere with local requests
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
export HTTP_PROXY=http://squid.auckland.ac.nz:3128
export HTTPS_PROXY=http://squid.auckland.ac.nz:3128
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

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

# Wait for vLLM API to become ready
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

# Configure environment variables for Youtu-Agent
export UTU_LLM_TYPE=chat.completions
export UTU_LLM_MODEL="${VLLM_SERVED_NAME}"
export UTU_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export UTU_LLM_API_KEY="local-token"
export DB_URL="${UTU_DB_URL}"
echo "[INFO] UTU_LLM_BASE_URL=${UTU_LLM_BASE_URL}"

cat > "${REPO_ROOT}/.env" <<EOF
UTU_LLM_TYPE=${UTU_LLM_TYPE}
UTU_LLM_MODEL=${UTU_LLM_MODEL}
UTU_LLM_BASE_URL=${UTU_LLM_BASE_URL}
UTU_LLM_API_KEY=${UTU_LLM_API_KEY}
DB_URL=${UTU_DB_URL}
SERPER_API_KEY=10ea7b9b4992460dfa27ffbc910bf90adfa9a8d4
JINA_API_KEY=jina_088d95811e374a1f8ee2628597af857fcxnCbTVePqA1DrJ2XEkFQfN1gldf
EOF
echo "[INFO] Wrote .env with UTU_LLM_BASE_URL=${UTU_LLM_BASE_URL}"

if [ "${MODE_LOWER}" = "agent" ]; then
  echo "[INFO] Testing Jina search connectivity..."
  JINA_KEY=jina_088d95811e374a1f8ee2628597af857fcxnCbTVePqA1DrJ2XEkFQfN1gldf
  JINA_STATUS=$(curl -s -o /tmp/jina_test.json -w "%{http_code}" \
    -H "Accept: application/json" \
    -H "Authorization: Bearer ${JINA_KEY}" \
    -H "X-Respond-With: no-content" \
    "https://s.jina.ai/?q=test")
  if [ "${JINA_STATUS}" != "200" ]; then
    echo "[ERROR] Jina connectivity test failed with HTTP status ${JINA_STATUS}." >&2
    echo "[ERROR] Response body:" >&2
    cat /tmp/jina_test.json >&2 || true
    exit 1
  fi
  echo "[INFO] Jina connectivity verified."
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
DOMAIN=${DOMAIN:-math}
MODE=${MODE:-agent}
DATASET=${DATASET:-DAPO-Math-17k}
DATASET_ORIG="${DATASET}"
DATASET_TRUNCATE=${DATASET_TRUNCATE:-}
EPOCHS=${EPOCHS:-3}
BATCHSIZE=${BATCHSIZE:-50}
GRPO_N=${GRPO_N:-5}
ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY:-16}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-16384}
TASK_TIMEOUT=${TASK_TIMEOUT:-1800}

## (moved) Auto-generate experiment name after resolving final DATASET

# Auto-preprocess for diff domain if needed
if [ "${DOMAIN}" = "diff" ]; then
  # N controls per-class (C and non-C) sample count for train; default to 1000 if not provided
  PRE_N=${PRE_N:-1000}
  # Allow shorthand DATASET like "D1" -> resolve to PKL prefix "D1_*" and use *_train
  if [ -z "${DATASET##*_train}" ] || [ -z "${DATASET##*_test}" ]; then
    : # already a full name with split suffix
  else
    if echo "${DATASET}" | grep -q "_"; then
      : # looks like a full base already, keep as-is
    else
      mkdir -p data/diff_awareness/benchmark_suite
      # try resolve by prefix match in PKL directory
      CANDIDATES=(data/diff_awareness/benchmark_suite/${DATASET}_*.pkl)
      if [ ! -e "${CANDIDATES[0]}" ]; then
        echo "[ERROR] Could not resolve dataset base for '${DATASET}'. Expected PKL like ${DATASET}_*.pkl in data/diff_awareness/benchmark_suite" >&2
        exit 1
      fi
      # pick the first match after sorting by name
      RESOLVED_PKL=$(ls -1 ${CANDIDATES[@]} | sort | head -n1)
      DATASET_BASE=$(basename "${RESOLVED_PKL}" .pkl)   # e.g., D1_1k
      DATASET="${DATASET_BASE}_train"                    # temporary, will remap to base_n_train
      echo "[INFO] Resolved PKL base=$(basename "${RESOLVED_PKL}") -> temp DATASET=${DATASET}"
    fi
  fi

  # Expect DATASET like D1_1k_train or D1_1k_test
  DATASET_BASE="${DATASET%_*}"
  SPLIT_SUFFIX="${DATASET##*_}"
  OUT_DIR="data/diff_awareness/dataset"
  # Map PKL base like D1_1k -> output base D1, and always use PRE_N naming (e.g., D1_100_train/test)
  OUTPUT_BASE="${DATASET_BASE%%_*}"
  RUNTIME_DATASET="${OUTPUT_BASE}_${PRE_N}_${SPLIT_SUFFIX}"
  TRAIN_JSON="${OUT_DIR}/${OUTPUT_BASE}_${PRE_N}_train.json"
  TEST_JSON="${OUT_DIR}/${OUTPUT_BASE}_${PRE_N}_test.json"
  
  # Ensure PKL symlinks exist (non-fatal if already present)
  mkdir -p data/diff_awareness/benchmark_suite "${OUT_DIR}"
  for f in /data/jehc223/difference_awareness/benchmark_suite/*.pkl; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    tgt="data/diff_awareness/benchmark_suite/$bn"
    if [ ! -L "$tgt" ] && [ ! -f "$tgt" ]; then
      ln -s "$f" "$tgt" || true
    fi
  done

  if [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${TEST_JSON}" ]; then
    echo "[INFO] Preprocessing ${DATASET_BASE} -> ${OUTPUT_BASE}_${PRE_N}_{train,test}.json ..."
    python scripts/diff_awareness/preprocess.py \
      --dataset_name "${DATASET_BASE}" \
      --n "${PRE_N}" \
      --source_dir "data/diff_awareness/benchmark_suite" \
      --out_dir "${OUT_DIR}"
  else
    echo "[INFO] Found existing ${TRAIN_JSON} and ${TEST_JSON}, skipping preprocessing."
  fi

  # Override DATASET to the runtime dataset name (either original base or PRE_N-based)
  DATASET="${RUNTIME_DATASET}"
  echo "[INFO] Using runtime DATASET=${DATASET}"
fi

# Auto-generate experiment name if not provided, using final DATASET (e.g., diff_d1_100_prompt)
if [ -z "${EXPERIMENT_NAME}" ]; then
  DATASET_BASE_GUESS="${DATASET%_*}"  # strip trailing _train/_test when present
  DOMAIN_L=$(echo "${DOMAIN}" | tr '[:upper:]' '[:lower:]')
  DATASET_L=$(echo "${DATASET_BASE_GUESS}" | tr '[:upper:]' '[:lower:]')
  MODE_L=$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')
  EXPERIMENT_NAME="${DOMAIN_L}_${DATASET_L}_${MODE_L}"
  echo "[INFO] Auto EXPERIMENT_NAME=${EXPERIMENT_NAME}"
fi

echo "[INFO] Starting Training-Free GRPO run"

# Build optional args
EXTRA_ARGS=""
if [ -n "${DATASET_TRUNCATE}" ]; then
  EXTRA_ARGS+=" --dataset_truncate \"${DATASET_TRUNCATE}\""
fi

python -m training_free_grpo.train \
  --mode "${MODE}" \
  --domain "${DOMAIN}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --batchsize "${BATCHSIZE}" \
  --grpo_n "${GRPO_N}" \
  --rollout_concurrency "${ROLLOUT_CONCURRENCY}" \
  --rollout_temperature "${ROLLOUT_TEMPERATURE}" \
  --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
  --task_timeout "${TASK_TIMEOUT}" \
  ${EXTRA_ARGS}

echo "[INFO] Training run completed"
