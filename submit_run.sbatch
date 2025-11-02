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

# Activate project virtual environment
source "${REPO_ROOT}/.venv/bin/activate"

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
export NO_PROXY=127.0.0.1,localhost; export no_proxy=127.0.0.1,localhost
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "[INFO] Launching vLLM server on port ${VLLM_PORT} with model ${VLLM_MODEL_ID}"
python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL_ID}" \
  --served-model-name "${VLLM_SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
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
EOF
echo "[INFO] Wrote .env with UTU_LLM_BASE_URL=${UTU_LLM_BASE_URL}"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen_demo}
DOMAIN=${DOMAIN:-math}
MODE=${MODE:-agent}
DATASET=${DATASET:-DAPO-Math-17k}
DATASET_TRUNCATE=${DATASET_TRUNCATE:-100}
EPOCHS=${EPOCHS:-3}
BATCHSIZE=${BATCHSIZE:-50}
GRPO_N=${GRPO_N:-5}
ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY:-16}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-16384}
TASK_TIMEOUT=${TASK_TIMEOUT:-1800}

echo "[INFO] Starting Training-Free GRPO run"
python -m training_free_grpo.train \
  --mode "${MODE}" \
  --domain "${DOMAIN}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --dataset "${DATASET}" \
  --dataset_truncate "${DATASET_TRUNCATE}" \
  --epochs "${EPOCHS}" \
  --batchsize "${BATCHSIZE}" \
  --grpo_n "${GRPO_N}" \
  --rollout_concurrency "${ROLLOUT_CONCURRENCY}" \
  --rollout_temperature "${ROLLOUT_TEMPERATURE}" \
  --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
  --task_timeout "${TASK_TIMEOUT}"

echo "[INFO] Training run completed"
