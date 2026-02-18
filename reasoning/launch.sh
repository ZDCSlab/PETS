#!/usr/bin/env bash
set -euo pipefail

export VLLM_CLIENT_TIMEOUT="${VLLM_CLIENT_TIMEOUT:-14400}"
export VLLM_CLIENT_STREAM="${VLLM_CLIENT_STREAM:-0}"
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ──────────────────────────────────────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
Usage: ./launch.sh --model-dir PATH --task TASK [options]

Complete benchmark launcher that starts vLLM server with confidence plugin and runs evaluation.

Required Arguments:
  --model-dir PATH           Path to the model to serve
  --task TASK                Task to run: aime24, aime25, hmmt, brumo, gpqa, mmlu_pro

Optional Arguments:
  --n N                      Number of samples per question (default: 64)
  --tp-size N                Tensor parallel size (default: 1)
  --dp-size N                Data parallel size (default: 1)
  --api-server-count N       API server count (default: 2 * dp-size)
  --enable-expert-parallel   Enable expert parallel, recommended for MoE models
                             (flag alone => 1, or pass 0/1 explicitly; default: 0)
  --temp N                   Sampling temperature (default: 1.1)
  --top-p N                  Sampling top-p (default: 0.95)
  --top-logprobs N           Top-k logprobs for confidence (default: 20)
  --port N                   API server port (default: 8000)
  --out PATH                 Output file path (default: results/<task>_<model>_<n>.jsonl)
  --data-path PATH           Dataset path override (default comes from selected task)
                             Local path can be absolute/relative and can be outside PETS/
  --max-workers N            Max concurrent requests (default: 16)
  --gpu-memory-utilization N GPU memory utilization (default: 0.97)
  --server-only              Only start the server, do not run benchmark
  -h, --help                 Show this help message

Environment Variables:
  VLLM_CONF_MODE             Confidence mode: stats | per_token | summary | empty (default: stats)
  VLLM_CONF_TOPK             Top-k for confidence calculation (default: 20)
  SERVER_READY_TIMEOUT       Timeout waiting for server (default: 7200 seconds)

Examples:
  # Run AIME 2024 with 64 samples on 8 GPUs (TP=1, DP=8)
  ./launch.sh --model-dir /path/to/model --task aime24 --n 64

  # Run with tensor parallel=2, data parallel=4
  ./launch.sh --model-dir /path/to/model --task aime24 --tp-size 2 --dp-size 4

  # Run MMLU-Pro with custom data path
  ./launch.sh --model-dir /path/to/model --task mmlu_pro --data-path TIGER-Lab/MMLU-Pro

  # Run AIME24 with local parquet folder
  ./launch.sh --model-dir /path/to/model --task aime24 --data-path /path/to/aime24_dir

  # Run AIME25 with local jsonl file
  ./launch.sh --model-dir /path/to/model --task aime25 --data-path /path/to/aime25.jsonl
EOF
}

# ──────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR_VALUE=""
TASK_VALUE=""
N_VALUE=""
TP_SIZE_VALUE=""
DP_SIZE_VALUE=""
ENABLE_EXPERT_PARALLEL_VALUE="0"
API_SERVER_COUNT_VALUE=""
TEMP_VALUE="${TEMPERATURE:-1.1}"
TOP_P_VALUE="${TOP_P:-0.95}"
TOP_LOGPROBS_VALUE="${TOP_LOGPROBS:-20}"
API_PORT="${API_PORT:-8000}"
OUTPUT_FILE=""
DATA_PATH=""
MAX_WORKERS="${MAX_WORKERS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
SERVER_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      shift || { echo "[ERROR] --model-dir requires a value"; usage; exit 1; }
      MODEL_DIR_VALUE="$1"
      ;;
    --task)
      shift || { echo "[ERROR] --task requires a value"; usage; exit 1; }
      TASK_VALUE="$1"
      ;;
    --n)
      shift || { echo "[ERROR] --n requires a value"; usage; exit 1; }
      N_VALUE="$1"
      ;;
    --tp-size)
      shift || { echo "[ERROR] --tp-size requires a value"; usage; exit 1; }
      TP_SIZE_VALUE="$1"
      ;;
    --dp-size)
      shift || { echo "[ERROR] --dp-size requires a value"; usage; exit 1; }
      DP_SIZE_VALUE="$1"
      ;;
    --api-server-count)
      shift || { echo "[ERROR] --api-server-count requires a value"; usage; exit 1; }
      API_SERVER_COUNT_VALUE="$1"
      ;;
    --enable-expert-parallel)
      # Supports both:
      #   --enable-expert-parallel        (sets to 1)
      #   --enable-expert-parallel 0|1    (explicit value)
      if [[ $# -ge 2 ]]; then
        if [[ "$2" =~ ^[01]$ ]]; then
          ENABLE_EXPERT_PARALLEL_VALUE="$2"
          shift
        elif [[ "$2" == --* ]]; then
          ENABLE_EXPERT_PARALLEL_VALUE="1"
        else
          echo "[ERROR] --enable-expert-parallel value must be 0 or 1 (got '$2')"
          usage
          exit 1
        fi
      else
        ENABLE_EXPERT_PARALLEL_VALUE="1"
      fi
      ;;
    --temp|--temperature)
      shift || { echo "[ERROR] --temp requires a value"; usage; exit 1; }
      TEMP_VALUE="$1"
      ;;
    --top-p)
      shift || { echo "[ERROR] --top-p requires a value"; usage; exit 1; }
      TOP_P_VALUE="$1"
      ;;
    --top-logprobs)
      shift || { echo "[ERROR] --top-logprobs requires a value"; usage; exit 1; }
      TOP_LOGPROBS_VALUE="$1"
      ;;
    --port)
      shift || { echo "[ERROR] --port requires a value"; usage; exit 1; }
      API_PORT="$1"
      ;;
    --out)
      shift || { echo "[ERROR] --out requires a value"; usage; exit 1; }
      OUTPUT_FILE="$1"
      ;;
    --data-path)
      shift || { echo "[ERROR] --data-path requires a value"; usage; exit 1; }
      DATA_PATH="$1"
      ;;
    --max-workers)
      shift || { echo "[ERROR] --max-workers requires a value"; usage; exit 1; }
      MAX_WORKERS="$1"
      ;;
    --gpu-memory-utilization)
      shift || { echo "[ERROR] --gpu-memory-utilization requires a value"; usage; exit 1; }
      GPU_MEM_UTIL="$1"
      ;;
    --server-only)
      SERVER_ONLY="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────
if [[ -z "${MODEL_DIR_VALUE}" ]]; then
  echo "[ERROR] --model-dir must be provided"
  usage
  exit 1
fi

if [[ "${SERVER_ONLY}" == "0" && -z "${TASK_VALUE}" ]]; then
  echo "[ERROR] --task must be provided (or use --server-only)"
  usage
  exit 1
fi

# Default values
[[ -z "${N_VALUE}" ]] && N_VALUE=64
[[ -z "${TP_SIZE_VALUE}" ]] && TP_SIZE_VALUE=1
[[ -z "${DP_SIZE_VALUE}" ]] && DP_SIZE_VALUE=1

if ! [[ "${N_VALUE}" =~ ^[0-9]+$ ]] || (( N_VALUE <= 0 )); then
  echo "[ERROR] --n must be a positive integer (got '${N_VALUE}')"
  exit 1
fi

if ! [[ "${TP_SIZE_VALUE}" =~ ^[0-9]+$ ]] || (( TP_SIZE_VALUE <= 0 )); then
  echo "[ERROR] --tp-size must be a positive integer (got '${TP_SIZE_VALUE}')"
  exit 1
fi

if ! [[ "${DP_SIZE_VALUE}" =~ ^[0-9]+$ ]] || (( DP_SIZE_VALUE <= 0 )); then
  echo "[ERROR] --dp-size must be a positive integer (got '${DP_SIZE_VALUE}')"
  exit 1
fi

# API server count defaults to 2x data-parallel size.
if [[ -z "${API_SERVER_COUNT_VALUE}" ]]; then
  API_SERVER_COUNT_VALUE=$(( DP_SIZE_VALUE * 2 ))
fi

if ! [[ "${API_SERVER_COUNT_VALUE}" =~ ^[0-9]+$ ]] || (( API_SERVER_COUNT_VALUE <= 0 )); then
  echo "[ERROR] --api-server-count must be a positive integer (got '${API_SERVER_COUNT_VALUE}')"
  exit 1
fi

if ! [[ "${ENABLE_EXPERT_PARALLEL_VALUE}" =~ ^[01]$ ]]; then
  echo "[ERROR] --enable-expert-parallel must be 0 or 1 (got '${ENABLE_EXPERT_PARALLEL_VALUE}')"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR_VALUE}"
MODEL_NAME=$(basename "${MODEL_DIR}")
TASK="${TASK_VALUE}"
N="${N_VALUE}"

# Expert Parallel args
EXPERT_PARALLEL_ARGS=()
if [[ "${ENABLE_EXPERT_PARALLEL_VALUE}" == "1" ]]; then
  EXPERT_PARALLEL_ARGS+=(--enable-expert-parallel)
  echo "[INFO] Expert Parallel ENABLED."
else
  echo "[INFO] Expert Parallel DISABLED."
fi

# Directories
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
mkdir -p "${RESULTS_DIR}"

# Output file
if [[ -z "${OUTPUT_FILE}" ]]; then
  OUTPUT_FILE="${RESULTS_DIR}/${TASK}_${MODEL_NAME}_${N}.jsonl"
fi

echo "=============================================="
echo "[INFO] PETS Benchmark Configuration"
echo "=============================================="
echo "[INFO] Model: ${MODEL_DIR}"
echo "[INFO] Model Name: ${MODEL_NAME}"
echo "[INFO] Task: ${TASK:-server-only}"
echo "[INFO] Tensor Parallel Size: ${TP_SIZE_VALUE}"
echo "[INFO] Data Parallel Size: ${DP_SIZE_VALUE}"
echo "[INFO] API Server Count: ${API_SERVER_COUNT_VALUE}"
echo "[INFO] Samples per question: ${N}"
echo "[INFO] Temperature: ${TEMP_VALUE}"
echo "[INFO] Top-P: ${TOP_P_VALUE}"
echo "[INFO] Top Logprobs: ${TOP_LOGPROBS_VALUE}"
echo "[INFO] API Port: ${API_PORT}"
echo "[INFO] Output File: ${OUTPUT_FILE}"
echo "=============================================="

# ──────────────────────────────────────────────────────────────────────────────
# vLLM Server with Confidence Plugin
# ──────────────────────────────────────────────────────────────────────────────
timestamp="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${LOG_DIR}/vllm_server_${timestamp}.log"

echo "[INFO] Launching vLLM API server with confidence plugin ..."
echo "[INFO] Server log: ${SERVER_LOG}"

# Enable confidence plugin (required)
export VLLM_PLUGINS="confidence_logprobs"
export VLLM_CONF_TOPK="${VLLM_CONF_TOPK:-20}"
export VLLM_CONF_MODE="${VLLM_CONF_MODE:-stats}"
export VLLM_FLAT_LOGPROBS=1

vllm serve "${MODEL_DIR}" \
  --api-server-count "${API_SERVER_COUNT_VALUE}" \
  --data-parallel-size "${DP_SIZE_VALUE}" \
  --tensor-parallel-size "${TP_SIZE_VALUE}" \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  "${EXPERT_PARALLEL_ARGS[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Cleanup on exit
cleanup() {
  echo "[INFO] Stopping vLLM server (PID: ${SERVER_PID}) ..."
  kill ${SERVER_PID} >/dev/null 2>&1 || true
  wait ${SERVER_PID} 2>/dev/null || true
  echo "[INFO] Server stopped."
}
trap cleanup EXIT

# Wait for server ready
echo "[INFO] Waiting for API server to become ready ..."
READY_DEADLINE=$((SECONDS + ${SERVER_READY_TIMEOUT:-7200}))
while true; do
  if ! kill -0 ${SERVER_PID} >/dev/null 2>&1; then
    echo "[ERROR] vLLM server exited prematurely. Last 30 lines of log:"
    tail -n 30 "${SERVER_LOG}" || true
    exit 1
  fi
  if python3 - <<PY 2>/dev/null
import json, urllib.request, urllib.error
try:
    with urllib.request.urlopen("http://127.0.0.1:${API_PORT}/v1/models", timeout=3) as resp:
        data = json.load(resp)
    if isinstance(data, dict) and data.get("data"):
        raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
  then
    echo "[INFO] vLLM is ready!"
    break
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "[ERROR] Timed out waiting for vLLM to become ready. Last 30 lines of log:"
    tail -n 30 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 3
done

# Server-only mode
if [[ "${SERVER_ONLY}" == "1" ]]; then
  echo "[INFO] Server-only mode. Server is running at http://localhost:${API_PORT}"
  echo "[INFO] Press Ctrl+C to stop."
  wait ${SERVER_PID}
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────────────
# Run Benchmark
# ──────────────────────────────────────────────────────────────────────────────
echo "[INFO] Running inference for task=${TASK} with N=${N} ..."
cd "${SCRIPT_DIR}"

# Clear old output
rm -f "${OUTPUT_FILE}"

# Common args
COMMON_ARGS=(
  --host localhost
  --port "${API_PORT}"
  --n "${N}"
  --temperature "${TEMP_VALUE}"
  --top_p "${TOP_P_VALUE}"
  --top_logprobs "${TOP_LOGPROBS_VALUE}"
  --max_workers "${MAX_WORKERS}"
  --out "${OUTPUT_FILE}"
)

# Add data path if provided
if [[ -n "${DATA_PATH}" ]]; then
  COMMON_ARGS+=(--data_path "${DATA_PATH}")
fi

case "${TASK}" in
  aime24)
    python3 aime24.py "${COMMON_ARGS[@]}"
    ;;
  aime25)
    python3 aime25.py "${COMMON_ARGS[@]}"
    ;;
  hmmt)
    python3 hmmt.py "${COMMON_ARGS[@]}"
    ;;
  brumo)
    python3 brumo.py "${COMMON_ARGS[@]}"
    ;;
  gpqa)
    python3 gpqa.py "${COMMON_ARGS[@]}"
    ;;
  mmlu_pro)
    python3 mmlu_pro.py "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "[ERROR] Unknown task: ${TASK}"
    echo "Available: aime24, aime25, hmmt, brumo, gpqa, mmlu_pro"
    exit 1
    ;;
esac

echo "=============================================="
echo "[INFO] Inference completed!"
echo "[INFO] Results saved to: ${OUTPUT_FILE}"
echo "=============================================="
