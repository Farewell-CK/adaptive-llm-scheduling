#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: test_dual_4gpu_benchmark.sh [options]

Starts (optional) two 4-GPU vLLM OpenAI servers (ports 8001/8002 by default),
assumes router is already running (default http://localhost:5000),
then runs a workload + benchmark and saves outputs into one run directory.

Options:
  --minutes N          Workload duration minutes (default: 10)
  --qps X              Offered QPS (default: 1.0)
  --out-root DIR       Output root dir (default: workspace/tools/deploy_tests)
  --skip-start         Do not start vLLM instances (assume already running)
  --keep-services      Do not stop vLLM instances on exit
  -h, --help           Show this help

Env overrides:
  CONDA_ENV_NAME=vllm
  MODEL_PATH=/data/coding/model/Llama-3.3-70B-Instruct-FP8-Dynamic
  SERVED_MODEL_NAME=llama-3.3-70b
  MODEL_NAME=llama-3.3-70b                (payload model field for router/benchmark)
  GPU_LONG=0,1,2,3  GPU_SHORT=4,5,6,7
  PORT_LONG=8001    PORT_SHORT=8002
  ROUTER_URL=http://localhost:5000/v1/chat/completions
  GPU_MEMORY_UTILIZATION=0.90  MAX_MODEL_LEN=8192
  READY_TIMEOUT_SECONDS=180
  START_DELAY_SECONDS=5
USAGE
}

MINUTES="${MINUTES:-10}"
QPS="${QPS:-1.0}"
OUT_ROOT="${OUT_ROOT:-}"
SKIP_START_VLLM="${SKIP_START_VLLM:-0}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2 ;;
    --qps) QPS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --skip-start) SKIP_START_VLLM=1; shift ;;
    --keep-services) KEEP_SERVICES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"
MODEL_PATH="${MODEL_PATH:-/data/coding/model/Llama-3.3-70B-Instruct-FP8-Dynamic}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.3-70b}"
MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"

GPU_LONG="${GPU_LONG:-0,1,2,3}"
GPU_SHORT="${GPU_SHORT:-4,5,6,7}"
PORT_LONG="${PORT_LONG:-8001}"
PORT_SHORT="${PORT_SHORT:-8002}"
ROUTER_URL="${ROUTER_URL:-http://localhost:5000/v1/chat/completions}"

DTYPE="${DTYPE:-auto}"
AUTODETECT_QUANTIZATION="${AUTODETECT_QUANTIZATION:-1}"
QUANTIZATION="${QUANTIZATION:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HOST="${HOST:-0.0.0.0}"

READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-180}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-5}"
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"

_count_csv_items() {
  local s="${1//[[:space:]]/}"
  local IFS=','
  # shellcheck disable=SC2206
  local arr=($s)
  echo "${#arr[@]}"
}

TP_LONG="${TP_LONG:-$(_count_csv_items "${GPU_LONG}")}"
TP_SHORT="${TP_SHORT:-$(_count_csv_items "${GPU_SHORT}")}"

detect_quant_method() {
  python3 - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("${MODEL_PATH}") / "config.json"
if not p.exists():
  raise SystemExit(0)
obj = json.loads(p.read_text())
qc = obj.get("quantization_config") or {}
qm = qc.get("quant_method") or ""
print(qm)
PY
}

if [[ -z "${QUANTIZATION}" && "${AUTODETECT_QUANTIZATION}" == "1" ]]; then
  _qm="$(detect_quant_method)"
  if [[ -n "${_qm}" ]]; then
    QUANTIZATION="${_qm}"
  fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
TOOLS_DIR="${WORKSPACE_DIR}/tools"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${TOOLS_DIR}/deploy_tests"
fi

RUN_DIR="${OUT_ROOT}/deploy_dual4_${RUN_ID}"
mkdir -p "${RUN_DIR}"

activate_conda_env() {
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    return 0
  fi

  local python_path=""
  python_path="$(command -v python3 2>/dev/null || true)"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" \
    && -n "${CONDA_PREFIX:-}" \
    && -n "${python_path}" \
    && "${python_path}" == "${CONDA_PREFIX}/"* ]]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    return 0
  fi

  local -a candidates=()
  if [[ -n "${CONDA_EXE:-}" ]]; then
    candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ "${CONDA_PREFIX}" == *"/envs/"* ]]; then
      candidates+=("${CONDA_PREFIX%/envs/*}")
    else
      candidates+=("${CONDA_PREFIX}")
    fi
  fi
  if [[ -n "${CONDA_BASE:-}" ]]; then
    candidates+=("${CONDA_BASE}")
  fi
  candidates+=(
    "/opt/conda"
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "$HOME/miniforge3"
    "$HOME/mambaforge"
    "/data/miniconda"
  )

  local base
  for base in "${candidates[@]}"; do
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${base}/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME}"
      return 0
    fi
  done

  echo "ERROR: 未找到 conda。请先确保 conda 可用，或设置 CONDA_BASE=/path/to/conda，然后再运行脚本。" >&2
  exit 1
}

port_available() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    if ss -ltnH | awk '{print $4}' | grep -Eq "(:|\\])${port}$"; then
      return 1
    fi
  fi
  return 0
}

wait_http_200() {
  local url="$1"
  local timeout_s="$2"
  local start
  start="$(date +%s)"
  while true; do
    if command -v curl >/dev/null 2>&1; then
      if curl -sS -o /dev/null -m 2 -w "%{http_code}" "${url}" | grep -q "^200$"; then
        return 0
      fi
    else
      python3 - <<PY >/dev/null 2>&1
import urllib.request, sys
try:
  with urllib.request.urlopen("${url}", timeout=2) as r:
    sys.exit(0 if r.status == 200 else 1)
except Exception:
  sys.exit(1)
PY
      if [[ $? -eq 0 ]]; then
        return 0
      fi
    fi

    local now
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      return 1
    fi
    sleep 2
  done
}

PIDS=()
STARTED_BY_ME=0
cleanup() {
  if [[ "${KEEP_SERVICES}" == "1" ]]; then
    return 0
  fi
  if [[ "${STARTED_BY_ME}" != "1" ]]; then
    return 0
  fi
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "========================================================"
echo "Dual 4-GPU Deploy Test"
echo "RUN_DIR: ${RUN_DIR}"
echo "Router:  ${ROUTER_URL}"
echo "Model:   ${MODEL_NAME}"
echo "Workload: ${MINUTES} min @ ${QPS} QPS"
echo "========================================================"

{
  echo "RUN_ID=${RUN_ID}"
  echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "GPU_LONG=${GPU_LONG} PORT_LONG=${PORT_LONG}"
  echo "GPU_SHORT=${GPU_SHORT} PORT_SHORT=${PORT_SHORT}"
  echo "ROUTER_URL=${ROUTER_URL}"
  echo "DTYPE=${DTYPE} QUANTIZATION=${QUANTIZATION} KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
  echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
} > "${RUN_DIR}/config.env"

activate_conda_env

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "❌ ERROR: MODEL_PATH 不存在：${MODEL_PATH}" >&2
  exit 1
fi

for mod in vllm fastapi uvicorn httpx aiohttp; do
  if ! python3 - <<PY >/dev/null 2>&1
import ${mod}  # noqa: F401
PY
  then
    echo "❌ ERROR: Missing python module in env '${CONDA_DEFAULT_ENV:-<none>}': ${mod}" >&2
    exit 1
  fi
done

start_instance() {
  local name="$1"
  local gpus="$2"
  local port="$3"
  local tp_size="$4"
  local log_file="${RUN_DIR}/vllm_${name}.log"
  local pid_file="${RUN_DIR}/vllm_${name}.pid"

  if ! port_available "${port}"; then
    echo "❌ ERROR: Port ${port} already in use." >&2
    exit 1
  fi

  echo "---------------------------------------------------"
  echo "🚀 Starting vLLM ${name}"
  echo "   - GPUs: ${gpus}"
  echo "   - Port: ${port}"
  echo "   - TP:   ${tp_size}"
  echo "   - Log:  ${log_file}"

  local -a extra_args=()
  if [[ -n "${QUANTIZATION}" ]]; then
    extra_args+=(--quantization "${QUANTIZATION}")
  fi
  if [[ -n "${KV_CACHE_DTYPE}" ]]; then
    extra_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  fi

  CUDA_VISIBLE_DEVICES="${gpus//[[:space:]]/}" nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${tp_size}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --host "${HOST}" \
    --port "${port}" \
    --trust-remote-code \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${pid_file}"
  PIDS+=("${pid}")
}

if [[ "${SKIP_START_VLLM}" != "1" ]]; then
  STARTED_BY_ME=1
  start_instance "A_Long" "${GPU_LONG}" "${PORT_LONG}" "${TP_LONG}"
  sleep "${START_DELAY_SECONDS}"
  start_instance "B_Short" "${GPU_SHORT}" "${PORT_SHORT}" "${TP_SHORT}"
  sleep "${START_DELAY_SECONDS}"
fi

echo "Waiting for workers to be ready..."
if ! wait_http_200 "http://localhost:${PORT_LONG}/v1/models" "${READY_TIMEOUT_SECONDS}"; then
  echo "❌ ERROR: Worker LONG not ready on port ${PORT_LONG} (GET /v1/models)." >&2
  exit 1
fi
if ! wait_http_200 "http://localhost:${PORT_SHORT}/v1/models" "${READY_TIMEOUT_SECONDS}"; then
  echo "❌ ERROR: Worker SHORT not ready on port ${PORT_SHORT} (GET /v1/models)." >&2
  exit 1
fi

echo "Smoke test via router..."
if command -v curl >/dev/null 2>&1; then
  code="$(curl -sS -m 30 -o "${RUN_DIR}/smoke_response.json" -w "%{http_code}" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"stream\":false}" \
    "${ROUTER_URL}" || true)"
  if [[ "${code}" != "200" ]]; then
    echo "❌ ERROR: Router smoke test failed (HTTP ${code}). See ${RUN_DIR}/smoke_response.json" >&2
    exit 1
  fi
else
  python3 - <<PY
import json, urllib.request
data=json.dumps({"model":"${MODEL_NAME}","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"stream":False}).encode()
req=urllib.request.Request("${ROUTER_URL}", data=data, headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
  body=r.read()
open("${RUN_DIR}/smoke_response.json","wb").write(body)
PY
fi

MONITOR_LONG_PID=""
MONITOR_SHORT_PID=""
stop_monitors() {
  if [[ -n "${MONITOR_LONG_PID}" ]]; then
    kill "${MONITOR_LONG_PID}" 2>/dev/null || true
    wait "${MONITOR_LONG_PID}" 2>/dev/null || true
    MONITOR_LONG_PID=""
  fi
  if [[ -n "${MONITOR_SHORT_PID}" ]]; then
    kill "${MONITOR_SHORT_PID}" 2>/dev/null || true
    wait "${MONITOR_SHORT_PID}" 2>/dev/null || true
    MONITOR_SHORT_PID=""
  fi
}
trap 'stop_monitors; cleanup' EXIT INT TERM

monitor_duration=$(( MINUTES * 60 + 600 ))
echo "Starting metrics monitors (${monitor_duration}s)..."
python3 "${TOOLS_DIR}/monitor_vllm.py" \
  --url "http://localhost:${PORT_LONG}/metrics" \
  --duration "${monitor_duration}" \
  --output "${RUN_DIR}/metrics_long.csv" \
  > "${RUN_DIR}/monitor_long.log" 2>&1 &
MONITOR_LONG_PID=$!

python3 "${TOOLS_DIR}/monitor_vllm.py" \
  --url "http://localhost:${PORT_SHORT}/metrics" \
  --duration "${monitor_duration}" \
  --output "${RUN_DIR}/metrics_short.csv" \
  > "${RUN_DIR}/monitor_short.log" 2>&1 &
MONITOR_SHORT_PID=$!

echo "Generating trace..."
python3 "${TOOLS_DIR}/workload_gen.py" --minutes "${MINUTES}" --qps "${QPS}" --output "${RUN_DIR}/trace.jsonl"

echo "Running benchmark..."
python3 "${TOOLS_DIR}/benchmark_client.py" \
  --trace "${RUN_DIR}/trace.jsonl" \
  --output "${RUN_DIR}/result.csv" \
  --url "${ROUTER_URL}" \
  --model "${MODEL_NAME}"

echo "Stopping monitors..."
stop_monitors

echo "========================================================"
echo "✅ Done."
echo "Run dir: ${RUN_DIR}"
echo "  - ${RUN_DIR}/result.csv"
echo "  - ${RUN_DIR}/metrics_long.csv"
echo "  - ${RUN_DIR}/metrics_short.csv"
echo "  - ${RUN_DIR}/smoke_response.json"
echo "========================================================"
