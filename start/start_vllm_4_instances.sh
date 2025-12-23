#!/usr/bin/env bash

set -euo pipefail

# ================= 0. 环境准备 =================
# 目标：启动 4 个 vLLM OpenAI Server 实例，供上层 Router/负载均衡做动态路由。
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"

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

activate_conda_env

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# ================= 1. 全局配置 =================
MODEL_PATH="${MODEL_PATH:-/data/coding/model/Llama-3.3-70B-Instruct-FP8-Dynamic}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "❌ ERROR: MODEL_PATH 不存在：${MODEL_PATH}" >&2
    exit 1
fi

SERVED_MODEL_NAME_PREFIX="${SERVED_MODEL_NAME_PREFIX:-llama-3.3-70b-unit}"
DTYPE="${DTYPE:-auto}"
AUTODETECT_QUANTIZATION="${AUTODETECT_QUANTIZATION:-1}"
QUANTIZATION="${QUANTIZATION:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HOST="${HOST:-0.0.0.0}"
CHECK_PORTS="${CHECK_PORTS:-1}"

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

# 4 个实例的默认 GPU/端口（可通过环境变量覆盖）
GPU_1="${GPU_1:-0,1}"
GPU_2="${GPU_2:-2,3}"
GPU_3="${GPU_3:-4,5}"
GPU_4="${GPU_4:-6,7}"

PORT_1="${PORT_1:-8001}"
PORT_2="${PORT_2:-8002}"
PORT_3="${PORT_3:-8003}"
PORT_4="${PORT_4:-8004}"

_count_csv_items() {
    local s="${1//[[:space:]]/}"
    local IFS=','
    # shellcheck disable=SC2206
    local arr=($s)
    echo "${#arr[@]}"
}

TP_1="${TP_1:-$(_count_csv_items "${GPU_1}")}"
TP_2="${TP_2:-$(_count_csv_items "${GPU_2}")}"
TP_3="${TP_3:-$(_count_csv_items "${GPU_3}")}"
TP_4="${TP_4:-$(_count_csv_items "${GPU_4}")}"

if ! python3 -c 'import vllm' >/dev/null 2>&1; then
    echo "❌ ERROR: vllm 未安装或不在当前 Python 环境中（CONDA_ENV_NAME=${CONDA_ENV_NAME}）。" >&2
    exit 1
fi

port_available() {
    local port="$1"
    if ! command -v ss >/dev/null 2>&1; then
        return 0
    fi
    if ss -ltnH | awk '{print $4}' | grep -Eq "(:|\\])${port}$"; then
        return 1
    fi
    return 0
}

# ================= 2. 启动/清理 =================
START_DELAY_SECONDS="${START_DELAY_SECONDS:-2}"
DETACH="${DETACH:-0}"

PIDS=()

start_worker() {
    local id="$1"
    local gpus="$2"
    local port="$3"
    local tp_size="$4"
    local served_name="${SERVED_MODEL_NAME_PREFIX}-${id}"
    local log_file="${LOG_DIR}/unit_${id}_${RUN_ID}.log"
    local pid_file="${LOG_DIR}/unit_${id}.pid"

    echo "---------------------------------------------------"
    echo "🚀 Starting Unit ${id}"
    echo "   - GPUs: ${gpus}"
    echo "   - Port: ${port}"
    echo "   - TP:   ${tp_size}"
    echo "   - Name: ${served_name}"
    echo "   - Log:  ${log_file}"

    if [[ -z "${gpus//[[:space:]]/}" ]]; then
        echo "❌ ERROR: Unit ${id} GPU list is empty." >&2
        exit 1
    fi
    if [[ "${tp_size}" -lt 1 ]]; then
        echo "❌ ERROR: Unit ${id} tensor-parallel-size must be >= 1 (got ${tp_size})." >&2
        exit 1
    fi

    local -a extra_args=()
    if [[ -n "${QUANTIZATION}" ]]; then
        extra_args+=(--quantization "${QUANTIZATION}")
    fi
    if [[ -n "${KV_CACHE_DTYPE}" ]]; then
        extra_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    fi

    CUDA_VISIBLE_DEVICES="${gpus//[[:space:]]/}" nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${served_name}" \
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
    echo "✅ Unit ${id} started with PID: ${pid}"
}

cleanup() {
    echo ""
    echo "🛑 Stopping vLLM units..."
    local pid
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    echo "✅ All units stopped."
}

trap cleanup INT TERM

# ================= 3. 执行启动 =================
echo "Starting 4 vLLM units..."
echo "Model Path: ${MODEL_PATH}"
echo "Conda Env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "RUN_ID: ${RUN_ID}"

if [[ "${CHECK_PORTS}" == "1" ]]; then
    for p in "${PORT_1}" "${PORT_2}" "${PORT_3}" "${PORT_4}"; do
        if ! port_available "${p}"; then
            echo "❌ ERROR: Port ${p} is already in use. Set PORT_*=... or stop the existing process." >&2
            exit 1
        fi
    done
fi

start_worker 1 "${GPU_1}" "${PORT_1}" "${TP_1}"
sleep "${START_DELAY_SECONDS}"
start_worker 2 "${GPU_2}" "${PORT_2}" "${TP_2}"
sleep "${START_DELAY_SECONDS}"
start_worker 3 "${GPU_3}" "${PORT_3}" "${TP_3}"
sleep "${START_DELAY_SECONDS}"
start_worker 4 "${GPU_4}" "${PORT_4}" "${TP_4}"

echo "---------------------------------------------------"
echo "🎉 All 4 units started."
echo "   - Unit1: http://localhost:${PORT_1}"
echo "   - Unit2: http://localhost:${PORT_2}"
echo "   - Unit3: http://localhost:${PORT_3}"
echo "   - Unit4: http://localhost:${PORT_4}"
echo "   - Logs:  ${LOG_DIR}"
echo "⏳ Waiting for processes... (Press Ctrl+C to stop ALL)"

if [[ "${DETACH:-0}" == "1" ]]; then
    echo "DETACH=1 set; leaving processes in background."
    exit 0
fi

wait
