#!/usr/bin/env bash

set -euo pipefail

# ================= 0. 环境准备 =================
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

# 避免因为 CWD 下存在同名目录（例如 ./vllm）导致 Python 导入被意外 shadow。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
TOOLS_DIR="${WORKSPACE_DIR}/tools"

RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"

# 创建日志目录
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# ================= 1. 全局配置 =================
# 模型路径 (FP8 版本)
MODEL_PATH="${MODEL_PATH:-/data/coding/model/Llama-3.3-70B-Instruct-FP8-Dynamic}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "❌ ERROR: MODEL_PATH 不存在：${MODEL_PATH}" >&2
    exit 1
fi

# 通用 vLLM 参数
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.3-70b}"
DTYPE="${DTYPE:-auto}"
AUTODETECT_QUANTIZATION="${AUTODETECT_QUANTIZATION:-1}"
# 留空表示不传 --quantization，让 vLLM 按模型 config.json 的 quantization_config 自动选择
QUANTIZATION="${QUANTIZATION:-}"
# 留空表示不传 --kv-cache-dtype
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"  # 4张卡跑70B比较极限，留一点余量，或者设为 0.95
HOST="${HOST:-0.0.0.0}"

# 实例配置（可用环境变量覆盖）
GPU_LONG="${GPU_LONG:-0,1,2,3}"
GPU_SHORT="${GPU_SHORT:-4,5,6,7}"
PORT_LONG="${PORT_LONG:-8001}"
PORT_SHORT="${PORT_SHORT:-8002}"

ROUTER_IMPL="${ROUTER_IMPL:-smart}"   # smart | static
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-5000}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-3000}"
ROUTER_MONITOR_WINDOW="${ROUTER_MONITOR_WINDOW:-10}"
START_ROUTER="${START_ROUTER:-1}"

port_available() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        if ss -ltnH | awk '{print $4}' | grep -Eq "(:|\\])${port}$"; then
            return 1
        fi
    fi
    return 0
}

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

_count_csv_items() {
    local s="${1//[[:space:]]/}"
    local IFS=','
    # shellcheck disable=SC2206
    local arr=($s)
    echo "${#arr[@]}"
}

TP_LONG="${TP_LONG:-$(_count_csv_items "${GPU_LONG}")}"
TP_SHORT="${TP_SHORT:-$(_count_csv_items "${GPU_SHORT}")}"

_python_import_check() {
    local module="$1"
    python3 - <<PY >/dev/null 2>&1
import ${module}  # noqa: F401
PY
}

for mod in vllm httpx fastapi uvicorn; do
    if ! _python_import_check "${mod}"; then
        echo "❌ ERROR: Python module missing in env '${CONDA_DEFAULT_ENV:-<none>}': ${mod}" >&2
        echo "   请先进入正确的 conda env，或安装依赖后再启动。" >&2
        exit 1
    fi
done

# ================= 2. 启动函数 =================
start_instance() {
    local instance_name=$1
    local gpu_ids=$2
    local port=$3
    local tp_size=$4
    local log_file="${LOG_DIR}/vllm_${instance_name}_${RUN_ID}.log"
    local pid_file="${LOG_DIR}/vllm_${instance_name}.pid"

    if ! port_available "${port}"; then
        echo "❌ ERROR: Port ${port} already in use. Stop existing process or set PORT_LONG/PORT_SHORT." >&2
        exit 1
    fi

    echo "---------------------------------------------------"
    echo "🚀 Starting Instance: ${instance_name}"
    echo "   - GPUs: ${gpu_ids}"
    echo "   - Port: ${port}"
    echo "   - TP Size: ${tp_size}"
    echo "   - Log:  ${log_file}"

    local -a extra_args=()
    if [[ -n "${QUANTIZATION}" ]]; then
        extra_args+=(--quantization "${QUANTIZATION}")
    fi
    if [[ -n "${KV_CACHE_DTYPE}" ]]; then
        extra_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    fi

    # 使用 nohup 后台启动，重定向日志
    # 注意：CUDA_VISIBLE_DEVICES 必须在命令前指定
    CUDA_VISIBLE_DEVICES=${gpu_ids} nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${tp_size}" \
        --dtype "${DTYPE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --port "${port}" \
        --host "${HOST}" \
        --trust-remote-code \
        "${extra_args[@]}" \
        > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" > "${pid_file}"
    STARTED_PID="${pid}"
}

start_router() {
    local log_file="${LOG_DIR}/router_${ROUTER_IMPL}_${RUN_ID}.log"
    local pid_file="${LOG_DIR}/router.pid"

    if ! port_available "${ROUTER_PORT}"; then
        echo "❌ ERROR: Router port ${ROUTER_PORT} already in use." >&2
        echo "   If you already started router manually, run: START_ROUTER=0 bash start_vllm_dual.sh" >&2
        exit 1
    fi

    local router_script=""
    case "${ROUTER_IMPL}" in
        smart)  router_script="${TOOLS_DIR}/router_smart.py" ;;
        static) router_script="${WORKSPACE_DIR}/router_static.py" ;;
        *)
            echo "❌ ERROR: Unknown ROUTER_IMPL='${ROUTER_IMPL}' (expected smart|static)" >&2
            exit 1
            ;;
    esac

    if [[ ! -f "${router_script}" ]]; then
        echo "❌ ERROR: Router script not found: ${router_script}" >&2
        exit 1
    fi

    echo "---------------------------------------------------"
    echo "🧭 Starting Router (${ROUTER_IMPL})"
    echo "   - Listen: ${ROUTER_HOST}:${ROUTER_PORT}"
    echo "   - Long Worker:  http://localhost:${PORT_LONG}"
    echo "   - Short Worker: http://localhost:${PORT_SHORT}"
    echo "   - Threshold: ${ROUTER_THRESHOLD}"
    echo "   - Log: ${log_file}"

    URL_WORKER_LONG="http://localhost:${PORT_LONG}/v1/chat/completions" \
    URL_WORKER_SHORT="http://localhost:${PORT_SHORT}/v1/chat/completions" \
    STATIC_THRESHOLD="${ROUTER_THRESHOLD}" \
    MONITOR_WINDOW="${ROUTER_MONITOR_WINDOW}" \
    ROUTER_HOST="${ROUTER_HOST}" \
    ROUTER_PORT="${ROUTER_PORT}" \
    nohup python3 "${router_script}" > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" > "${pid_file}"
    STARTED_PID="${pid}"
}

# ================= 3. 执行启动 =================

echo "Preparing to start 2 vLLM instances + router..."
echo "Model Path: ${MODEL_PATH}"
echo "Conda Env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "RUN_ID: ${RUN_ID}"

# --- Instance A (Prefill/Long) ---
# 使用 GPU 0,1,2,3
start_instance "A_Long" "${GPU_LONG}" "${PORT_LONG}" "${TP_LONG}"
PID_A="${STARTED_PID}"
echo "✅ Instance A started with PID: $PID_A"

# 等待几秒，防止端口或显存初始化冲突（可选）
sleep 5

# --- Instance B (Decode/Short) ---
# 使用 GPU 4,5,6,7
start_instance "B_Short" "${GPU_SHORT}" "${PORT_SHORT}" "${TP_SHORT}"
PID_B="${STARTED_PID}"
echo "✅ Instance B started with PID: $PID_B"

sleep 2

PID_ROUTER=""
if [[ "${START_ROUTER}" == "1" ]]; then
    start_router
    PID_ROUTER="${STARTED_PID}"
    echo "✅ Router started with PID: $PID_ROUTER"
else
    echo "ℹ️  START_ROUTER=0 set; skipping router start."
fi

echo "---------------------------------------------------"
echo "🎉 Environment is up."
echo "   - Long Text Worker:  http://localhost:${PORT_LONG}"
echo "   - Short Text Worker: http://localhost:${PORT_SHORT}"
echo "   - Router:            http://localhost:${ROUTER_PORT} (START_ROUTER=${START_ROUTER})"
echo "   - Logs:              ${LOG_DIR}"
echo "   - For benchmark_client.py:"
echo "       --url http://localhost:${ROUTER_PORT}/v1/chat/completions"
echo "   - For monitor_vllm.py (single instance metrics):"
echo "       --url http://localhost:${PORT_LONG}/metrics   (or :${PORT_SHORT}/metrics)"
echo "⏳ Waiting for processes... (Press Ctrl+C to stop ALL)"

# ================= 4. 进程守护与清理 =================

# 定义清理函数：当脚本收到 Ctrl+C (SIGINT) 或被杀 (SIGTERM) 时执行
cleanup() {
    echo ""
    echo "🛑 Stopping router + vLLM instances..."
    kill "${PID_ROUTER:-}" 2>/dev/null || true
    kill "${PID_A:-}" 2>/dev/null || true
    kill "${PID_B:-}" 2>/dev/null || true
    wait "${PID_ROUTER:-}" 2>/dev/null || true
    wait "${PID_A:-}" 2>/dev/null || true
    wait "${PID_B:-}" 2>/dev/null || true
    echo "✅ All processes stopped."
    exit 0
}

# 注册信号捕获
trap cleanup SIGINT SIGTERM

if [[ "${DETACH:-0}" == "1" ]]; then
    echo "DETACH=1 set; exiting without waiting."
    exit 0
fi

# 阻塞脚本，等待子进程
wait
