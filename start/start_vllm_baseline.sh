#!/bin/bash

set -euo pipefail

# ================= 0. 环境准备 =================
# 在脚本里不要直接用 `source activate xxx`（容易在非交互 shell 下失败）。
# 推荐：先加载 conda 的 hook / conda.sh，再 `conda activate`。
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

# ================= 1. 路径配置 =================
# 你的 Llama 3.3 模型路径
MODEL_PATH="${MODEL_PATH:-/data/coding/model/Llama-3.3-70B-Instruct-FP8-Dynamic}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH 不存在或不是目录：${MODEL_PATH}" >&2
    exit 1
fi

# ================= 2. 启动命令 =================
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # 显卡全开（可通过外部覆盖：CUDA_VISIBLE_DEVICES=... ./start_vllm_baseline.sh）
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    # export CUDA_VISIBLE_DEVICES="0,1"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
IFS=',' read -r -a _CUDA_DEVS <<< "${CUDA_VISIBLE_DEVICES}"
TP_SIZE="${TP_SIZE:-${#_CUDA_DEVS[@]}}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.3-70b}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting vLLM..."
echo "Model Path: $MODEL_PATH"
echo "Conda Env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Tensor Parallel Size: ${TP_SIZE}"

# 启动 vLLM
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --trust-remote-code
