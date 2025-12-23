#!/usr/bin/env bash

set -u
set -o pipefail

# ================= 实验配置列表 (在这里修改) =================

# 实验时长列表 (分钟)，例如: (30 60)
DURATION_LIST=(30 40 60)

# QPS 列表，例如: (1.0 1.5 2.0)
QPS_LIST=(1.0 1.5 2.0 2.5 3.0 3.5 4.0)

# 每次实验之间的冷却时间 (秒)
# 建议至少设为 100s，等待 vLLM 彻底消化完上一轮的积压请求
COOLDOWN_SECONDS=100

# ===========================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/experiments}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK_URL="${BENCHMARK_URL:-}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-}"
METRICS_URL="${METRICS_URL:-}"
METRICS_URL_LONG="${METRICS_URL_LONG:-${METRICS_URL}}"
METRICS_URL_SHORT="${METRICS_URL_SHORT:-}"

RUN_TAG="${RUN_TAG:-overnight_$(date +"%Y%m%d_%H%M%S")}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/$RUN_TAG}"

mkdir -p "$RUN_ROOT"

CURRENT_MONITOR_LONG_PID=""
CURRENT_MONITOR_SHORT_PID=""

if ! "$PYTHON_BIN" -c 'import aiohttp' >/dev/null 2>&1; then
    echo "❌ Missing Python dependency: aiohttp"
    echo "   Install (recommended in a venv): pip install aiohttp"
    exit 1
fi

cleanup() {
    if [ -n "${CURRENT_MONITOR_LONG_PID}" ]; then
        kill "$CURRENT_MONITOR_LONG_PID" 2>/dev/null || true
        wait "$CURRENT_MONITOR_LONG_PID" 2>/dev/null || true
        CURRENT_MONITOR_LONG_PID=""
    fi
    if [ -n "${CURRENT_MONITOR_SHORT_PID}" ]; then
        kill "$CURRENT_MONITOR_SHORT_PID" 2>/dev/null || true
        wait "$CURRENT_MONITOR_SHORT_PID" 2>/dev/null || true
        CURRENT_MONITOR_SHORT_PID=""
    fi
}

# 捕捉退出/中断信号，确保退出时杀掉后台监控进程（避免 kill 0 误伤当前终端）
trap cleanup EXIT INT TERM

run_one_experiment() {
    local min="$1"
    local qps="$2"

    local timestamp
    timestamp="$(date +"%Y%m%d_%H%M%S")"

    local qps_tag="${qps//./_}"
    local exp_dir="${RUN_ROOT}/experiment_qps${qps_tag}_min${min}_${timestamp}"
    mkdir -p "$exp_dir"

    echo ""
    echo "########################################################"
    echo "   ▶️  Running Experiment $CURRENT_RUN / $TOTAL_RUNS"
    echo "   📂 Directory: $exp_dir"
    echo "   ⏱️  Configuration: ${min} mins @ ${qps} QPS"
    echo "########################################################"

	    local trace_file="${exp_dir}/trace_qps${qps_tag}_min${min}.jsonl"
	    local result_file="${exp_dir}/result_qps${qps_tag}_min${min}.csv"
	    local log_file="${exp_dir}/execution.log"
	    local metrics_file="${exp_dir}/metrics_qps${qps_tag}_min${min}.csv"     # 聚合（供汇总脚本使用）
	    local metrics_long_file="${exp_dir}/metrics_long.csv"                  # Instance A / Long（原始）
	    local metrics_short_file="${exp_dir}/metrics_short.csv"                # Instance B / Short（原始）
	    local monitor_long_log="${exp_dir}/monitor_long.log"
	    local monitor_short_log="${exp_dir}/monitor_short.log"

    # 让本轮实验的所有输出同时写到屏幕和日志文件（不使用 pipeline，避免 subshell 导致 continue/变量失效）
    exec 3>&1 4>&2
    exec > >(tee -a "$log_file") 2>&1

    echo "[$(date)] === Experiment Start: QPS=$qps, Min=$min ==="
    echo "[$(date)] Output Dir: $exp_dir"

    echo "[$(date)] Step 1: Generating Workload Trace..."
    if ! "$PYTHON_BIN" "$SCRIPT_DIR/workload_gen.py" --minutes "$min" --qps "$qps" --output "$trace_file"; then
        echo "❌ Error: Trace generation failed."
        exec 1>&3 2>&4
        exec 3>&- 4>&-
        return 1
    fi

	    local monitor_duration=$((min * 60 + 600))
	    echo "[$(date)] Step 2: Starting Monitor (Duration: ${monitor_duration}s)..."
	    if [ -n "${METRICS_URL_LONG}" ]; then
	        local monitor_long_cmd=("$PYTHON_BIN" "$SCRIPT_DIR/monitor_vllm.py" --duration "$monitor_duration" --output "$metrics_long_file" --url "${METRICS_URL_LONG}")
	        "${monitor_long_cmd[@]}" >"$monitor_long_log" 2>&1 &
	        CURRENT_MONITOR_LONG_PID=$!
	        echo "   -> Monitor LONG PID: $CURRENT_MONITOR_LONG_PID (${METRICS_URL_LONG})"
	    fi
	    if [ -n "${METRICS_URL_SHORT}" ]; then
	        local monitor_short_cmd=("$PYTHON_BIN" "$SCRIPT_DIR/monitor_vllm.py" --duration "$monitor_duration" --output "$metrics_short_file" --url "${METRICS_URL_SHORT}")
	        "${monitor_short_cmd[@]}" >"$monitor_short_log" 2>&1 &
	        CURRENT_MONITOR_SHORT_PID=$!
	        echo "   -> Monitor SHORT PID: $CURRENT_MONITOR_SHORT_PID (${METRICS_URL_SHORT})"
	    fi
	    if [ -z "${METRICS_URL_LONG}" ] && [ -z "${METRICS_URL_SHORT}" ]; then
	        echo "⚠️  Warning: METRICS_URL_LONG/METRICS_URL_SHORT not set; metrics will be skipped."
	    fi

    echo "[$(date)] Step 3: Running Benchmark Client..."
    local client_exit_code=0
    local bench_cmd=("$PYTHON_BIN" "$SCRIPT_DIR/benchmark_client.py" --trace "$trace_file" --output "$result_file")
    if [ -n "${BENCHMARK_URL}" ]; then
        bench_cmd+=(--url "${BENCHMARK_URL}")
    fi
    if [ -n "${BENCHMARK_MODEL}" ]; then
        bench_cmd+=(--model "${BENCHMARK_MODEL}")
    fi
    if (cd "$exp_dir" && "${bench_cmd[@]}"); then
        client_exit_code=0
    else
        client_exit_code=$?
    fi

	    echo "[$(date)] Step 4: Stopping Monitor..."
	    cleanup

	    echo "[$(date)] Step 5: Aggregating Metrics..."
	    if [ -s "$metrics_long_file" ] && [ -s "$metrics_short_file" ]; then
	        "$PYTHON_BIN" - <<PY
import csv
from pathlib import Path

long_path = Path("${metrics_long_file}")
short_path = Path("${metrics_short_file}")
out_path = Path("${metrics_file}")

def read_rows(p: Path):
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames or [], list(r)

long_fields, long_rows = read_rows(long_path)
short_fields, short_rows = read_rows(short_path)

target_metrics = [c for c in long_fields if c not in ("timestamp", "elapsed_seconds")]
kv_key = "vllm:kv_cache_usage_perc"

n = min(len(long_rows), len(short_rows))
if n == 0:
    raise SystemExit(0)

out_fields = ["timestamp", "elapsed_seconds"] + target_metrics
out_fields += [f"long_{m}" for m in target_metrics]
out_fields += [f"short_{m}" for m in target_metrics]

def f(x):
    try:
        return float(x)
    except Exception:
        return 0.0

with out_path.open("w", newline="") as f_out:
    w = csv.DictWriter(f_out, fieldnames=out_fields)
    w.writeheader()
    for i in range(n):
        lr = long_rows[i]
        sr = short_rows[i]
        row = {
            "timestamp": lr.get("timestamp") or sr.get("timestamp") or "",
            "elapsed_seconds": lr.get("elapsed_seconds") or sr.get("elapsed_seconds") or "",
        }
        for m in target_metrics:
            lv = f(lr.get(m))
            sv = f(sr.get(m))
            if m == kv_key:
                row[m] = max(lv, sv)
            else:
                row[m] = lv + sv
            row[f"long_{m}"] = lv
            row[f"short_{m}"] = sv
        w.writerow(row)
PY
	        echo "   -> Aggregated metrics saved to: $metrics_file"
	    elif [ -s "$metrics_long_file" ]; then
	        cp -f "$metrics_long_file" "$metrics_file"
	        echo "   -> Only LONG metrics found; copied to: $metrics_file"
	    elif [ -s "$metrics_short_file" ]; then
	        cp -f "$metrics_short_file" "$metrics_file"
	        echo "   -> Only SHORT metrics found; copied to: $metrics_file"
	    else
	        echo "   -> No metrics captured."
	    fi

	    if [ -s "$metrics_file" ]; then
	        echo "   -> Metrics (agg) saved to: $metrics_file"
	    fi
	    if [ -s "$metrics_long_file" ]; then
	        echo "   -> Metrics (long) saved to: $metrics_long_file"
	    fi
	    if [ -s "$metrics_short_file" ]; then
	        echo "   -> Metrics (short) saved to: $metrics_short_file"
	    fi

	    if [ "$client_exit_code" -eq 0 ]; then
	        echo "[$(date)] ✅ Experiment Finished Successfully."
    else
        echo "[$(date)] ❌ Experiment Finished with Errors (Code: $client_exit_code)."
    fi

    exec 1>&3 2>&4
    exec 3>&- 4>&-

    return "$client_exit_code"
}

# 计算总任务数
TOTAL_RUNS=$((${#DURATION_LIST[@]} * ${#QPS_LIST[@]}))
CURRENT_RUN=0

echo "========================================================"
echo "   🔄 Auto-Loop Benchmark Script"
echo "========================================================"
echo "Duration List: ${DURATION_LIST[*]} (mins)"
echo "QPS List:      ${QPS_LIST[*]}"
echo "Total Runs:    $TOTAL_RUNS"
echo "Run Root:      $RUN_ROOT"
echo "--------------------------------------------------------"
echo "Starting in 5 seconds..."
sleep 5

# 开始双重循环
for MIN in "${DURATION_LIST[@]}"; do
    for QPS in "${QPS_LIST[@]}"; do
        
        ((CURRENT_RUN++))
        run_one_experiment "$MIN" "$QPS" || true

        # 6. 冷却时间
        if [ $CURRENT_RUN -lt $TOTAL_RUNS ]; then
            echo "❄️  Cooling down for ${COOLDOWN_SECONDS}s before next run..."
            sleep $COOLDOWN_SECONDS
        fi

    done
done

echo ""
echo "========================================================"
echo "🎉 All $TOTAL_RUNS experiments completed!"
echo "📂 Outputs saved under: $RUN_ROOT"
echo "========================================================"
