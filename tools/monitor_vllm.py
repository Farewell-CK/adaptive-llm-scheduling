import argparse
import time
import csv
import datetime
import sys
import os
from urllib.request import Request, urlopen

# ================= 配置区域 =================
# vLLM 默认 metrics 端口
DEFAULT_METRICS_URL = "http://localhost:8000/metrics"

# 我们要抓取的核心指标 (与你提供的 curl 输出严格对应)
# 这里的 Key 是去掉 {...} 标签后的名字
TARGET_METRICS = [
    # --- 核心状态 (Figure 1 & Figure 5 素材) ---
    "vllm:num_requests_waiting",      # 当前排队数 (最重要!)
    "vllm:num_requests_running",      # 当前运行数
    "vllm:kv_cache_usage_perc",       # 显存 KV Cache 占用率 (0.0 - 1.0)
    
    # --- 吞吐量 (计算 TPS 用) ---
    "vllm:prompt_tokens_total",       # 累计处理的 Input Token
    "vllm:generation_tokens_total",   # 累计生成的 Output Token
    
    # --- 统计计数 ---
    "vllm:request_success_total",     # 成功请求总数 (FINISHED)
    
    # --- 延迟相关 (累计值，用于事后计算平均值) ---
    "vllm:time_to_first_token_seconds_sum",   # TTFT 总时间
    "vllm:time_to_first_token_seconds_count", # TTFT 总次数
    "vllm:e2e_request_latency_seconds_sum",   # E2E 延迟总时间
    "vllm:e2e_request_latency_seconds_count"  # E2E 延迟总次数
]

def fetch_metrics_text(url):
    """请求 vLLM 接口获取原始文本"""
    try:
        req = Request(url, headers={"User-Agent": "monitor_vllm/1.0"})
        with urlopen(req, timeout=2) as resp:
            body = resp.read()
        return body.decode("utf-8", errors="replace")
    except Exception:
        return None

def parse_prometheus_text(text):
    """解析 Prometheus 格式，提取 TARGET_METRICS 里的值"""
    data = {}
    if not text:
        return data

    for line in text.splitlines():
        line = line.strip()
        # 跳过注释和空行
        if not line or line.startswith("#"):
            continue
            
        try:
            # 格式示例: vllm:num_requests_waiting{...} 0.0
            # 1. 分割 Key 和 Value (以最后一个空格为界)
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            raw_key_part = parts[0]
            value_str = parts[1]
            
            # 2. 去掉 Key 里的标签 {...}
            # vllm:num_requests_waiting{engine="0"...} -> vllm:num_requests_waiting
            if '{' in raw_key_part:
                metric_name = raw_key_part.split('{')[0]
            else:
                metric_name = raw_key_part
            
            # 3. 如果是我们关心的指标，存入字典
            if metric_name in TARGET_METRICS:
                # 注意：如果有多个模型，这里会发生覆盖，但在单体测试中没问题
                # 遇到 NaN 或 Inf 处理一下
                if 'inf' in value_str.lower() or 'nan' in value_str.lower():
                    val = 0.0
                else:
                    val = float(value_str)
                
                # 有些指标可能有多个 entry (比如 success_total 有 finished_reason="stop/length")
                # 我们这里做简单的累加 (Sum)
                if metric_name in data:
                    data[metric_name] += val
                else:
                    data[metric_name] = val

        except Exception:
            continue
            
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=DEFAULT_METRICS_URL)
    parser.add_argument("--output", type=str, default="vllm_metrics.csv")
    parser.add_argument("--interval", type=float, default=1.0) # 采样间隔
    parser.add_argument("--duration", type=int, default=3600)  # 监控时长
    args = parser.parse_args()

    print(f"[*] Monitor Started.")
    print(f"    URL: {args.url}")
    print(f"    CSV: {args.output}")
    print(f"    Stop after: {args.duration}s")

    # CSV 表头
    headers = ["timestamp", "elapsed_seconds"] + TARGET_METRICS

    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    start_time = time.time()

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        try:
            while True:
                now = time.time()
                elapsed = now - start_time
                
                if elapsed > args.duration:
                    print("[*] Duration reached. Exiting.")
                    break

                # 1. 抓取
                text = fetch_metrics_text(args.url)
                
                # 2. 解析
                metrics_data = parse_prometheus_text(text)
                
                # 3. 组装行数据
                row = {
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "elapsed_seconds": round(elapsed, 1)
                }
                
                # 填充指标 (没抓到的填 0)
                for m in TARGET_METRICS:
                    row[m] = metrics_data.get(m, 0)
                
                # 4. 写入
                writer.writerow(row)
                f.flush() # 立即落盘

                # 5. 等待
                # 扣除掉处理时间，保持精确间隔
                process_cost = time.time() - now
                sleep_time = max(0, args.interval - process_cost)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[*] Interrupted by user.")

if __name__ == "__main__":
    main()
