import argparse
import time
import csv
import datetime
import sys
import os
import requests
from urllib.parse import urlparse

# ================= 配置区域 =================

# 我们要抓取的核心指标
TARGET_METRICS = [
    # 核心状态
    "vllm:num_requests_waiting",      
    "vllm:num_requests_running",      
    "vllm:kv_cache_usage_perc",       # GPU block usage
    
    # 吞吐量
    "vllm:prompt_tokens_total",       
    "vllm:generation_tokens_total",   
    
    # 统计
    "vllm:request_success_total",     
    
    # 延迟
    "vllm:time_to_first_token_seconds_sum",   
    "vllm:time_to_first_token_seconds_count", 
    "vllm:e2e_request_latency_seconds_sum",   
    "vllm:e2e_request_latency_seconds_count"  
]

def fetch_metrics_text(url):
    """请求 vLLM 接口获取原始文本"""
    try:
        # 增加 timeout 防止卡死
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None

def parse_prometheus_text(text):
    """解析 Prometheus 格式"""
    data = {}
    if not text:
        return data

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        try:
            # vllm:num_requests_waiting{...} 0.0
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            raw_key_part = parts[0]
            value_str = parts[1]
            
            if '{' in raw_key_part:
                metric_name = raw_key_part.split('{')[0]
            else:
                metric_name = raw_key_part
            
            if metric_name in TARGET_METRICS:
                if 'inf' in value_str.lower() or 'nan' in value_str.lower():
                    val = 0.0
                else:
                    val = float(value_str)
                
                # 简单累加 (如果有多个 model tag，这里会合并)
                if metric_name in data:
                    data[metric_name] += val
                else:
                    data[metric_name] = val

        except Exception:
            continue
    return data

def main():
    parser = argparse.ArgumentParser()
    # 支持逗号分隔的多个 URL
    parser.add_argument("--urls", type=str, required=True, help="List of metrics URLs (e.g. http://h1:8000/metrics,http://h2:8000/metrics)")
    parser.add_argument("--output", type=str, default="vllm_metrics.csv")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=3600)
    args = parser.parse_args()

    url_list = [u.strip() for u in args.urls.split(",") if u.strip()]
    print(f"[*] Monitor Started.")
    print(f"    Targets: {len(url_list)} instances -> {url_list}")
    print(f"    CSV: {args.output}")
    print(f"    Duration: {args.duration}s")

    # CSV 表头: timestamp, elapsed, instance_url, [metrics...]
    headers = ["timestamp", "elapsed_seconds", "instance_url"] + TARGET_METRICS

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

                current_timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                # 遍历所有实例抓取
                for url in url_list:
                    text = fetch_metrics_text(url)
                    metrics_data = parse_prometheus_text(text)
                    
                    # 只有抓到数据才写
                    if metrics_data:
                        row = {
                            "timestamp": current_timestamp,
                            "elapsed_seconds": round(elapsed, 1),
                            "instance_url": url
                        }
                        for m in TARGET_METRICS:
                            row[m] = metrics_data.get(m, 0)
                        
                        writer.writerow(row)
                
                f.flush()

                # 精确 sleep
                process_cost = time.time() - now
                sleep_time = max(0, args.interval - process_cost)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[*] Interrupted by user.")

if __name__ == "__main__":
    main()