import argparse
import asyncio
import json
import time
import aiohttp
import csv
import sys
import os

# ================= 配置 =================
DEFAULT_API_URL = "http://localhost:5000/v1/chat/completions"  # 注意：流式通常推荐用 chat 接口，兼容性更好
DEFAULT_MODEL_NAME = "llama-3.3-70b"

async def send_request(session, api_url, model_name, req_data, csv_writer, file_handle):
    """发送流式请求并计算 TTFT 和 E2E 延迟"""
    prompt = req_data["prompt"]
    max_tokens = req_data["max_tokens"]
    req_id = req_data["id"]
    req_type = req_data["type"]
    
    # 构建请求体
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,  # <--- 开启流式
        "stream_options": {"include_usage": True} # <--- 让 vLLM 在最后返回 Token 统计
    }
    
    start_time = time.time()
    ttft = 0.0
    end_time = 0.0
    
    p_tokens = 0
    c_tokens = 0
    
    result_row = {
        "id": req_id,
        "type": req_type,
        "submit_time": start_time,
        "ttft": 0,
        "e2e_latency": 0,
        "status": "pending",
        "prompt_tokens": 0,
        "completion_tokens": 0
    }

    try:
        timeout = aiohttp.ClientTimeout(total=None)  # <--- 关键：设置为 None，永不超时
        async with session.post(api_url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                first_token_received = False
                
                # 逐行读取流式响应
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:] # 去掉 "data: " 前缀
                    
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(data_str)
                        
                        # 1. 捕捉 TTFT (收到第一个非空内容块的时间)
                        # 有些块可能是空的或只有 usage，需要判断 choices
                        if not first_token_received and len(data.get("choices", [])) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                ttft = time.time() - start_time
                                first_token_received = True
                        
                        # 2. 捕捉 Token 统计 (通常在最后一个包)
                        if "usage" in data:
                            p_tokens = data["usage"].get("prompt_tokens", 0)
                            c_tokens = data["usage"].get("completion_tokens", 0)
                            
                    except json.JSONDecodeError:
                        continue

                # 循环结束，记录总时间
                end_time = time.time()
                total_latency = end_time - start_time
                
                # 如果没捕捉到 TTFT (比如直接返回了空)，则 TTFT = Total
                if ttft == 0.0:
                    ttft = total_latency

                result_row.update({
                    "ttft": ttft,
                    "e2e_latency": total_latency,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "status": "success"
                })
            else:
                print(f"[Error] ID={req_id} Status={response.status}")
                result_row["status"] = "error"
                
    except Exception as e:
        print(f"[Exception] ID={req_id} Error={e}")
        result_row["status"] = "exception"
    
    # 立即写入 CSV 并 Flush
    csv_writer.writerow(result_row)
    file_handle.flush()

async def benchmark(trace_file, output_file, api_url, model_name):
    print(f"Loading trace from {trace_file}...")
    with open(trace_file, 'r') as f:
        requests = [json.loads(line) for line in f]
    
    print(f"Loaded {len(requests)} requests. Starting STREAMING benchmark...")
    print(f"API URL: {api_url}")
    print(f"Model: {model_name}")
    print(f"Writing results incrementally to {output_file}...")
    
    with open(output_file, 'w', newline='') as f_out:
        fieldnames = ["id", "type", "submit_time", "ttft", "e2e_latency", "status", "prompt_tokens", "completion_tokens"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        f_out.flush()
        
        # 增加 Client 端连接池限制，防止客户端自我阻塞
        connector = aiohttp.TCPConnector(limit=2000)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            benchmark_start_time = time.time()
            
            for req in requests:
                target_time = benchmark_start_time + req["arrival_time"]
                current_time = time.time()
                wait_time = target_time - current_time
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                task = asyncio.create_task(send_request(session, api_url, model_name, req, writer, f_out))
                tasks.append(task)
                
                if req["id"] % 10 == 0:
                    sys.stdout.write(f"\r[Running] Scheduled request {req['id']}/{len(requests)}")
                    sys.stdout.flush()
            
            print("\nAll requests scheduled. Waiting for completion...")
            await asyncio.gather(*tasks)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, default="devil_trace.jsonl")
    parser.add_argument("--output", type=str, default="baseline_result_stream.csv")
    parser.add_argument("--url", type=str, default=None, help="OpenAI-compatible chat completions endpoint")
    parser.add_argument("--model", type=str, default=None, help="Model name in request payload")
    args = parser.parse_args()
    
    api_url = args.url or os.getenv("API_URL") or DEFAULT_API_URL
    model_name = args.model or os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME
    asyncio.run(benchmark(args.trace, args.output, api_url, model_name))
