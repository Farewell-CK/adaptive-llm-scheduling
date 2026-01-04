import argparse
import asyncio
import aiohttp
import time
import numpy as np
import os
import sys
import json
import uuid
from datetime import datetime

# ================= Configuration =================
# Targets
DEFAULT_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-unit-1"

# Load Settings
CONCURRENCY_LEVELS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
DURATION_PER_LEVEL = 10  # Seconds to run per level
WARMUP_DURATION = 3      # Seconds to warm up

# Payload
PROMPT_TEXT = "The quick brown fox jumps over the lazy dog. " * 50  # ~450-500 tokens
MAX_TOKENS = 4096

# =============================================

async def send_request(session, url, sem):
    # Add random prefix to bust KV cache
    random_content = f"ID-{uuid.uuid4()}: {PROMPT_TEXT}"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": random_content}],
        "max_tokens": MAX_TOKENS,
        "stream": True,  # Enable streaming for TTFT
        "stream_options": {"include_usage": True},
        "temperature": 0.7
    }
    
    async with sem:
        start_time = time.time()
        ttft = 0.0
        first_token_received = False
        gen_tokens = 0
        total_tokens = 0
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            
                            # Capture TTFT
                            if not first_token_received and len(data.get("choices", [])) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    ttft = time.time() - start_time
                                    first_token_received = True
                            
                            # Capture Usage (Final chunk)
                            if "usage" in data:
                                total_tokens = data["usage"].get("total_tokens", 0)
                                gen_tokens = data["usage"].get("completion_tokens", 0)
                                
                        except json.JSONDecodeError:
                            continue

                    end_time = time.time()
                    e2e_latency = end_time - start_time
                    
                    # Fallback if TTFT missed (e.g. fast completion)
                    if ttft == 0:
                        ttft = e2e_latency
                        
                    return {
                        "ttft": ttft,
                        "e2e_latency": e2e_latency,
                        "tokens": gen_tokens, 
                        "total_tokens": total_tokens,
                        "success": True
                    }
                else:
                    text = await resp.text()
                    print(f"Error {resp.status}: {text[:100]}")
                    return {"success": False}
        except Exception as e:
            print(f"Exception: {e}")
            return {"success": False}

async def benchmark_level(url, concurrency, duration):
    sem = asyncio.Semaphore(concurrency)
    stop_time = time.time() + duration + WARMUP_DURATION
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(worker(session, url, sem, stop_time, results))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
            
    return results

async def worker(session, url, sem, stop_time, results):
    while time.time() < stop_time:
        is_warmup = (stop_time - time.time()) > DURATION_PER_LEVEL
        
        res = await send_request(session, url, sem)
        
        if not is_warmup and res["success"]:
            results.append(res)

def save_results(results_dir, summary_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save JSON
    json_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
        
    # 2. Save Markdown
    md_path = os.path.join(results_dir, "README.md")
    with open(md_path, "w") as f:
        f.write("# vLLM Micro-Benchmark Results (Streaming)\n\n")
        f.write(f"**Date:** {timestamp}\n")
        f.write(f"**Model:** {MODEL_NAME}\n")
        f.write(f"**Max Tokens:** {MAX_TOKENS}\n\n")
        
        headers = [
            "Concurrency", "QPS", "TPS (Gen)", 
            "TTFT Avg(s)", "TTFT P99(s)", 
            "E2E Avg(s)", "E2E P99(s)", 
            "Total Tokens", "Duration(s)"
        ]
        
        # Markdown Table Header
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for row in summary_data:
            line = [
                str(row['concurrency']),
                f"{row['qps']:.2f}",
                f"{row['tps']:.1f}",
                f"{row['avg_ttft']:.3f}",
                f"{row['p99_ttft']:.3f}",
                f"{row['avg_e2e']:.2f}",
                f"{row['p99_e2e']:.2f}",
                str(row['total_gen_tokens']),
                f"{DURATION_PER_LEVEL}"
            ]
            f.write("| " + " | ".join(line) + " |\n")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Markdown report: {md_path}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    
    # Setup Output Directory
    if args.out_dir:
        results_dir = args.out_dir
    else:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results_final")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(base_dir, f"micro_bench_{ts}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"🚀 Starting Micro-Benchmark (Streaming)")
    print(f"   Target: {args.url}")
    print(f"   Levels: {CONCURRENCY_LEVELS}")
    print(f"   Output: {results_dir}\n")
    
    summary = []
    
    for conc in CONCURRENCY_LEVELS:
        print(f"Benchmarking Concurrency = {conc} ... ", end="", flush=True)
        results = await benchmark_level(args.url, conc, DURATION_PER_LEVEL)
        
        if not results:
            print("Failed (No successful requests)")
            continue
            
        # Stats
        ttfts = [r["ttft"] for r in results]
        e2es = [r["e2e_latency"] for r in results]
        total_gen_tokens = sum([r["tokens"] for r in results])
        
        tps = total_gen_tokens / DURATION_PER_LEVEL
        qps = len(results) / DURATION_PER_LEVEL
        
        avg_ttft = np.mean(ttfts)
        p99_ttft = np.percentile(ttfts, 99)
        
        avg_e2e = np.mean(e2es)
        p99_e2e = np.percentile(e2es, 99)
        
        print(f"TPS: {tps:.1f}, TTFT: {avg_ttft:.3f}s")
        
        summary.append({
            "concurrency": conc,
            "tps": tps,
            "qps": qps,
            "avg_ttft": avg_ttft,
            "p99_ttft": p99_ttft,
            "avg_e2e": avg_e2e,
            "p99_e2e": p99_e2e,
            "total_gen_tokens": total_gen_tokens,
            "sample_size": len(results)
        })
        
        # Cool down
        time.sleep(2)

    save_results(results_dir, summary)

if __name__ == "__main__":
    asyncio.run(main())