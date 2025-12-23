import argparse
import random
import json

def generate_verification_trace(output_file, duration_minutes=3, qps=2.0):
    """
    只生成 Mixed 阶段的负载：
    50% Analysis (超长输入，耗时，用来占连接)
    50% Chat (短输入，快速，用来检测延迟)
    """
    print(f"Generating verification trace: {duration_minutes} min @ {qps} QPS (Mixed Only)")
    
    requests = []
    current_time = 0
    total_seconds = duration_minutes * 60
    
    WORKLOADS = {
        "chat": {"input": (100, 500), "output": (200, 800)},
        "analysis": {"input": (6000, 7500), "output": (50, 200)}
    }
    
    BASE_TEXT = "The quick brown fox jumps over the lazy dog. " * 2000
    
    req_id = 0
    while current_time < total_seconds:
        interval = random.expovariate(qps)
        current_time += interval
        if current_time > total_seconds:
            break
            
        req_type = "analysis" if random.random() < 0.5 else "chat"
        cfg = WORKLOADS[req_type]
        
        prompt_len = random.randint(*cfg["input"])
        output_len = random.randint(*cfg["output"])
        
        char_len = prompt_len * 4
        prompt_text = (BASE_TEXT * (char_len // len(BASE_TEXT) + 1))[:char_len]
        
        requests.append({
            "id": req_id,
            "arrival_time": round(current_time, 3),
            "type": req_type,
            "prompt": prompt_text,
            "prompt_len": prompt_len,
            "max_tokens": output_len
        })
        req_id += 1
        
    with open(output_file, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(requests)} requests to {output_file}")

if __name__ == "__main__":
    generate_verification_trace("trace_verification.jsonl", duration_minutes=3)