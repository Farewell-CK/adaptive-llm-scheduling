import json
import random
import time
import uuid
import argparse

# ================= 配置区域 =================

# 定义三种负载类型
WORKLOADS = {
    "chat": {
        "input_len_range": (100, 500),    # 短输入
        "output_len_range": (200, 800),   # 中长输出
        "weight": 0.8
    },
    "analysis": {
        "input_len_range": (6000, 7500),  # 超长输入 (接近8K)
        "output_len_range": (50, 200),    # 短输出
        "weight": 0.2
    }
}

# 为了不让文件太大，我们预生成一段很长的废话文本
# Llama tokenizer 大约 1个单词 = 1.3 token，这里简单模拟
BASE_TEXT = "The quick brown fox jumps over the lazy dog. " * 2000 
# 这段文本足够长，后面我们会通过切片来截取指定长度

# 修改前：随机 ID 在屁股后面
# random_noise = f"\n[Random ID: {uuid.uuid4()}]" 
# return text + random_noise

# 修改后：随机 ID 在最前面，甚至插在中间
def get_dummy_prompt(token_len):
    # 生成一个很长的随机头，彻底破坏前缀缓存
    random_header = f"Random Task ID {uuid.uuid4()}: " * 20 
    
    char_len = token_len * 4
    # 重新计算需要的剩余长度
    remaining_len = max(0, char_len - len(random_header))
    
    body = ""
    if remaining_len > 0:
        if remaining_len > len(BASE_TEXT):
            body = (BASE_TEXT * (remaining_len // len(BASE_TEXT) + 1))[:remaining_len]
        else:
            body = BASE_TEXT[:remaining_len]
            
    return random_header + body

def generate_trace(duration_minutes, qps, output_file):
    """
    生成符合泊松分布到达时间的请求流
    """
    print(f"Generating trace for {duration_minutes} minutes with QPS={qps}...")
    
    requests = []
    current_time = 0
    total_seconds = duration_minutes * 60
    
    # 阶段定义
    # 0-1/3 时间: Chat 阶段
    # 1/3-2/3 时间: Analysis 阶段 (恶魔阶段)
    # 2/3-End 时间: Mixed 混合阶段
    
    phase_1_end = total_seconds / 3
    phase_2_end = total_seconds * 2 / 3
    
    request_count = 0
    
    while current_time < total_seconds:
        # 1. 确定下一个请求的到达时间 (泊松过程)
        # random.expovariate(qps) 返回下一次到达的时间间隔
        inter_arrival_time = random.expovariate(qps)
        current_time += inter_arrival_time
        
        if current_time > total_seconds:
            break

        # 2. 确定当前阶段和请求类型
        req_type = ""
        prompt_len = 0
        output_len = 0
        
        if current_time < phase_1_end:
            # Phase 1: 纯 Chat
            req_type = "chat"
        elif current_time < phase_2_end:
            # Phase 2: 纯 Analysis (这是让 GPU 显存爆炸的阶段)
            req_type = "analysis"
        else:
            # Phase 3: Mixed (50% 概率)
            req_type = "chat" if random.random() < 0.5 else "analysis"
            
        # 3. 生成参数
        cfg = WORKLOADS[req_type]
        prompt_len = random.randint(*cfg["input_len_range"])
        output_len = random.randint(*cfg["output_len_range"])
        
        prompt_text = get_dummy_prompt(prompt_len)
        
        req_data = {
            "id": request_count,
            "arrival_time": round(current_time, 3), # 相对开始时间的秒数
            "type": req_type,
            "prompt": prompt_text,
            "prompt_len": prompt_len,     # 仅用于记录，发给 vLLM 时不需要
            "max_tokens": output_len      # 期望生成的长度
        }
        
        requests.append(req_data)
        request_count += 1
        
    # 写入文件
    with open(output_file, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
            
    print(f"Done! Generated {len(requests)} requests in {output_file}")
    print(f"Phase 1 (Chat): 0 - {phase_1_end/60:.1f} min")
    print(f"Phase 2 (Analysis): {phase_1_end/60:.1f} - {phase_2_end/60:.1f} min")
    print(f"Phase 3 (Mixed): {phase_2_end/60:.1f} - {duration_minutes} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=30, help="Duration of the trace in minutes")
    parser.add_argument("--qps", type=float, default=2.0, help="Average queries per second")
    parser.add_argument("--output", type=str, default="devil_trace.jsonl", help="Output file name")
    args = parser.parse_args()
    
    generate_trace(args.minutes, args.qps, args.output)