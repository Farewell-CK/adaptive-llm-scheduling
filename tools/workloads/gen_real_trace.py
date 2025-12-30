import argparse
import json
import random
import time
import math
import sys
import uuid
import os

# ================= Configuration =================

# 1. 阶段定义 (Duration in seconds, QPS)
PHASES = [
    {"name": "Chat_Burst",     "duration": 600, "qps": 4.0, "source": "sharegpt"}, # 0-10 min
    {"name": "Analysis_Burst", "duration": 600, "qps": 0.5, "source": "longbench"}, # 10-20 min
    {"name": "Mixed_Steady",   "duration": 600, "qps": 2.0, "source": "mixed"}     # 20-30 min
]

# Paths
DATASETS_ROOT = "datasets"
SHAREGPT_PATH = os.path.join(DATASETS_ROOT, "sharegpt", "ShareGPT_V4.3_unfiltered_cleaned_split.json")
LONGBENCH_PATH = os.path.join(DATASETS_ROOT, "longbench", "data", "narrativeqa.jsonl")
TRACES_DIR = os.path.join(DATASETS_ROOT, "traces")

# Mock Data Fallback
BASE_TEXT = "The quick brown fox jumps over the lazy dog. " * 2000 

def get_dummy_prompt(token_len):
    random_header = f"Random Task ID {uuid.uuid4()}: " * 20 
    char_len = token_len * 4
    remaining_len = max(0, char_len - len(random_header))
    body = ""
    if remaining_len > 0:
        if remaining_len > len(BASE_TEXT):
            body = (BASE_TEXT * (remaining_len // len(BASE_TEXT) + 1))[:remaining_len]
        else:
            body = BASE_TEXT[:remaining_len]
    return random_header + body

# ================= Data Loading =================

def load_local_data():
    """
    Load data from local datasets/ directory.
    """
    sharegpt_prompts = []
    longbench_prompts = []

    # 1. Load ShareGPT
    if os.path.exists(SHAREGPT_PATH):
        print(f"📖 Loading local ShareGPT from {SHAREGPT_PATH}...")
        try:
            with open(SHAREGPT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # ShareGPT format is a list of dicts with 'conversations'
                for item in data:
                    convs = item.get("conversations", [])
                    if not convs: continue
                    prompt = ""
                    for turn in convs:
                        if turn.get("from") == "human":
                            prompt = turn.get("value", "")
                            break
                    if prompt and len(prompt) < 2000:
                        sharegpt_prompts.append(prompt)
                    if len(sharegpt_prompts) >= 5000: break
            print(f"   -> Successfully loaded {len(sharegpt_prompts)} ShareGPT prompts.")
        except Exception as e:
            print(f"⚠️  Error loading ShareGPT: {e}")
    else:
        print(f"⚠️  ShareGPT file not found at {SHAREGPT_PATH}")

    # 2. Load LongBench (narrativeqa)
    if os.path.exists(LONGBENCH_PATH):
        print(f"📖 Loading local LongBench from {LONGBENCH_PATH}...")
        try:
            with open(LONGBENCH_PATH, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    item = json.loads(line)
                    context = item.get("context", "")
                    inp = item.get("input", "")
                    full_prompt = f"Context: {context}\n\nQuestion: {inp}"
                    # Truncate to ~8k tokens (approx 32k chars)
                    if len(full_prompt) > 32000:
                        full_prompt = full_prompt[:32000]
                    if len(full_prompt) > 5000:
                        longbench_prompts.append(full_prompt)
                        count += 1
                    if count >= 1000: break
            print(f"   -> Successfully loaded {len(longbench_prompts)} LongBench prompts.")
        except Exception as e:
            print(f"⚠️  Error loading LongBench: {e}")
    else:
        print(f"⚠️  LongBench file not found at {LONGBENCH_PATH}")

    return sharegpt_prompts, longbench_prompts

# ================= Trace Generation =================

def generate_tidal_trace(output_file, duration_minutes=None, mock=False):
    # Load Data
    if not mock:
        sharegpt_data, longbench_data = load_local_data()
    else:
        sharegpt_data, longbench_data = [], []

    # Check if data loaded successfully
    use_mock_short = (not sharegpt_data)
    use_mock_long = (not longbench_data)
    
    if use_mock_short: print("⚠️  Using Mock data for SHORT tasks.")
    if use_mock_long: print("⚠️  Using Mock data for LONG tasks.")

    requests = []
    req_id = 0
    
    actual_phases = []
    if duration_minutes:
        total_def_duration = sum(p["duration"] for p in PHASES)
        scale = (duration_minutes * 60) / total_def_duration
        for p in PHASES:
            new_p = p.copy()
            new_p["duration"] = p["duration"] * scale
            actual_phases.append(new_p)
        print(f"🌊 Generating Scaled Tidal Trace: {duration_minutes} min (Scale: {scale:.2f})")
    else:
        actual_phases = PHASES
        total_sec = sum(p["duration"] for p in PHASES)
        print(f"🌊 Generating Fixed Tidal Trace: {total_sec/60:.1f} min")

    current_time = 0
    for phase_idx, phase_cfg in enumerate(actual_phases):
        phase_start_time = current_time
        phase_end_time = phase_start_time + phase_cfg["duration"]
        qps = phase_cfg["qps"]
        source = phase_cfg["source"]
        
        print(f"   -> Processing Phase: {phase_cfg['name']} (QPS: {qps}, Duration: {phase_cfg['duration']:.1f}s)")
        
        while current_time < phase_end_time:
            interval = random.expovariate(qps)
            current_time += interval
            if current_time >= phase_end_time:
                break
            
            # Determine if long task
            is_long = False
            if source == "sharegpt":
                is_long = False
            elif source == "longbench":
                is_long = True
            else: # mixed
                is_long = (random.random() < 0.5)
                
            # Generate Content
            prompt = ""
            max_tokens = 256
            
            if is_long:
                if not use_mock_long:
                    prompt = random.choice(longbench_data)
                else:
                    prompt = get_dummy_prompt(7000)
                max_tokens = random.randint(50, 200)
                task_type = "analysis"
            else:
                if not use_mock_short:
                    prompt = random.choice(sharegpt_data)
                else:
                    prompt = get_dummy_prompt(300)
                max_tokens = random.randint(200, 800)
                task_type = "chat"
                
            requests.append({
                "id": req_id,
                "arrival_time": round(current_time, 3),
                "type": task_type,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "phase_name": phase_cfg["name"]
            })
            req_id += 1
            
    # Write to file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ Generated {len(requests)} requests in {output_file}")
    for p in actual_phases:
        count = len([r for r in requests if r['phase_name']==p['name']])
        print(f"   - {p['name']}: {count} reqs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=None, help="Total duration in minutes (scales PHASES)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--mock", action="store_true", help="Force use mock data (no download)")
    args = parser.parse_args()
    
    # Default output path
    output = args.output
    if output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = os.path.join(TRACES_DIR, f"tidal_trace_{ts}.jsonl")

    generate_tidal_trace(output, args.minutes, args.mock)
