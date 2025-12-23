import argparse
import subprocess
import time
import os
import shutil
import sys
import requests
import signal
import json

# ================= 配置区域 =================

# 项目根目录 (假设脚本在 workspace/tools/ 下)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
START_DIR = os.path.join(WORKSPACE_DIR, "start")
RESULTS_ROOT = os.path.join(WORKSPACE_DIR, "results_final")

# 实验定义
EXPERIMENTS = {
    1: {
        "name": "Monolithic_Baseline",
        "start_script": "start_vllm_baseline.sh",
        "target_url": "http://localhost:8000/v1/chat/completions",
        "metric_urls": "http://localhost:8000/metrics",
        "check_port": 8000,
        "desc": "Single vLLM instance (8 GPUs)"
    },
    2: {
        "name": "Static_Partitioning",
        "start_script": "start_vllm_dual.sh",
        "target_url": "http://localhost:5000/v1/chat/completions",
        "metric_urls": "http://localhost:8001/metrics,http://localhost:8002/metrics",
        "check_port": 5000, # Check Router
        "extra_env": {"ROUTER_IMPL": "static"}, # 强制使用静态路由
        "desc": "Two instances (Long/Short) with Static Router"
    },
    3: {
        "name": "AdaSplit_Dynamic",
        "start_script": "start_vllm_4_instances.sh",
        "target_url": "http://localhost:5000/v1/chat/completions",
        "metric_urls": "http://localhost:8001/metrics,http://localhost:8002/metrics,http://localhost:8003/metrics,http://localhost:8004/metrics",
        "check_port": 5000, # Check Router
        "extra_env": {"ROUTER_IMPL": "dynamic"}, # 强制使用 AdaSplit Router
        "desc": "Four instances with AdaSplit Dynamic Router"
    }
}

# ================= 辅助函数 =================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_trace(duration_min, qps, output_path):
    print(f"   [Gen] Generating trace: Duration={duration_min}m, QPS={qps} -> {output_path}")
    cmd = [
        "python3", os.path.join(SCRIPT_DIR, "workloads", "workload_gen.py"),
        "--minutes", str(duration_min),
        "--qps", str(qps),
        "--output", output_path
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

def wait_for_health(port, timeout=600):
    """轮询直到服务端口返回 200 OK，或者超时"""
    print(f"   [Health] Waiting for service at port {port} (Timeout: {timeout}s)...")
    start_wait = time.time()
    url = f"http://localhost:{port}/health" # vLLM 标准健康检查接口
    # 如果是 Router (5000)，它可能没有 /health，我们检查 /docs 或者直接检查 connect
    if port == 5000:
        url = f"http://localhost:{port}/docs" 

    while True:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"   [Health] Service is UP!")
                return True
        except requests.RequestException:
            pass

        if time.time() - start_wait > timeout:
            print(f"   [Health] Timeout waiting for port {port}.")
            return False
        
        time.sleep(5)
        sys.stdout.write(".")
        sys.stdout.flush()

def stop_environment(script_name):
    """
    主要靠 pkill 来清理。
    注意：这比较暴力，会杀掉所有 python vllm 进程。
    生产环境需谨慎，但在独占的实验机上是最高效的。
    """
    print("   [Stop] Cleaning up processes...")
    # 1. 尝试调用脚本自带的 cleanup (如果脚本运行在前台很难，所以直接杀进程)
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "router_static.py"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "router_dynamic.py"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "router_smart.py"], stderr=subprocess.DEVNULL)
    time.sleep(5) # 等待资源释放

def run_single_experiment(exp_id, qps, duration_min):
    cfg = EXPERIMENTS[exp_id]
    exp_name = cfg["name"]
    
    # 1. 准备结果目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_ROOT, f"Exp{exp_id}_{exp_name}", f"QPS_{qps}_Min_{duration_min}_{timestamp}")
    ensure_dir(result_dir)
    
    print(f"\n{'='*60}")
    print(f"🚀 Running {cfg['desc']}")
    print(f"   Config: QPS={qps}, Duration={duration_min}m")
    print(f"   Output: {result_dir}")
    print(f"{'='*60}\n")

    # 2. 生成 Trace
    trace_file = os.path.join(result_dir, "trace.jsonl")
    generate_trace(duration_min, qps, trace_file)

    # 3. 启动环境
    print(f"   [Boot] Starting environment via {cfg['start_script']}...")
    log_dir = os.path.join(result_dir, "logs")
    ensure_dir(log_dir)
    
    env_vars = os.environ.copy()
    env_vars["LOG_DIR"] = log_dir
    # 注入特定的环境变量 (如 ROUTER_IMPL)
    if "extra_env" in cfg:
        env_vars.update(cfg["extra_env"])
        
    start_cmd = ["bash", os.path.join(START_DIR, cfg["start_script"])]
    
    # 后台启动脚本
    # 注意：我们的 start 脚本设计为前台 wait，所以这里 popen 后它会一直运行
    # 我们需要设 DETACH=0 (默认) 让它 block 住，但我们在 python 里用 Popen 是非阻塞的
    proc = subprocess.Popen(start_cmd, cwd=START_DIR, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # 4. 等待服务就绪
        if not wait_for_health(cfg["check_port"]):
            print("❌ Setup failed. Aborting.")
            return

        # 5. 启动监控 (Monitor)
        print(f"   [Mon] Starting metrics monitor...")
        monitor_csv = os.path.join(result_dir, "vllm_metrics.csv")
        monitor_cmd = [
            "python3", os.path.join(SCRIPT_DIR, "analysis", "monitor_vllm.py"),
            "--urls", cfg["metric_urls"],
            "--output", monitor_csv,
            "--duration", str(int(duration_min * 60) + 60) # 监控时长比压测稍长一点
        ]
        # 后台运行
        monitor_proc = subprocess.Popen(monitor_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 6. 运行 Benchmark Client
        print(f"   [Bench] Starting client...")
        client_output_csv = os.path.join(result_dir, "metrics.csv")
        
        bench_cmd = [
            "python3", os.path.join(SCRIPT_DIR, "workloads", "benchmark_client.py"),
            "--trace", trace_file,
            "--output", client_output_csv,
            "--url", cfg["target_url"],
            "--model", "llama-3.3-70b" # 这里假设模型名，如果变动需要传参
        ]
        
        # 实时打印 Client 输出
        subprocess.check_call(bench_cmd)
        
        print("   [Bench] Finished.")

    except KeyboardInterrupt:
        print("\n   [!] Interrupted by user.")
    except Exception as e:
        print(f"   [!] Error: {e}")
    finally:
        # 7. 清理环境
        if 'monitor_proc' in locals():
            monitor_proc.terminate()
        
        proc.terminate()
        stop_environment(cfg["start_script"])
        
        # 7. 归档额外信息
        # 比如把当时的配置写进去
        with open(os.path.join(result_dir, "meta.json"), "w") as f:
            json.dump({
                "exp_id": exp_id,
                "qps": qps,
                "duration": duration_min,
                "config": str(cfg)
            }, f, indent=2)
            
    print(f"✅ Experiment Done. Results saved to {result_dir}")

# ================= 主入口 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaSplit Experiment Suite Runner")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True, help="Experiment ID (1=Base, 2=Static, 3=AdaSplit)")
    parser.add_argument("--qps", type=str, default="1.0,2.0,3.0", help="Comma separated QPS list")
    parser.add_argument("--min", type=str, default="30", help="Comma separated Duration list (minutes)")
    
    args = parser.parse_args()
    
    qps_list = [float(x) for x in args.qps.split(",")]
    min_list = [int(x) for x in args.min.split(",")]
    
    print(f"Plan: Run Exp {args.exp} for QPS={qps_list} and Duration={min_list}")
    
    for m in min_list:
        for q in qps_list:
            run_single_experiment(args.exp, q, m)
            # 实验间隔休息，让 GPU 冷却一下
            time.sleep(10)
