# 文件路径: /workspace/tools/router_smart.py

import time
import asyncio
import logging
import httpx
import os
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
import uvicorn

# ================= 配置区域 =================

# 两个 vLLM 实例地址 (对应 Baseline 2 的静态配置)
# Instance A (Prefill/Long): 专门处理长文本 (GPU 0-3)
URL_WORKER_LONG = os.getenv("URL_WORKER_LONG", "http://localhost:8001/v1/chat/completions")
# Instance B (Decode/Short): 专门处理短文本 (GPU 4-7)
URL_WORKER_SHORT = os.getenv("URL_WORKER_SHORT", "http://localhost:8002/v1/chat/completions")

# 静态路由阈值 (超过这个长度去 Instance A)
try:
    STATIC_THRESHOLD = int(os.getenv("STATIC_THRESHOLD", "3000"))
except ValueError:
    STATIC_THRESHOLD = 3000

# 监控窗口大小 (秒)
try:
    MONITOR_WINDOW = int(os.getenv("MONITOR_WINDOW", "10"))
except ValueError:
    MONITOR_WINDOW = 10

# ===========================================

# 配置日志格式，方便观察
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AdaSplit-Router")

app = FastAPI()

# 全局 HTTP 客户端 (永不超时，保持长连接)
# 注：某些环境会设置 ALL_PROXY/HTTP(S)_PROXY 为 socks5，httpx 会尝试走 SOCKS 代理并要求 socksio；
# 对本地 localhost 路由没有意义，因此禁用读取环境代理变量以避免依赖问题。
# 增加连接池大小，避免在高并发长耗时任务下（如 Analysis）耗尽默认的 100 个连接，导致短任务被阻塞
limits = httpx.Limits(max_keepalive_connections=50, max_connections=1000)
http_client = httpx.AsyncClient(timeout=None, trust_env=False, limits=limits)

# --- 核心模块: 流量监控 ---
class TrafficMonitor:
    def __init__(self, window_seconds):
        self.window = window_seconds
        # 队列存储元组: (timestamp, token_len)
        self.history = deque()
        self.total_reqs = 0

    def record_request(self, token_len):
        """记录一个新的请求"""
        now = time.time()
        self.history.append((now, token_len))
        self.total_reqs += 1
        self._cleanup(now)

    def _cleanup(self, now):
        """清除滑动窗口之外的过期数据"""
        while self.history and (now - self.history[0][0] > self.window):
            self.history.popleft()

    def get_stats(self):
        """计算当前窗口内的流量特征"""
        now = time.time()
        self._cleanup(now)
        
        count = len(self.history)
        if count == 0:
            return 0.0, 0 # QPS, AvgLen
        
        total_len = sum(x[1] for x in self.history)
        avg_len = total_len / count
        # QPS = 窗口内的请求数 / 窗口时间 (或者实际流逝时间，这里简单用窗口时间)
        qps = count / self.window
        
        return qps, avg_len

# 初始化监控器
monitor = TrafficMonitor(window_seconds=MONITOR_WINDOW)

# --- 辅助函数 ---

def estimate_token_count(messages: list) -> int:
    """
    简单估算 Token 数 (字符数 / 4)。
    为了性能，我们在路由层不使用 Tokenizer，这种估算误差在可接受范围内。
    """
    if not messages or not isinstance(messages, list):
        return 0
    
    txt = ""
    for m in messages:
        # get 可能会返回 None，转成 str 并在为空时给空字符串
        content = m.get("content")
        if content:
            txt += str(content)
            
    return len(txt) // 4

async def forward_request(target_url: str, request: Request, body: dict):
    """通用转发逻辑: Client -> Router -> vLLM"""
    try:
        # 构建转发请求
        req = http_client.build_request(
            request.method,
            target_url,
            json=body,
            timeout=None # 关键: 让 vLLM 慢慢算，Router 不主动断开
        )
        # 发送请求 (Stream 模式)
        r = await http_client.send(req, stream=True)
        
        # 将 vLLM 的流式响应透传回 Client
        return StreamingResponse(
            r.aiter_raw(),
            status_code=r.status_code,
            media_type=r.headers.get("content-type"),
            background=BackgroundTask(r.aclose),
        )
    except Exception as e:
        logger.error(f"Forward failed to {target_url}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 后台任务：打印监控看板 ---
async def print_stats_loop():
    """每 2 秒打印一次当前的流量状态"""
    logger.info("Traffic Monitor Started...")
    while True:
        try:
            qps, avg_len = monitor.get_stats()
            # 只有在有流量时才打印，避免刷屏
            if qps > 0.1:
                # 判断当前是长文本主导还是短文本主导
                status = "🐘 LONG-HEAVY" if avg_len > STATIC_THRESHOLD else "🐇 SHORT-HEAVY"
                logger.info(f"📊 [Monitor] QPS: {qps:.1f} | Avg Len: {avg_len:.0f} | State: {status}")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    # 启动后台监控打印任务
    asyncio.create_task(print_stats_loop())

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        
        # 1. 估算长度并记录到监控器
        input_len = estimate_token_count(body.get("messages", []))
        monitor.record_request(input_len) 
        
        # 2. 路由决策 (Baseline 2: 静态分离)
        # ==========================================
        if input_len > STATIC_THRESHOLD:
            target = URL_WORKER_LONG
            tag = "[LONG -> A]"
        else:
            target = URL_WORKER_SHORT
            tag = "[SHORT -> B]"
        # ==========================================
        
        # 打印决策日志 (可选，生产环境可关掉)
        logger.info(f"{tag} Len={input_len} -> {target}")
        
        # 3. 执行转发
        return await forward_request(target, request, body)
        
    except Exception as e:
        logger.error(f"Router Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Router Internal Error"})

if __name__ == "__main__":
    # Router 运行在 5000 端口，与 vLLM (8000/8001/8002) 区分开
    host = os.getenv("ROUTER_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("ROUTER_PORT", "5000"))
    except ValueError:
        port = 5000
    uvicorn.run(app, host=host, port=port)
