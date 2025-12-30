import time
import asyncio
import logging
import httpx
import os
from enum import Enum
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# ================= 配置与阈值 (Theory Parameters) =================

# 1. 宏观控制参数 (Q-HAP)
HAP_HIGH_WATERMARK = 10   # 长队列积压超过 10 个，准备扩容
HAP_LOW_WATERMARK = 2     # 长队列积压低于 2 个，准备缩容
HAP_COOLDOWN = 5          # 状态切换冷却时间 (秒)，防止抖动

# 2. 微观控制参数 (RASP)
RASP_STEAL_COOLDOWN = 2.0 # 短节点必须空闲超过 2秒 才能被窃取
WORKER_CONCURRENCY_LIMIT = 8 # 每个 Worker (vLLM Instance) 允许的最大并发请求数

# 3. 静态定义 (根据你的 4 个 Unit)
WORKER_URLS = [
    "http://localhost:8001/v1/chat/completions", # Unit 1
    "http://localhost:8002/v1/chat/completions", # Unit 2
    "http://localhost:8003/v1/chat/completions", # Unit 3
    "http://localhost:8004/v1/chat/completions"  # Unit 4
]
STATIC_THRESHOLD = 3000   # 区分长短任务的 Token 阈值

# ==============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("AdaSplit")

app = FastAPI()
# 禁用环境代理变量（例如 socks5 代理导致需要 socksio），本路由只转发到本机 worker。
http_client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
    trust_env=False,
)

class TaskType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

# --- 1. Worker 抽象 (状态管理的最小单元) ---
class Worker:
    def __init__(self, worker_id, url):
        self.id = worker_id
        self.url = url
        self.current_role = TaskType.LONG # 默认为 Long，会被 Scheduler 修改
        self.active_requests = 0
        self.last_active_time = time.time() # 用于 RASP 计算空闲时长

    def inc_requests(self):
        self.active_requests += 1
    
    def dec_requests(self):
        self.active_requests = max(0, self.active_requests - 1)
        if self.active_requests == 0:
            self.last_active_time = time.time()

    def can_accept(self):
        return self.active_requests < WORKER_CONCURRENCY_LIMIT

    def get_idle_duration(self):
        if self.active_requests > 0:
            return 0
        return time.time() - self.last_active_time

    def __repr__(self):
        return f"[W{self.id}:{self.current_role.value[0]}:{self.active_requests}]"

# --- 2. 核心调度器 (The Brain) ---
class AdaSplitScheduler:
    def __init__(self):
        # 初始化 4 个 Worker
        self.workers = [Worker(i+1, url) for i, url in enumerate(WORKER_URLS)]
        
        # 初始状态：Balanced (2 Long : 2 Short)
        # 强制设定: 0,1 为 Long; 2,3 为 Short
        self.workers[0].current_role = TaskType.LONG
        self.workers[1].current_role = TaskType.LONG
        self.workers[2].current_role = TaskType.SHORT
        self.workers[3].current_role = TaskType.SHORT

        # 等待队列 (存放 Router 暂时处理不过来的请求)
        # 存储格式: (TaskType, asyncio.Future, request_body)
        self.queue_long = deque()
        self.queue_short = deque()
        
        # 状态控制
        self.last_rebalance_time = time.time()

    def get_partition_status(self):
        """返回当前的分组状态 (e.g., 3:1)"""
        n_long = sum(1 for w in self.workers if w.current_role == TaskType.LONG)
        n_short = sum(1 for w in self.workers if w.current_role == TaskType.SHORT)
        return n_long, n_short

    # === Part A: Q-HAP 宏观调度逻辑 (Background Loop) ===
    async def run_qhap_loop(self):
        logger.info("启动 Q-HAP 宏观调度监控...")
        while True:
            await asyncio.sleep(1) # 1秒运行一次
            
            now = time.time()
            if now - self.last_rebalance_time < HAP_COOLDOWN:
                continue

            q_long_size = len(self.queue_long)
            n_long, n_short = self.get_partition_status()
            
            # --- 扩容逻辑 (Scale Up) ---
            # 如果长队列积压严重，且 Short 组还有富余 (至少保留1个)
            if q_long_size > HAP_HIGH_WATERMARK and n_short > 1:
                # 找一个 Short Worker 变成 Long
                target = next((w for w in self.workers if w.current_role == TaskType.SHORT), None)
                if target:
                    target.current_role = TaskType.LONG
                    self.last_rebalance_time = now
                    logger.warning(f"🌊 [Q-HAP Trigger] 扩容! 长积压={q_long_size}. 切换 Worker {target.id} -> LONG. (当前 {n_long+1}:{n_short-1})")

            # --- 缩容逻辑 (Scale Down) ---
            # 如果长队列很空，且我们有多余的 Long Worker (恢复 Balanced 2:2)
            # 注意：这里我们设定 Default 是 2:2，所以只有 n_long > 2 时才缩容
            elif q_long_size < HAP_LOW_WATERMARK and n_long > 2:
                # 找一个 Long Worker 变成 Short (优先找编号大的)
                target = next((w for w in reversed(self.workers) if w.current_role == TaskType.LONG), None)
                if target:
                    target.current_role = TaskType.SHORT
                    self.last_rebalance_time = now
                    logger.info(f"🍃 [Q-HAP Trigger] 缩容. 长积压={q_long_size}. 切换 Worker {target.id} -> SHORT. (当前 {n_long-1}:{n_short+1})")

    # === Part B: RASP 微观分发逻辑 (Per Request) ===
    def try_get_worker(self, task_type: TaskType):
        """
        尝试获取一个可用 Worker。
        包含：本职工作分配 + RASP 窃取逻辑
        """
        
        # 1. 优先找【本职工作】且还有并发余量的 Worker
        # ------------------------------------------------
        candidates = [w for w in self.workers if w.current_role == task_type and w.can_accept()]
        if candidates:
            # 负载均衡：选择当前负载最低的那个
            return min(candidates, key=lambda x: x.active_requests)

        # 2. RASP 窃取逻辑 (Risk-Aware Stealing Policy)
        # ------------------------------------------------
        # 只有 Long 任务允许去偷 Short 节点 (激进策略)
        if task_type == TaskType.LONG:
            # 筛选可以被偷的 Short Worker
            short_q_empty = (len(self.queue_short) == 0)
            
            for w in self.workers:
                if w.current_role == TaskType.SHORT and w.can_accept():
                    # RASP 核心公式检查：短任务队列为空，且节点已空闲一段时间
                    if short_q_empty and w.get_idle_duration() > RASP_STEAL_COOLDOWN:
                        logger.info(f"🥷 [RASP Steal] Worker {w.id} (Short) 正在被窃取执行 Long 任务! (Load: {w.active_requests})")
                        return w
        
        return None # 没有可用资源

scheduler = AdaSplitScheduler()

# --- 辅助函数 ---
def estimate_token_count(messages):
    if not messages: return 0
    txt = "".join([str(m.get("content", "")) for m in messages])
    return len(txt) // 4

async def process_request(worker, body, request_obj):
    """实际执行转发，管理 Worker 忙/闲状态"""
    worker.inc_requests()
    try:
        # 构造请求
        req = http_client.build_request("POST", worker.url, json=body, timeout=None)
        r = await http_client.send(req, stream=True)
        return StreamingResponse(
            r.aiter_raw(), 
            status_code=r.status_code, 
            media_type=r.headers.get("content-type"),
            background=None
        )
    finally:
        # 只要请求响应开始返回（如果是 Streaming，这不代表结束，但在本 Router 架构中
        # 为了不阻塞后续请求进入 vLLM 内部队列，我们在发送后不久或结束后释放。
        # 实际上 vLLM 内部有更大的队列。
        # 为了更准确模拟并发控制，理想情况应该在 Streaming 结束时 dec_requests。
        
        # TODO: 如果要严格限制 vLLM 内部并发，需要解析 SSE 并在结束时回调。
        # 目前这里的逻辑是：请求发出并建立流连接后即视为占用一个 slot。
        # 由于 FastAPI StreamingResponse 的特性，我们无法简单地在此处 await 结束。
        
        worker.dec_requests() 
        # 重新触发一次调度，看队列里有没有等待的
        asyncio.create_task(dispatch_queue())

async def dispatch_queue():
    """消费等待队列中的任务"""
    # 1. 处理 Short 队列 (高优)
    while scheduler.queue_short:
        worker = scheduler.try_get_worker(TaskType.SHORT)
        if worker:
            task_future, body, req_obj = scheduler.queue_short.popleft()
            # 启动任务
            asyncio.create_task(run_task(worker, body, req_obj, task_future))
        else:
            break # 没资源了

    # 2. 处理 Long 队列
    while scheduler.queue_long:
        worker = scheduler.try_get_worker(TaskType.LONG)
        if worker:
            task_future, body, req_obj = scheduler.queue_long.popleft()
            asyncio.create_task(run_task(worker, body, req_obj, task_future))
        else:
            break

async def run_task(worker, body, req_obj, future):
    try:
        response = await process_request(worker, body, req_obj)
        future.set_result(response)
    except Exception as e:
        future.set_exception(e)

# --- FastAPI 接口 ---

@app.on_event("startup")
async def startup():
    asyncio.create_task(scheduler.run_qhap_loop())

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    token_len = estimate_token_count(body.get("messages", []))
    
    # 1. 分类
    task_type = TaskType.LONG if token_len > STATIC_THRESHOLD else TaskType.SHORT
    
    # 2. 尝试直接获取 Worker
    worker = scheduler.try_get_worker(task_type)
    
    if worker:
        # log 只有在 RASP 没触发时才打，不然 RASP 那里打过了
        # logger.info(f"Direct Dispatch {task_type.value} -> Worker {worker.id}")
        return await process_request(worker, body, request)
    
    else:
        # 3. 没资源，进入队列 (Queuing)
        # 这是一个简单的“挂起”逻辑
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        if task_type == TaskType.LONG:
            scheduler.queue_long.append((future, body, request))
            if len(scheduler.queue_long) % 5 == 0:
                logger.info(f"📥 Long Task Queued. Size: {len(scheduler.queue_long)}")
        else:
            scheduler.queue_short.append((future, body, request))
        
        # 等待调度器处理 Future
        return await future

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
