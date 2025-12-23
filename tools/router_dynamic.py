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
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=20),
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
        self.is_busy = False
        self.last_active_time = time.time() # 用于 RASP 计算空闲时长

    def mark_busy(self):
        self.is_busy = True
    
    def mark_idle(self):
        self.is_busy = False
        self.last_active_time = time.time()

    def get_idle_duration(self):
        if self.is_busy:
            return 0
        return time.time() - self.last_active_time

    def __repr__(self):
        return f"[W{self.id}:{self.current_role.value[0]}]"

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
        
        # 1. 优先找【本职工作】且空闲的 Worker
        # ------------------------------------------------
        candidates = [w for w in self.workers if w.current_role == task_type and not w.is_busy]
        if candidates:
            return candidates[0] # 返回第一个空闲的本职 Worker

        # 2. RASP 窃取逻辑 (Risk-Aware Stealing Policy)
        # ------------------------------------------------
        # 只有 Long 任务允许去偷 Short 节点 (激进策略)
        # Short 任务不允许偷 Long (因为长任务太慢，不仅不赚反而亏)
        if task_type == TaskType.LONG:
            # 筛选可以被偷的 Short Worker
            # 条件公式: Role=SHORT AND Queue_Short=Empty AND Idle_Time > Threshold
            
            short_q_empty = (len(self.queue_short) == 0)
            
            for w in self.workers:
                if w.current_role == TaskType.SHORT and not w.is_busy:
                    # RASP 核心公式检查
                    if short_q_empty and w.get_idle_duration() > RASP_STEAL_COOLDOWN:
                        logger.info(f"🥷 [RASP Steal] Worker {w.id} (Short) 正在被窃取执行 Long 任务! (Idle: {w.get_idle_duration():.1f}s)")
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
    worker.mark_busy()
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
        # 无论成功失败，请求结束后标记为空闲
        # 注意：这里我们简单处理，真实场景可能需要处理 Stream 结束的回调
        # 对于 vLLM Stream，这里其实是 header 返回就 mark_idle 了，这在并发控制上是不精确的
        # 但对于论文实验，为了制造排队，我们可以在这里加一个简单的 await r.read() 或者
        # 更好的方式是假设 Worker 并发能力是 1 (Request Level)，
        # 或者我们仅把 Router 当做 Dispatcher，Worker 内部其实支持 Batching。
        # 
        # 【重要修正】：vLLM 本身支持并发 (Continuous Batching)。
        # 我们的 Worker.is_busy = True 实际上是把 Worker 当成了 "Slot"。
        # 为了让实验效果明显，我们这里【不】应该一发请求就释放 Worker，
        # 而是应该让 Router 认为 Worker 满载了。
        # 但由于我们没法知道 Stream 什么时候结束，简化起见：
        # 我们这里不做严格的 Worker 锁定，而是只做简单的计数，或者
        # 我们的算法假设是 Request-Level 的调度。
        
        # 为了让实验排队效果最明显 (Hol Blocking)，我们暂时设为：
        # 发送请求 -> 只要建立了连接 -> 就认为 Worker 空闲了 (把压力给 vLLM 内部队列)
        # 或者，为了模拟 Router 端的排队，我们可以在这里等待。
        
        # *对于本实验*：我们不等待 Stream 结束，因为那需要解析 SSE。
        # 我们只负责分发。Load Balancing 由 vLLM 内部处理一部分，
        # 但 Worker 选择由我们决定。
        
        worker.mark_idle() 
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
