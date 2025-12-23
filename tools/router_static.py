import argparse
import asyncio
import logging
import time
import httpx
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask

# ================= 配置区域 =================
# 以后这两个端口对应两个 FP8 的 vLLM 实例
URL_PREFILL_WORKER = os.getenv("URL_WORKER_LONG", "http://localhost:8001/v1/chat/completions") # 慢速道
URL_DECODE_WORKER  = os.getenv("URL_WORKER_SHORT", "http://localhost:8002/v1/chat/completions") # 快速道

# 阈值：超过多少 Token 算“长文本”？
# 根据你的 Analysis 任务特征，建议设为 3000 或 4000
try:
    LENGTH_THRESHOLD = int(os.getenv("STATIC_THRESHOLD", "3000"))
except ValueError:
    LENGTH_THRESHOLD = 3000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StaticRouter")

app = FastAPI()

# 全局 Client，复用连接池
# 禁用环境代理变量（例如 socks5 代理导致需要 socksio），本路由只转发到本机 worker。
# 增加连接池大小，防止长时间任务占满连接导致新请求阻塞
limits = httpx.Limits(max_keepalive_connections=50, max_connections=1000)
http_client = httpx.AsyncClient(timeout=None, trust_env=False, limits=limits) # 永不超时，让客户端自己决定

def estimate_token_count(messages: list) -> int:
    """
    简单粗暴的 Token 估算。
    为了不引入 tokenizer 的计算开销，我们用 字符数 / 4 来估算。
    对于路由决策来说，这个精度足够了。
    """
    txt = ""
    for m in messages:
        txt += m.get("content", "")
    return len(txt) // 4

async def forward_request(target_url: str, request: Request, body: dict):
    """通用转发逻辑"""
    try:
        # 构建转发请求
        # 注意：stream=True 是必须的，为了支持流式输出
        req = http_client.build_request(
            request.method,
            target_url,
            json=body,
            timeout=None
        )
        
        # 发送请求并获取流式响应
        r = await http_client.send(req, stream=True)
        
        return StreamingResponse(
            r.aiter_raw(),
            status_code=r.status_code,
            media_type=r.headers.get("content-type"),
            background=BackgroundTask(r.aclose),
        )
    except Exception as e:
        logger.error(f"Forward to {target_url} failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        
        # 1. 估算长度
        input_len = estimate_token_count(body.get("messages", []))
        
        # 2. 路由决策 (核心逻辑)
        if input_len > LENGTH_THRESHOLD:
            target = URL_PREFILL_WORKER
            tag = "[LONG -> A]"
        else:
            target = URL_DECODE_WORKER
            tag = "[SHORT -> B]"
            
        logger.info(f"{tag} Len={input_len} => Forwarding to {target}")
        
        # 3. 执行转发
        return await forward_request(target, request, body)
        
    except Exception as e:
        logger.error(f"Router Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Router Internal Error"})

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    # Router 跑在 5000 端口
    import uvicorn
    host = os.getenv("ROUTER_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("ROUTER_PORT", "5000"))
    except ValueError:
        port = 5000
    uvicorn.run(app, host=host, port=port)
