import asyncio
import time
import httpx
import statistics
import random

ROUTER_URL = "http://localhost:5000/v1/chat/completions"

# 模拟超长 Prompt
BASE_TEXT = "The quick brown fox jumps over the lazy dog. " * 2000 
ANALYSIS_PROMPT = (BASE_TEXT * 10)[:28000] # ~7000 tokens
CHAT_PROMPT = "Hello, tell me a short joke."

async def send_request(client, prompt, req_type, req_id):
    """发送请求并计算 TTFT"""
    payload = {
        "model": "llama-3.3-70b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10 if req_type == "analysis" else 100, # Analysis 我们不在乎输出，只在乎占位
        "stream": True
    }
    
    start_time = time.time()
    try:
        async with client.stream("POST", ROUTER_URL, json=payload, timeout=None) as response:
            if response.status_code != 200:
                print(f"[{req_type} #{req_id}] Error: {response.status_code}")
                return None
            
            # 等待第一个 chunk (TTFT)
            async for chunk in response.aiter_raw():
                # 收到第一个字节就算 TTFT
                ttft = time.time() - start_time
                if req_type == "chat":
                    print(f"[{req_type} #{req_id}] TTFT: {ttft:.4f}s")
                return ttft
    except Exception as e:
        print(f"[{req_type} #{req_id}] Exception: {e}")
        return None

async def main():
    print(f"--- Starting Router Fix Verification ---")
    print(f"Target: {ROUTER_URL}")
    
    # 必须设置较大的连接池，以免测试客户端自己阻塞自己
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=200)
    async with httpx.AsyncClient(limits=limits, timeout=None, trust_env=False) as client:
        
        # 1. 发射一批 Analysis 任务 (Background)
        print("1. Flooding 50 Analysis requests (Long Context)...")
        tasks = []
        for i in range(50):
            tasks.append(asyncio.create_task(send_request(client, ANALYSIS_PROMPT, "analysis", i)))
            
        # 稍微等一下，确保 Router 接收到了压力
        await asyncio.sleep(2)
        
        # 2. 发射 Chat 任务 (我们需要测量的对象)
        print("2. Sending 10 Chat requests (Short Context)...")
        chat_tasks = []
        for i in range(10):
            # 模拟随机到达
            await asyncio.sleep(0.1)
            chat_tasks.append(asyncio.create_task(send_request(client, CHAT_PROMPT, "chat", i)))
            
        # 3. 等待 Chat 任务完成
        print("3. Waiting for Chat results...")
        results = await asyncio.gather(*chat_tasks)
        
        # 过滤有效结果
        chat_ttfts = [r for r in results if r is not None]
        
        print("\n=== Results ===")
        if chat_ttfts:
            print(f"Count: {len(chat_ttfts)}")
            print(f"Max TTFT: {max(chat_ttfts):.4f} s")
            print(f"Avg TTFT: {statistics.mean(chat_ttfts):.4f} s")
            print(f"Median TTFT: {statistics.median(chat_ttfts):.4f} s")
            
            if statistics.mean(chat_ttfts) < 2.0:
                print("\n✅ PASSED: Chat requests are fast despite congestion.")
            else:
                print("\n❌ FAILED: Chat requests are delayed.")
        else:
            print("❌ FAILED: No chat requests finished.")

if __name__ == "__main__":
    asyncio.run(main())
