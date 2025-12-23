# AdaSplit: Adaptive Split-Scheduling for Mixed-Workload LLM Inference

**AdaSplit** is a dynamic resource partitioning framework designed to optimize Large Language Model (LLM) inference performance under mixed workloads (combining short conversational chats and long-context analysis tasks).

## 🎯 Goal
To maximize system throughput and minimize latency by dynamically allocating GPU resources between "Short" and "Long" task partitions based on real-time demand.

## 🏗 System Architecture

### Hardware Abstraction
*   **Total Resources**: 8 GPUs.
*   **Units**: Abstraced into **4 Homogeneous Units** (Unit 1-4).
*   **Configuration**: Each Unit operates as an independent **vLLM** instance with Tensor Parallelism (TP)=2 and FP8 quantization.
*   **Endpoints**: Ports 8001, 8002, 8003, 8004.

### Components
*   **Control Plane**: A centralized **Router** (FastAPI) acting as the gateway. It implements **Late Binding**, holding requests in internal queues until a specific Worker slot becomes available.
*   **Data Plane**: The vLLM Workers responsible for the actual inference execution.

## 🧠 Core Algorithms

AdaSplit employs a **Two-Level Scheduling Framework**:

### 1. Macro-Layer: Q-HAP (Queue-aware Hysteretic Adaptive Partitioning)
Dynamically adjusts the number of workers assigned to the "Long" vs. "Short" pools.
*   **Logic**: Monitors the backlog in the Router's internal queues.
*   **Scale Up**: If `queue_long > 10`, reassign a Short worker to the Long pool.
*   **Scale Down**: If `queue_long < 2`, return a Long worker to the Short pool.
*   **Guardrails**: Always maintain at least 1 worker for Short tasks ($N_{short} ≥ 1$).
*   **Hysteresis**: A cooldown period (e.g., 5s) prevents oscillation during rapid traffic changes.

### 2. Micro-Layer: RASP (Risk-Aware Stealing Policy)
Allows idle workers to process tasks from the overloaded partition under strict safety conditions.
*   **Logic**: Enables "Work Stealing" during "OFF" periods.
*   **Rule**: A Short worker can steal a Long task **IF AND ONLY IF**:
    1.  Its own `queue_short` is empty.
    2.  The worker has been idle for $> \tau_{cool}$ (e.g., 2s).

## 🚦 Concurrency Control (Crucial)
To fully utilize vLLM's **Continuous Batching** capabilities, AdaSplit moves beyond simple boolean (`is_busy`) locking.
*   **Mechanism**: **Semaphore/Counter**.
*   **Parameter**: `WORKER_CONCURRENCY_LIMIT` (Default: **8**).
*   **Logic**: A worker is considered "available" only if `active_requests < 8`. Requests exceeding this limit are held in the Router queue, which is essential for Q-HAP to detect saturation signals.

## 📂 Project Structure

*   **`start/`**: Launch scripts for the vLLM environment.
    *   `start_vllm_4_instances.sh`: Starts the 4 vLLM units.
*   **`tools/`**: 
    *   `router_dynamic.py`: **Main Logic**. Implements AdaSplit (Q-HAP + RASP) with semaphore-based concurrency.
    *   `router_static.py`: Baseline router with fixed partitioning.
    *   `benchmark_client.py`: Load generator and performance measurement.
    *   `monitor_vllm.py`: Metric scraper (Prometheus format).
    *   `workload_gen.py`: Generates synthetic mixed traces.

---

# AdaSplit: 混合负载下的自适应 LLM 推理资源分区

**AdaSplit** 是一个动态资源分区框架，旨在优化混合负载（即短文本对话与长文本分析混合）下的大模型推理性能。

## 🎯 项目目标
通过根据实时负载需求，在“长文本（Long）”和“短文本（Short）”任务分区之间动态分配 GPU 资源，从而最大化系统吞吐量并最小化延迟。

## 🏗 系统架构

### 硬件抽象
*   **资源总量**: 8 张 GPU 卡。
*   **单元划分**: 抽象为 **4 个同构单元 (Unit 1-4)**。
*   **配置**: 每个单元是一个独立运行的 **vLLM** 实例（TP=2, FP8 量化）。
*   **监听端口**: 8001, 8002, 8003, 8004。

### 组件
*   **控制平面 (Control Plane)**: 基于 FastAPI 的 **Router**，位于客户端和 Worker 之间。采用 **延迟绑定 (Late Binding)** 机制，请求先在 Router 内部排队，直到 Worker 明确有空位（信号量允许）时才分发。
*   **数据平面 (Data Plane)**: 负责实际推理的 vLLM Worker。

## 🧠 核心算法

我们需要实现一套 **双层调度框架 (Two-Level Scheduling Framework)**：

### 1. 宏观层：Q-HAP (基于队列的迟滞自适应分区)
根据 Router 内部队列的积压情况，动态调整 Worker 在 "Long" 和 "Short" 组之间的角色分配。
*   **扩容 (Scale Up)**: 如果长文本队列积压 `queue_long > 10`，增加 Long 组的 Worker。
*   **缩容 (Scale Down)**: 如果长文本队列积压 `queue_long < 2`，减少 Long 组的 Worker。
*   **安全护栏**: 必须始终保留至少 1 个 Worker 给短文本任务 ($N_{short} ≥ 1$)。
*   **迟滞 (Hysteresis)**: 设置冷却时间（如 5秒），防止角色频繁切换导致震荡。

### 2. 微观层：RASP (风险感知工作窃取)
允许空闲的 Worker 在安全的情况下“窃取”另一组的任务。
*   **规则**: 一个 Short 组的 Worker 仅当满足以下所有条件时，才允许窃取一个 Long 任务：
    1.  它自己的队列 `queue_short` 是空的。
    2.  该 Worker 已经持续空闲了超过 `RASP_STEAL_COOLDOWN` (例如 2秒)，这意味着当前处于流量低谷期 (OFF Period)。

## 🚦 并发控制 (关键细节)
为了充分利用 vLLM 的 **Continuous Batching (连续批处理)** 性能，我们摒弃了简单的布尔值锁定。
*   **机制**: **信号量/计数器 (Semaphore/Counter)**。
*   **参数**: `WORKER_CONCURRENCY_LIMIT` (默认: **8**)。
*   **逻辑**: 只有当 Worker 的 `active_requests < 8` 时，才认为该 Worker 是“可用”的。
*   **原理**: 8 是 Llama-3-70B 在我们硬件上吞吐量饱和的拐点。超过 8 个的请求必须滞留在 Router 队列中，以便触发 Q-HAP 的扩容信号。

## 📂 文件结构说明 (`/workspace/tools/`)

*   **`router_dynamic.py`**: **[核心代码]** 主要的 Router 逻辑。实现了基于信号量的并发控制，以及完整的 Q-HAP 和 RASP 算法。
*   **`router_static.py`**: 基线 Router（固定分区策略）。
*   **`monitor_vllm.py`**: 监控指标抓取脚本。
*   **`benchmark_client.py`**: 负载生成与压测客户端。
*   **`micro_bench.py`**: 用于测定最佳并发数的微基准测试脚本。
