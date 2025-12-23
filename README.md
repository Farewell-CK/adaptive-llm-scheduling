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
*   **Guardrails**: Always maintain at least 1 worker for Short tasks ($N_{short} \ge 1$).
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
