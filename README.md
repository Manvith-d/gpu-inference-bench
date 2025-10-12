# GPU Inference Optimization Prototype

A lightweight, reproducible **benchmark harness** to study how different LLM inference backends behave under **concurrent load** and **batching**. 
It simulates 100s of users, measures latency/throughput, and compares backends with knobs for **quantization**, **continuous batching**, and **KV-cache reuse**.

> This repo is intentionally backend-agnostic. It ships with:
> - a **DummyBackend** that simulates latency features (no GPU required),
> - an optional **Hugging Face backend** (CPU-friendly; e.g., `distilgpt2`),
> - interface stubs you can extend for **TensorRT-LLM** or **vLLM**.

---

## Features
- Synthetic **load generation** with configurable arrival rate and request size.
- **Continuous batching** scheduler (micro-batching).
- Metrics: P50/P90/P99 latency, tokens/sec, requests/sec.
- CSV exports and a simple **results summary**.
- Works out of the box with the **DummyBackend** (no heavy deps).

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py --backend dummy --users 200 --concurrency 64 --max-new-tokens 64 --arrival poisson --rate 50
```

Optional HF backend (CPU):
```bash
pip install torch transformers
python benchmark.py --backend hf --model distilgpt2 --users 50 --concurrency 8 --max-new-tokens 32
```

## Example Experiments
### 1) Effect of Batching
```bash
python benchmark.py --backend dummy --users 300 --concurrency 64 --batch-size 16
python benchmark.py --backend dummy --users 300 --concurrency 64 --batch-size 1
```
Compare P99 latency; batching should increase throughput and often reduce tail latency under load.

### 2) Effect of Quantization (simulated)
```bash
python benchmark.py --backend dummy --quantized true
```
Quantized flag reduces per-token latency in the DummyBackend to simulate INT8 speedups.

### 3) KV Cache Reuse (simulated)
```bash
python benchmark.py --backend dummy --kv-cache true
```

## Repository Layout
```
gpu-inference-bench/
  benchmark.py                  # CLI harness
  backends/
    base.py
    dummy_backend.py
    hf_backend.py               # optional, if transformers installed
  scheduler/
    continuous_batcher.py
  utils/
    metrics.py
  results/                      # CSV outputs and summaries
  scripts/
    run_benchmark.sh
  requirements.txt
  README.md
  LICENSE
```
## Extending to TensorRT-LLM or vLLM
Implement `backends/base.py:Backend` methods for a new backend and register it in `benchmark.py`. Keep the interface:
- `warmup()`
- `generate_batch(list_of_prompts, max_new_tokens)`
- `name()`

## License
MIT
