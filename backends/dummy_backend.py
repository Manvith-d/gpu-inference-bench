import time
import random
from typing import List
from .base import Backend

class DummyBackend(Backend):
    """Simulates LLM inference latency with tunable behaviors.
    - base_token_ms: baseline per-token latency
    - quantized: reduces per-token latency
    - kv_cache: reduces prefill latency
    - jitter: adds variance to simulate real clusters
    """
    def __init__(self, base_token_ms: float = 6.0, quantized: bool=False, kv_cache: bool=False, jitter: float=0.15):
        self.base_token_ms = base_token_ms
        self.quantized = quantized
        self.kv_cache = kv_cache
        self.jitter = jitter

    def name(self)->str:
        return f"dummy(q={self.quantized},kv={self.kv_cache})"

    def warmup(self)->None:
        time.sleep(0.05)

    def _sleep_ms(self, ms: float):
        # add multiplicative jitter
        ms *= random.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        time.sleep(ms/1000.0)

    def generate_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        batch = len(prompts)
        per_token = self.base_token_ms
        if self.quantized:
            per_token *= 0.65  # 35% faster to mimic INT8
        # prefill cost scales with input length; we simulate constant + cache gain
        prefill_ms = 3.0 * batch
        if self.kv_cache:
            prefill_ms *= 0.5
        # batching efficiency: modest kernel amortization
        per_token *= (0.9 ** max(0, batch-1))

        # total latency ~ prefill + tokens
        total_ms = prefill_ms + per_token * max_new_tokens
        self._sleep_ms(total_ms)
        return [f"[DUMMY_GEN]{p}::<{max_new_tokens}>" for p in prompts]
