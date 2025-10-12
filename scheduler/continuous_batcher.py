import time
import queue
import threading
from typing import List, Tuple, Callable, Any

class ContinuousBatcher:
    """Simple continuous batching scheduler.
    - collect requests up to batch_size or timeout_ms
    - call backend.generate_batch once per micro-batch
    - return futures to callers
    """
    def __init__(self, backend, batch_size: int = 8, timeout_ms: int = 20):
        self.backend = backend
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.q = queue.Queue()
        self.stop = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, prompt: str, max_new_tokens: int):
        evt = threading.Event()
        slot = {'prompt': prompt, 'tokens': max_new_tokens, 'out': None, 'evt': evt}
        self.q.put(slot)
        return slot

    def _run(self):
        while not self.stop:
            first = None
            try:
                first = self.q.get(timeout=self.timeout_ms/1000.0)
            except queue.Empty:
                continue

            micro = [first]
            start = time.perf_counter()
            while len(micro) < self.batch_size:
                try:
                    micro.append(self.q.get_nowait())
                except queue.Empty:
                    break

            prompts = [m['prompt'] for m in micro]
            toks = max(m['tokens'] for m in micro)
            outs = self.backend.generate_batch(prompts, toks)
            for m, o in zip(micro, outs):
                m['out'] = o
                m['evt'].set()

    def shutdown(self):
        self.stop = True
        self.worker.join(timeout=1.0)
