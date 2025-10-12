from typing import List
try:
    from transformers import pipeline, set_seed
    import torch
except Exception as e:
    pipeline = None

from .base import Backend

class HFBackend(Backend):
    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        if pipeline is None:
            raise RuntimeError("transformers not installed. pip install transformers torch")
        self.model_name = model_name
        self.device = 0 if (device is None and torch.cuda.is_available()) else -1
        self.pipe = pipeline("text-generation", model=model_name, device=self.device)

    def name(self)->str:
        return f"hf({self.model_name})"

    def warmup(self)->None:
        _ = self.pipe("warmup", max_new_tokens=8)

    def generate_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        outputs = self.pipe(prompts, max_new_tokens=max_new_tokens)
        return [o[0]['generated_text'] if isinstance(o, list) else o['generated_text'] for o in outputs]
