from typing import List

class Backend:
    def name(self) -> str:
        raise NotImplementedError

    def warmup(self) -> None:
        pass

    def generate_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Return generated strings for each prompt. Must be **pure** (no blocking on scheduler)."""
        raise NotImplementedError
