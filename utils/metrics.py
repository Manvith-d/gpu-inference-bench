import time, statistics as stats
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Record:
    t_submit: float
    t_done: float
    tokens: int

@dataclass
class Meter:
    recs: List[Record] = field(default_factory=list)

    def add(self, t_s: float, t_d: float, tokens: int):
        self.recs.append(Record(t_s, t_d, tokens))

    def summary(self)->Dict[str, float]:
        lats = [(r.t_done - r.t_submit) * 1000.0 for r in self.recs]  # ms
        total_tokens = sum(r.tokens for r in self.recs)
        dur = (max(r.t_done for r in self.recs) - min(r.t_submit for r in self.recs))
        rps = len(self.recs) / max(dur, 1e-6)
        tps = total_tokens / max(dur, 1e-6)
        return {
            'count': len(lats),
            'p50_ms': stats.quantiles(lats, n=100)[49] if len(lats)>=2 else lats[0] if lats else 0,
            'p90_ms': stats.quantiles(lats, n=10)[8] if len(lats)>=2 else lats[0] if lats else 0,
            'p99_ms': sorted(lats)[int(0.99*len(lats))-1] if len(lats)>1 else (lats[0] if lats else 0),
            'req_per_sec': rps,
            'tok_per_sec': tps
        }
