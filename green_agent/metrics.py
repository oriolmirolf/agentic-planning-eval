# green_agent/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from .val_wrapper import run_val

@dataclass
class PlanMetrics:
    valid: bool
    length: int
    cost_value: Optional[float]
    first_failure_at: Optional[int]
    unsat_count: int
    redundant_indices: List[int]
    val_stdout: str
    failure_reason: Optional[str]  # new

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "length": self.length,
            "cost_value": self.cost_value,
            "first_failure_at": self.first_failure_at,
            "unsat_count": self.unsat_count,
            "redundant_indices": self.redundant_indices,
            "failure_reason": self.failure_reason,
        }

def compute_metrics(*, domain: str, problem: str, plan_text: str, val_path: Optional[str] = None) -> PlanMetrics:
    base = run_val(domain, problem, plan_text, val_path=val_path)
    length = len([ln for ln in plan_text.splitlines() if ln.strip().startswith("(")])
    first_fail = base.unsatisfied[0].at_action_index if (base.unsatisfied) else None
    redundant: list[int] = []
    if length > 0:
        lines = [ln for ln in plan_text.splitlines() if ln.strip()]
        for i in range(len(lines)):
            if not lines[i].strip().startswith("("): continue
            variant = "\n".join(lines[j] for j in range(len(lines)) if j != i) + "\n"
            res = run_val(domain, problem, variant, val_path=val_path)
            if res.ok:
                redundant.append(i + 1)  # 1-based
    return PlanMetrics(
        valid=base.ok,
        length=length,
        cost_value=base.value,
        first_failure_at=first_fail,
        unsat_count=len(base.unsatisfied),
        redundant_indices=redundant,
        val_stdout=base.stdout,
        failure_reason=base.failure_reason,
    )
