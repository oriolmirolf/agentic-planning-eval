from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)\n```", re.DOTALL)
ACTION_LINE_RE = re.compile(r"""^\s*(?:\d+\s*[:.]\s*)?(?:\d+\.\d+\s*:\s*)?\(\s*([a-zA-Z0-9_-]+)(?:\s+[^)]*)?\)\s*(?:;.*)?$""", re.VERBOSE | re.IGNORECASE)
COST_RE = re.compile(r";\s*cost\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

@dataclass
class PlanStep:
    idx: int
    text: str

@dataclass
class ExtractedPlan:
    steps: List[PlanStep]
    raw_cost: Optional[float] = None
    def to_val_plan_text(self) -> str:
        return "\n".join(s.text for s in self.steps) + ("\n" if self.steps else "")

def extract_plan(raw: str) -> ExtractedPlan:
    m = CODE_BLOCK_RE.search(raw)
    body = m.group(1) if m else raw
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    steps: list[PlanStep] = []
    raw_cost = None
    for ln in lines:
        cost_match = COST_RE.search(ln)
        if cost_match:
            try:
                raw_cost = float(cost_match.group(1))
            except ValueError:
                pass
        if ACTION_LINE_RE.match(ln):
            action = ln[ln.index("("):] if "(" in ln else ln
            action = re.sub(r"\s+", " ", action)
            steps.append(PlanStep(idx=len(steps)+1, text=action.strip()))
    return ExtractedPlan(steps=steps, raw_cost=raw_cost)

def pretty(steps: Iterable[PlanStep]) -> str:
    return "\n".join(f"{s.idx:02d}: {s.text}" for s in steps)
