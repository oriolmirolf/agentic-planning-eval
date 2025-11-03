from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)\n```", re.DOTALL)
ACTION_LINE_RE = re.compile(r"""^\s*(?:\d+\s*[:.]\s*)?(?:\d+\.\d+\s*:\s*)?\(\s*([a-zA-Z0-9_-]+)(?:\s+[^)]*)?\)\s*(?:;.*)?$""", re.VERBOSE | re.IGNORECASE)

@dataclass(slots=True)
class PlanStep:
    text: str

@dataclass(slots=True)
class ExtractedPlan:
    steps: List[PlanStep]
    def to_val_plan_text(self) -> str:
        return "\n".join(s.text for s in self.steps) + ("\n" if self.steps else "")

def extract_plan(raw: str) -> ExtractedPlan:
    m = CODE_BLOCK_RE.search(raw)
    body = m.group(1) if m else raw
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    steps: list[PlanStep] = []
    for ln in lines:
        if ACTION_LINE_RE.match(ln):
            action = ln[ln.index("("):] if "(" in ln else ln
            action = re.sub(r"\s+", " ", action)
            steps.append(PlanStep(text=action.strip()))
    return ExtractedPlan(steps=steps)
