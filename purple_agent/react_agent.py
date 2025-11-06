# /Oriol-TFM/purple_agent/react_agent.py
from __future__ import annotations
import re, os
from typing import Optional, List
from openai import OpenAI
from .base import PurpleAgent
from .openai_agent import _extract_text_from_responses  # reuse your extractor

_FINISH_RE = re.compile(r"(?im)^\s*Finish\s*:\s*", re.IGNORECASE)
_ACTION_LINE_RE = re.compile(r"(?im)^\s*Action\s*:\s*(\([^)]+\))")

_REACT_SYSTEM = """You are a PDDL planning assistant using a ReAct loop (Thought, Action, Observation).
- Maintain a running candidate plan as PDDL actions.
- At each step, reason under 'Thought:' then propose exactly one next PDDL action under 'Action: (op ...)'.
- I'll reply with 'Observation: ok' after each action.
- When the plan reaches the goal, output 'Finish:' followed by ONLY one code block that contains the full plan, one action per line, no timestamps or comments.
"""

def _client(base_url: Optional[str], api_key: Optional[str]) -> OpenAI:
    kwargs = {}
    if base_url: kwargs["base_url"] = base_url
    if api_key: kwargs["api_key"] = api_key
    return OpenAI(**kwargs)  # type: ignore

class ReactPurpleAgent(PurpleAgent):
    """
    Very small ReAct driver:
      Repeats: Thought -> Action
      We feed back "Observation: ok" to advance the loop.
      Stops when the model writes 'Finish:' and a code block.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_steps: int = 16,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(temperature)
        self.max_steps = int(max_steps)
        self._client = _client(base_url, api_key)

    def _ask(self, text: str) -> str:
        resp = self._client.responses.create(model=self.model, input=text, temperature=self.temperature)
        return _extract_text_from_responses(resp)

    def generate_plan(self, *, problem_nl: str) -> str:
        ctx: List[str] = []
        ctx.append(_REACT_SYSTEM.strip())
        ctx.append("Problem:\n" + problem_nl.strip())
        ctx.append("Begin.\nThought:")  # first cue

        history = "\n\n".join(ctx)
        plan_lines: List[str] = []

        for step in range(self.max_steps):
            out = self._ask(history)
            if _FINISH_RE.search(out):
                # return whatever the model puts after Finish (ideally the code block)
                return out.strip()
            # try to extract one action line
            m = _ACTION_LINE_RE.search(out)
            if m:
                action = m.group(1).strip()
                plan_lines.append(action)
                history += f"\nAction: {action}\nObservation: ok\nThought:"
            else:
                # nudge to output a proper Action
                history += "\n[Hint: Please propose the next action as 'Action: (operator args)']\nThought:"

        # Fall back: emit whatever we accumulated as a code block
        if plan_lines:
            body = "\n".join(plan_lines)
            return f"```\n{body}\n```"
        # Otherwise, return the last output verbatim
        return out.strip()
