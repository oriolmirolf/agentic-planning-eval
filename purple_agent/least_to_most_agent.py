# /Oriol-TFM/purple_agent/least_to_most_agent.py
from __future__ import annotations

import os

from openai import OpenAI

from .base import PurpleAgent
from .openai_agent import _extract_text_from_responses

_LTM_SYSTEM = """You are a PDDL planning assistant using Least-to-Most decomposition.
1) List subgoals in a sensible order (Subgoals: ...).
2) Solve each subgoal in turn and accumulate a feasible global plan.
3) Output ONLY one final code block with the full plan, one action per line, no timestamps or comments.
"""

def _client(base_url: str | None, api_key: str | None) -> OpenAI:
    kwargs = {}
    if base_url: kwargs["base_url"] = base_url
    if api_key: kwargs["api_key"] = api_key
    return OpenAI(**kwargs)  # type: ignore

class LeastToMostPurpleAgent(PurpleAgent):
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(temperature)
        self._client = _client(base_url, api_key)

    def generate_plan(self, *, problem_nl: str) -> str:
        prompt = f"{_LTM_SYSTEM}\n\nProblem:\n{problem_nl.strip()}\n\nThink through subgoals, then provide the final plan."
        resp = self._client.responses.create(model=self.model, input=prompt, temperature=self.temperature)
        return _extract_text_from_responses(resp).strip()
