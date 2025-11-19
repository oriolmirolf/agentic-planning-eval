# purple_agent/react_dspy/react_agent.py

from __future__ import annotations
from typing import Optional, Sequence
import threading

import dspy

from .react_config import resolve_tools


_DSPY_LM_INITIALIZED = False
_DSPY_LM_LOCK = threading.Lock()


def _ensure_dspy_lm(model: str, base_url: str, api_key: str, temperature: float = 0.2):
    global _DSPY_LM_INITIALIZED

    if _DSPY_LM_INITIALIZED:
        return

    with _DSPY_LM_LOCK:
        if _DSPY_LM_INITIALIZED:
            return

        # Respect existing DSPy settings
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            _DSPY_LM_INITIALIZED = True
            return

        lm = dspy.LM(
            model,
            api_base=base_url,
            api_key=api_key,
            temperature=temperature,
            cache=False,
        )
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
        _DSPY_LM_INITIALIZED = True


class ReActPlanningSignature(dspy.Signature):
    problem_nl: str = dspy.InputField()
    plan: str = dspy.OutputField()


class ReActDSPyPurpleAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        temperature: float = 0.2,
        enabled_tool_names: Optional[Sequence[str]] = None,
        preset: Optional[str] = None,
        max_iters: int = 8,
    ):
        if api_key is None:
            api_key = "dummy"

        _ensure_dspy_lm(model, base_url, api_key, temperature)

        # Load tools either from:
        # - a preset
        # - an explicit list
        # - global DEFAULT_ENABLED_TOOLS in settings
        tools = resolve_tools(
            # tool_names=enabled_tool_names,
            tool_names=["llm_decompose_task", "llm_generate_plan_outline", "llm_critique_plan", "llm_refine_plan", "llm_summarize_context"],
            preset=preset,
        )

        self.agent = dspy.ReAct(
            signature=ReActPlanningSignature,
            tools=tools,
            max_iters=max_iters,
        )

    def generate_plan(self, *, problem_nl: str) -> str:
        out = self.agent(problem_nl=problem_nl)
        txt = out.plan

        if "```" not in txt:
            txt = f"```plan\n{txt.strip()}\n```"

        return txt.strip()
