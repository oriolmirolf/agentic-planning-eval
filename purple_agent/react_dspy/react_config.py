# purple_agent/react_dspy/react_config.py

from __future__ import annotations
from typing import List, Sequence, Callable, Dict

from . import react_tools
from .react_settings import DEFAULT_ENABLED_TOOLS, ABLATION_TOOLSETS

ToolFn = Callable[..., str]

# Name → function
ALL_TOOLS: Dict[str, ToolFn] = {
    "llm_decompose_task": react_tools.llm_decompose_task,
    "llm_generate_plan_outline": react_tools.llm_generate_plan_outline,
    "llm_critique_plan": react_tools.llm_critique_plan,
    "llm_refine_plan": react_tools.llm_refine_plan,
    "llm_summarize_context": react_tools.llm_summarize_context,
}


def resolve_tools(
    tool_names: Sequence[str] | None = None,
    preset: str | None = None,
) -> List[ToolFn]:
    """
    Resolve tools chosen by:
    - explicit list, or
    - preset name, or
    - global DEFAULT_ENABLED_TOOLS
    """
    if preset is not None:
        if preset not in ABLATION_TOOLSETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(ABLATION_TOOLSETS)}")
        tool_names = ABLATION_TOOLSETS[preset]

    if tool_names is None:
        tool_names = DEFAULT_ENABLED_TOOLS

    resolved = []
    for name in tool_names:
        if name not in ALL_TOOLS:
            raise ValueError(
                f"Unknown tool {name!r}. "
                f"Available: {list(ALL_TOOLS.keys())}"
            )
        resolved.append(ALL_TOOLS[name])

    return resolved
