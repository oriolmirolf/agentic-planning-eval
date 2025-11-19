# purple_agent/react_dspy/react_settings.py

"""
Global configuration for the DSPy ReAct planning agent.

Edit this to enable / disable tools without touching code.
"""

# Available tools (must match names in react_tools.ALL_TOOLS)
DEFAULT_ENABLED_TOOLS = [
    "llm_decompose_task",
    "llm_generate_plan_outline",
    "llm_critique_plan",
    "llm_refine_plan",
]

# Optional: define experiment presets
ABLATION_TOOLSETS = {
    "decomposition_only": ["llm_decompose_task"],
    "outline_only": ["llm_generate_plan_outline"],
    "critique_only": ["llm_critique_plan"],
    "refine_only": ["llm_refine_plan"],
    "all": [
        "llm_decompose_task",
        "llm_generate_plan_outline",
        "llm_critique_plan",
        "llm_refine_plan",
        "llm_summarize_context",
    ],
}
