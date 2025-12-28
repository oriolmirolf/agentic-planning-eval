from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

StrategyFn = Callable[[str, str], str]


@dataclass(frozen=True)
class StrategySpec:
    """
    A prompting strategy for eliciting PDDL-style plans from a model.

    NOTE: All strategies must preserve the benchmark's output contract:
    the final answer must be ONLY a fenced code block of one action per line.
    """

    name: str
    build: StrategyFn
    description: str


def _join(domain_prompt: str, strategy_instructions: str, problem_prompt: str) -> str:
    dp = (domain_prompt or "").strip()
    sp = (strategy_instructions or "").strip()
    pp = (problem_prompt or "").strip()

    parts = [p for p in (dp, sp, pp) if p]
    return "\n\n".join(parts).strip()


def baseline(domain_prompt: str, problem_prompt: str) -> str:
    """No extra instructions beyond the domain + problem prompt."""
    return _join(domain_prompt, "", problem_prompt)


def zero_shot_cot(domain_prompt: str, problem_prompt: str) -> str:
    """
    Zero-shot Chain-of-Thought prompting (Kojima et al., 2022):
    ask the model to reason step-by-step, but keep reasoning PRIVATE.
    """
    instr = (
        "STRATEGY: zero_shot_cot\n"
        "- Privately think step by step to ensure each action's preconditions hold.\n"
        "- Do NOT output your reasoning.\n"
        "- Output ONLY the final plan in the required fenced code block format."
    )
    return _join(domain_prompt, instr, problem_prompt)


def plan_and_solve(domain_prompt: str, problem_prompt: str) -> str:
    """
    Plan-and-Solve prompting (Wang et al., 2023):
    first devise a high-level plan (private), then output the final plan.
    """
    instr = (
        "STRATEGY: plan_and_solve\n"
        "- Privately devise a short high-level plan (subgoals) before "
        "writing actions.\n"
        "- Then execute that plan by outputting a valid action sequence.\n"
        "- Do NOT output the high-level plan or any reasoning.\n"
        "- Output ONLY the final plan in the required fenced code block format."
    )
    return _join(domain_prompt, instr, problem_prompt)


def step_back(domain_prompt: str, problem_prompt: str) -> str:
    """
    Step-Back Prompting (Zheng et al., 2023):
    first abstract to principles/constraints (private), then solve.
    """
    instr = (
        "STRATEGY: step_back\n"
        "- Privately 'take a step back' and summarize the task as "
        "high-level constraints/invariants.\n"
        "- Use those principles to guide your solution.\n"
        "- Do NOT output the abstraction or reasoning.\n"
        "- Output ONLY the final plan in the required fenced code block format."
    )
    return _join(domain_prompt, instr, problem_prompt)


def self_refine(domain_prompt: str, problem_prompt: str) -> str:
    """
    Self-Refine (Madaan et al., 2023):
    draft -> critique -> refine, but in a single response (private loop).
    """
    instr = (
        "STRATEGY: self_refine\n"
        "- Privately draft a candidate plan.\n"
        "- Privately critique it for invalid actions, wrong objects, wrong order, "
        "or wrong format.\n"
        "- Privately revise until it should satisfy all constraints.\n"
        "- Output ONLY the final plan in the required fenced code block format."
    )
    return _join(domain_prompt, instr, problem_prompt)


STRATEGIES: dict[str, StrategySpec] = {
    "baseline": StrategySpec(
        name="baseline",
        build=baseline,
        description="No extra prompting beyond the domain + problem prompt.",
    ),
    "zero_shot_cot": StrategySpec(
        name="zero_shot_cot",
        build=zero_shot_cot,
        description=(
            "Zero-shot CoT: privately reason step-by-step; output only the final plan."
        ),
    ),
    "plan_and_solve": StrategySpec(
        name="plan_and_solve",
        build=plan_and_solve,
        description=(
            "Plan-and-Solve: privately make a high-level plan, then output "
            "the final plan."
        ),
    ),
    "step_back": StrategySpec(
        name="step_back",
        build=step_back,
        description=(
            "Step-Back: privately abstract constraints/principles, then "
            "output the final plan."
        ),
    ),
    "self_refine": StrategySpec(
        name="self_refine",
        build=self_refine,
        description=(
            "Self-Refine: privately draft/critique/refine; output only the final plan."
        ),
    ),
}


def build_prompt(strategy: str, *, domain_prompt: str, problem_prompt: str) -> str:
    spec = STRATEGIES.get(strategy)
    if not spec:
        raise ValueError(f"Unknown strategy '{strategy}'. Known: {sorted(STRATEGIES)}")
    return spec.build(domain_prompt, problem_prompt)
