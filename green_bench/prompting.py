from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable

from .openai_client import LLMRequest, OpenAICompatClient


# Public Types

@dataclass(frozen=True)
class StrategyOutput:
    """
    final_text MUST satisfy the benchmark output contract:
      - ONLY a fenced code block
      - one action per line inside the block
    trace is optional and may include intermediate artifacts (plans, critiques, scores).
    """
    final_text: str
    trace: str | None = None


class StrategyRunner(Protocol):
    def __call__(
        self,
        client: OpenAICompatClient,
        model_name: str,
        domain_prompt: str,
        problem_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> StrategyOutput: ...


@dataclass(frozen=True)
class StrategySpec:
    name: str
    run: StrategyRunner
    description: str


# Helpers

def _join(*parts: str) -> str:
    return "\n\n".join([p.strip() for p in parts if (p or "").strip()]).strip()


def _call(
    client: OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    return client.generate(
        LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


# Single Call Strategies

def _run_baseline(
        client: OpenAICompatClient,
        model_name: str,
        domain_prompt: str,
        problem_prompt: str,
        temperature: float,
        max_tokens: int,
) -> StrategyOutput:
    trace_lines: list[str] = []
    prompt = _join(domain_prompt, problem_prompt)
    raw = _call(client, model=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    
    trace_lines.append("=== baseline: prompt ===")
    trace_lines.append(prompt)
    trace_lines.append("=== baseline: raw_response ===")
    trace_lines.append(raw)
    # TODO: Normalize output contract?
    return StrategyOutput(final_text=raw, trace="\n".join(trace_lines))



# Two-Call Strategies

def _run_zero_shot_cot(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022).
    """
    trace_lines: list[str] = []
    # TODO: Tweak the prompt.
    draft_prompt = _join(
        domain_prompt,
        "You are solving a planning problem. Let's think step by step.",
        "First, write your reasoning step-by-step (can be concise).",
        "Then provide a DRAFT plan in a fenced code block (one action per line).",
        problem_prompt,
    )

    trace_lines.append("=== zero_shot_cot: draft_prompt ===")
    trace_lines.append(draft_prompt)

    # TODO: Use larger temperature and smaller max_tokens?
    draft_call = _call(
        client, 
        model=model_name, 
        prompt=draft_prompt, 
        temperature=temperature,
        max_tokens=2048
    )

    final_prompt = _join(
        "Extract the plan from the previous message.",
        "Here is the previous message:",
        draft_call,
    )

    trace_lines.append("=== zero_shot_cot: draft_response ===")
    trace_lines.append(draft_call)
    trace_lines.append("=== zero_shot_cot: extract_prompt ===")
    trace_lines.append(final_prompt)

    final_call = _call(
        client, 
        model=model_name, 
        prompt=final_prompt, 
        temperature=temperature,
        max_tokens=max_tokens
    )

    trace_lines.append("=== zero_shot_cot: extracted_plan_response ===")
    trace_lines.append(final_call)
    return StrategyOutput(final_text=final_call, trace="\n".join(trace_lines))



def _run_plan_and_solve(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models (Wang et al., 2023).
    """
    trace_lines: list[str] = []
    plan_prompt = _join(
        domain_prompt,
        "TASK: Devise a high-level plan (subgoals) for solving the planning problem.",
        "Do NOT output PDDL actions yet. Output 4-10 bullet points max.",
        problem_prompt,
    )

    trace_lines.append("=== plan_and_solve: plan_prompt ===")
    trace_lines.append(plan_prompt)

    high_level = _call(
        client,
        model=model_name,
        prompt=plan_prompt,
        temperature=temperature,
        max_tokens=2048,
    )

    trace_lines.append("=== plan_and_solve: high_level_plan ===")
    trace_lines.append(high_level)
    solve_prompt = _join(
        domain_prompt,
        "Use the following high-level plan (subgoals) to construct a valid action sequence.",
        "High-level plan:",
        high_level,
        problem_prompt,
    )
    final = _call(
        client,
        model=model_name,
        prompt=solve_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    trace_lines.append("=== plan_and_solve: solve_prompt ===")
    trace_lines.append(solve_prompt)
    trace_lines.append("=== plan_and_solve: final_response ===")
    trace_lines.append(final)
    return StrategyOutput(final_text=final, trace="\n".join(trace_lines))


def _run_step_back(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models (Zheng et al., 2024).
    """
    trace_lines: list[str] = []

    abstract_prompt = _join(
        domain_prompt,
        "STEP-BACK: Before solving, abstract the instance into high-level constraints/invariants.",
        "Output 5-12 concise bullet points capturing goals, constraints, and invariants.",
        problem_prompt,
    )

    trace_lines.append("=== step_back: abstract_prompt ===")
    trace_lines.append(abstract_prompt)

    principles = _call(
        client,
        model=model_name,
        prompt=abstract_prompt,
        temperature=temperature,
        max_tokens=2048,
    )

    trace_lines.append("=== step_back: principles ===")
    trace_lines.append(principles)

    solve_prompt = _join(
        domain_prompt,
        "Solve the planning problem using these principles/constraints as guidance:",
        principles,
        problem_prompt,
    )

    final = _call(
        client,
        model=model_name,
        prompt=solve_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    trace_lines.append("=== step_back: solve_prompt ===")
    trace_lines.append(solve_prompt)
    trace_lines.append("=== step_back: final_response ===")
    trace_lines.append(final)
    return StrategyOutput(final_text=final, trace="\n".join(trace_lines))


# Self-Refine (iterative loop)

def _run_self_refine(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023).
    """
    trace_lines: list[str] = []
    # Initial draft
    draft_prompt = _join(
        domain_prompt,
        "Generate an initial candidate plan.",
        problem_prompt,
    )

    trace_lines.append("=== self_refine: draft_prompt ===")
    trace_lines.append(draft_prompt)

    current = _call(
        client,
        model=model_name,
        prompt=draft_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    trace_lines.append("=== self_refine: draft_response ===")
    trace_lines.append(current)

    max_rounds = 3
    for r in range(1, max_rounds + 1):
        feedback_prompt = _join(
            domain_prompt,
            "You are a strict plan critic.",
            "Given the domain/problem and the candidate plan, identify concrete issues:",
            "- invalid action names / wrong objects",
            "- violated preconditions or impossible ordering (as far as can be inferred)",
            "- missing steps likely needed to reach the goal",
            "- formatting violations",
            "If there are NO issues, output exactly: NO_ISSUES",
            "Candidate plan:",
            current,
            problem_prompt,
        )
        
        trace_lines.append(f"=== self_refine: feedback_prompt_round_{r} ===")
        trace_lines.append(feedback_prompt)
        
        fb = _call(
            client,
            model=model_name,
            prompt=feedback_prompt,
            temperature=temperature,
            max_tokens=2048,
        ).strip()

        trace_lines.append(f"=== self_refine: feedback_round_{r} ===")
        trace_lines.append(fb)

        if fb.strip() == "NO_ISSUES":
            break

        refine_prompt = _join(
            domain_prompt,
            "Revise the candidate plan to address ALL issues in the feedback.",
            "Feedback:",
            fb,
            "Previous plan:",
            current,
            problem_prompt,
        )

        trace_lines.append(f"=== self_refine: refine_prompt_round_{r} ===")
        trace_lines.append(refine_prompt)

        current = _call(
            client,
            model=model_name,
            prompt=refine_prompt,
            temperature=0.2,
            max_tokens=max_tokens,
        )

        trace_lines.append(f"=== self_refine: refined_response_round_{r} ===")
        trace_lines.append(current)

    return StrategyOutput(final_text=current, trace="\n".join(trace_lines))


# Tree of Thoughts (beam search)





# Registry

STRATEGIES: dict[str, StrategySpec] = {
    "baseline": StrategySpec(
        name="baseline",
        run=_run_baseline,
        description="No extra prompting beyond the domain + problem prompt.",
    ),
    "zero_shot_cot": StrategySpec(
        name="zero_shot_cot",
        run=_run_zero_shot_cot,
        description="Zero-shot CoT as a 2-stage procedure: draft reasoning+plan, then extract plan-only.",
    ),
    "plan_and_solve": StrategySpec(
        name="plan_and_solve",
        run=_run_plan_and_solve,
        description="Plan-and-Solve: explicit high-level plan step, then solve conditioned on it.",
    ),
    "step_back": StrategySpec(
        name="step_back",
        run=_run_step_back,
        description="Step-Back: explicit abstraction/principles step, then solve conditioned on it.",
    ),
    "self_refine": StrategySpec(
        name="self_refine",
        run=_run_self_refine,
        description="Self-Refine: iterative FEEDBACK -> REFINE loop; returns best refined plan-only output.",
    ),
    #"tree_of_thought": StrategySpec(
    #    name="tree_of_thought",
    #    run=_run_tree_of_thought,
    #    description="Tree of Thoughts: beam search over partial plan prefixes using self-scored chunk expansions.",
    #),
    #"graph_of_thought": StrategySpec(
    #    name="graph_of_thought",
    #    run=_run_graph_of_thought,
    #    description="Graph of Thoughts: generate multiple plan nodes, aggregate top-k, apply feedback loop refinement.",
    #),
}


def run_strategy(
    strategy: str,
    *,
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    spec = STRATEGIES.get(strategy)
    if not spec:
        raise ValueError(f"Unknown strategy '{strategy}'. Known: {sorted(STRATEGIES)}")
    return spec.run(
        client=client,
        model_name=model_name,
        domain_prompt=domain_prompt,
        problem_prompt=problem_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )