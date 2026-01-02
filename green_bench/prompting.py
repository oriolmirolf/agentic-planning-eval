from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable
import re

from .openai_client import LLMRequest, OpenAICompatClient


# Public Types
SYSTEM_PROMPT = \
"""
SYSTEM
You are an expert planner.

You will receive:
(1) a DOMAIN BRIEF describing the available actions and their required argument order, and
(2) an INSTANCE BRIEF describing the concrete objects, initial state, and goal.

Task:
Produce a single valid, goal-reaching plan.

Hard constraints:
- Output MUST be a single fenced code block (triple backticks).
- Inside the code block: one action per line, exactly in the form: action_name arg1 arg2 ...
- Use ONLY action names that appear in the DOMAIN BRIEF (case-insensitive matching is allowed, but keep names consistent).
- Use the EXACT argument order defined in the DOMAIN BRIEF.
- Use ONLY object names that appear in the INSTANCE BRIEF (no renaming, no aliases, no added objects).
- Do NOT include numbering, commentary, blank lines, or multiple alternative plans.
- If you are unsure, make the safest progress: prefer actions that you can justify from the INSTANCE BRIEF.
- Do NOT restate the task, the state, or the goal.

If the instance is unsolvable under the given constraints, output exactly:
```
UNSOLVABLE
```

USER
<domain_brief>
{{DOMAIN_BRIEF_NL}}
</domain_brief>

<instance_brief>
{{INSTANCE_BRIEF_NL}}
</instance_brief>

Return ONLY the plan as specified above. Example output:

```
action1 arg1 arg2
action2 arg2 arg3 arg4
...
```
"""


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

_OUTPUT_CONTRACT_REMINDER = """OUTPUT FORMAT (MANDATORY):
- Reply with ONLY one fenced code block.
- Inside the block: one action per line.
- No prose before or after the code block.
"""

_CODEBLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)\n?```", re.DOTALL)

def _normalize_plan_only(text: str) -> str:
    """
    Best-effort normalization to enforce the benchmark output contract.
    Returns ONLY one fenced code block.
    """
    raw = (text or "").strip()
    m = _CODEBLOCK_RE.search(raw)
    if m:
        body = m.group(1).strip("\n")
        return f"```\n{body}\n```"

    # No fenced block: salvage action-like lines if possible
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    action_lines = [ln for ln in lines if ln.startswith("(")]
    body = "\n".join(action_lines) if action_lines else raw
    return f"```\n{body}\n```"


def _extract_codeblock(text: str) -> str | None:
    """
    Return the *body* of the first fenced code block, or None if none exists.
    """
    raw = (text or "").strip()
    m = _CODEBLOCK_RE.search(raw)
    if not m:
        return None
    return (m.group(1) or "").strip("\n")


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
    prompt = SYSTEM_PROMPT.replace("{{DOMAIN_BRIEF_NL}}", domain_prompt).replace("{{INSTANCE_BRIEF_NL}}", problem_prompt)
    raw = _call(client, model=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    
    trace_lines.append("=== baseline: prompt ===")
    trace_lines.append(prompt)
    trace_lines.append("=== baseline: raw_response ===")
    trace_lines.append(raw)
    # TODO: Normalize output contract?
    return StrategyOutput(final_text=_normalize_plan_only(raw), trace="\n".join(trace_lines))


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
        "Return ONLY the plan.",
        "Here is the previous message:",
        draft_call,
        _OUTPUT_CONTRACT_REMINDER,
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
    return StrategyOutput(final_text=_normalize_plan_only(final_call), trace="\n".join(trace_lines))


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
        _OUTPUT_CONTRACT_REMINDER,
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
    return StrategyOutput(final_text=_normalize_plan_only(final), trace="\n".join(trace_lines))


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
        _OUTPUT_CONTRACT_REMINDER,
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
    return StrategyOutput(final_text=_normalize_plan_only(final), trace="\n".join(trace_lines))


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
        _OUTPUT_CONTRACT_REMINDER,
    )


    trace_lines.append("=== self_refine: draft_prompt ===")
    trace_lines.append(draft_prompt)

    current = _normalize_plan_only(_call(
        client,
        model=model_name,
        prompt=draft_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    ))


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
            _OUTPUT_CONTRACT_REMINDER,
        )

        trace_lines.append(f"=== self_refine: refine_prompt_round_{r} ===")
        trace_lines.append(refine_prompt)

        current = _normalize_plan_only(_call(
            client,
            model=model_name,
            prompt=refine_prompt,
            temperature=0.2,
            max_tokens=max_tokens,
        ))

        trace_lines.append(f"=== self_refine: refined_response_round_{r} ===")
        trace_lines.append(current)

    return StrategyOutput(final_text=current, trace="\n".join(trace_lines))


# Tree of Thoughts (beam search)
@dataclass
class _ToTNode:
    actions: list[str]
    score: float
    parent_idx: int | None = None


def _parse_tot_candidates(text: str) -> list[tuple[float, list[str]]]:
    """
    Expected format (model-generated):
      CANDIDATE 1
      SCORE: 7
      ```
      (action ...)
      (action ...)
      ```
    """
    parts = re.split(r"\bCANDIDATE\s+\d+\b", text or "", flags=re.IGNORECASE)
    out: list[tuple[float, list[str]]] = []

    for p in parts[1:]:
        m = re.search(r"SCORE\s*:\s*([0-9]+(?:\.[0-9]+)?)", p, flags=re.IGNORECASE)
        score = float(m.group(1)) if m else 0.0
        body = _extract_codeblock(p)
        # Prefer codeblock body; otherwise salvage action-like lines.
        if body is not None:
            lines = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("(")]
        else:
            lines = [ln.strip() for ln in p.splitlines() if ln.strip().startswith("(")]

        # Enforce the intended "chunk" size (1–3 actions).
        lines = lines[:3]

        if lines:
            out.append((score, lines))

    out.sort(key=lambda x: x[0], reverse=True)
    return out



def _run_tree_of_thought(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Tree of Thoughts (Yao et al., 2023): generate multiple candidate "thoughts"

    Practical adaptation for PDDL-plan generation:
      - Node state = partial action sequence
      - Expansion = propose K next chunks (1-3 actions) + self-score
      - Search = beam over cumulative score
      - Completion = complete from best prefix, then normalize to plan-only
    """
    trace_lines: list[str] = []
    beam_width = 2
    branching = 3
    max_depth = 6

    # Start with empty plan prefix
    beam: list[_ToTNode] = [_ToTNode(actions=[], score=0.0, parent_idx=None)]

    trace_lines.append("=== TREE_OF_THOUGHT: CONFIG ===")
    trace_lines.append(f"beam_width={beam_width}, branching={branching}, max_depth={max_depth}")

    gen_temp = max(temperature, 0.7)  # ToT needs diversity; keep eval self-score deterministic-ish via instructions

    for depth in range(max_depth):
        new_nodes: list[_ToTNode] = []
        trace_lines.append(f"=== TREE_OF_THOUGHT: DEPTH {depth} ===")

        for b_idx, node in enumerate(beam):
            prefix_block = "```\n" + "\n".join(node.actions) + "\n```" if node.actions else "```\n\n```"

            expand_prompt = _join(
                domain_prompt,
                "TREE-OF-THOUGHTS EXPANSION:",
                "IMPORTANT: For this EXPANSION step only, you MUST output MULTIPLE fenced code blocks (one per candidate) as specified below, even if earlier instructions require a single code block.",
                "Given the domain/problem and the current partial plan prefix, propose "
                f"{branching} candidates for the NEXT SMALL CHUNK of actions (1-3 actions).",
                "For each candidate, provide a SCORE from 1 to 10 indicating how promising/valid it is.",
                "Return EXACTLY this structure for ALL candidates 1.." + str(branching) + ":",
                "Each candidate must follow:\n"
                "CANDIDATE <k>\nSCORE: <1-10>\n```\n(action ...)\n(action ...)\n```",

                "Current partial plan prefix:",
                prefix_block,
                problem_prompt,
            )

            expanded = _call(
                client,
                model=model_name,
                prompt=expand_prompt,
                temperature=gen_temp,
                max_tokens=768,
            )

            trace_lines.append(f"[beam={b_idx}] prefix_len={len(node.actions)} score={node.score}")
            trace_lines.append(expanded.strip())

            cands = _parse_tot_candidates(expanded)
            for sc, chunk in cands[:branching]:
                new_nodes.append(_ToTNode(actions=node.actions + chunk, score=node.score + sc, parent_idx=b_idx))

        # Beam select
        new_nodes.sort(key=lambda n: n.score, reverse=True)
        # Beam select
        if not new_nodes:
            trace_lines.append("=== TREE_OF_THOUGHT: NO NEW NODES; STOPPING ===")
            break

        new_nodes.sort(key=lambda n: n.score, reverse=True)
        beam = new_nodes[:beam_width]

        if not beam:
            break

    # Completion from best prefix
    best = max(beam, key=lambda n: n.score)
    prefix = "```\n" + "\n".join(best.actions) + "\n```" if best.actions else "```\n\n```"

    complete_prompt = _join(
        domain_prompt,
        "Complete the plan starting from the given prefix (you may keep the prefix as-is, or adjust minimally if needed).",
        "Prefix:",
        prefix,
        problem_prompt,
        _OUTPUT_CONTRACT_REMINDER,
    )
    final = _call(
        client,
        model=model_name,
        prompt=complete_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    final_text = _normalize_plan_only(final)
    trace_lines.append("=== TREE_OF_THOUGHT: SELECTED PREFIX ===")
    trace_lines.append(prefix)
    trace_lines.append("=== TREE_OF_THOUGHT: FINAL ===")
    trace_lines.append(final_text)

    return StrategyOutput(final_text=final_text, trace="\n".join(trace_lines))


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
    "tree_of_thought": StrategySpec(
        name="tree_of_thought",
        run=_run_tree_of_thought,
        description="Tree of Thoughts: beam search over partial plan prefixes using self-scored chunk expansions.",
    ),
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