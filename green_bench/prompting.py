from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import re

from .openai_client import LLMRequest, OpenAICompatClient


# Public Types

@dataclass(frozen=True)
class StrategyOutput:
    """
    Output container for a prompting strategy.

    Contract:
      - final_text MUST be a single fenced code block (triple backticks)
      - inside the block: one action per line (no surrounding prose)
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
    """Join non-empty blocks with a blank line separator."""
    return "\n\n".join([p.strip() for p in parts if (p or "").strip()]).strip()


def _context(domain_prompt: str, problem_prompt: str) -> str:
    """
    Neutral wrapper shared across all strategies.

    Intentionally:
      - includes domain + problem content
      - does NOT include output-format rules (those appear only in final calls)
    """
    return _join(
        "You will receive:",
        "(1) a DOMAIN BRIEF describing available actions and their required argument order, and",
        "(2) a PROBLEM PROMPT describing concrete objects, initial state, and goal.",
        "DOMAIN BRIEF:",
        domain_prompt,
        "PROBLEM PROMPT:",
        problem_prompt,
    )


_OUTPUT_REQUIREMENTS = """OUTPUT REQUIREMENTS (MANDATORY):
- Output MUST be a single fenced code block (triple backticks).
- Inside the code block: one action per line, exactly in the form: action_name arg1 arg2 ...
- Use ONLY action names that appear in the DOMAIN BRIEF (case-insensitive matching is allowed, but keep names consistent).
- Use the EXACT argument order defined in the DOMAIN BRIEF.
- Use ONLY object names that appear in the PROBLEM PROMPT (no renaming, no aliases, no added objects).
- Do NOT include numbering, commentary, blank lines, or multiple alternative plans.
- If you are unsure, make the safest progress: prefer actions that you can justify from the PROBLEM PROMPT.
- Do NOT restate the task, the state, or the goal.
- If the instance is unsolvable under the given constraints, output exactly one fenced code block containing:
UNSOLVABLE

Example (exactly):
```
UNSOLVABLE
```
"""


def _final_prompt(
    domain_prompt: str,
    problem_prompt: str,
    *,
    technique: str | None = None,
    extra: str | None = None,
) -> str:
    """Final-step prompt: includes output requirements."""
    return _join(
        _context(domain_prompt, problem_prompt),
        technique or "",
        extra or "",
        _OUTPUT_REQUIREMENTS,
    )


def _aux_prompt(
    domain_prompt: str,
    problem_prompt: str,
    *,
    instruction: str,
    extra: str | None = None,
) -> str:
    """
    Auxiliary-step prompt: MUST NOT include output requirements
    (prevents intermediate-step format learning/contamination).
    """
    return _join(
        _context(domain_prompt, problem_prompt),
        instruction,
        extra or "",
    )


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


# Minimal robustness: extract the first fenced code block if present.
_CODEBLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)\n?```", re.DOTALL)

def _ensure_single_codeblock(text: str) -> str:
    """
    Ensure StrategyOutput.final_text is a single fenced code block.

    Policy:
      - If multiple fenced blocks exist, keep the LAST one.
        Rationale: models often put scratch/drafts earlier and the final answer last.
      - If no fenced block exists, wrap the raw text.
    """
    return text
    raw = (text or "").strip()
    blocks = _CODEBLOCK_RE.findall(raw)
    if blocks:
        body = (blocks[-1] or "").strip("\n")
        return f"```\n{body}\n```"
    return f"```\n{raw}\n```"


def _merge_action_text(prefix: str, chunk: str) -> str:
    """
    Merge two action-text fragments into a single action list with NO blank lines.

    Important for ToT: using _join() would insert blank lines, violating your contract.
    """
    lines: list[str] = []
    for part in (prefix, chunk):
        for ln in (part or "").splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return "\n".join(lines).strip()


def _clean_chunk(text: str, *, max_lines: int = 3) -> str:
    """Keep up to max_lines non-empty lines, trimmed; used for ToT chunks."""
    out: list[str] = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(ln)
        if len(out) >= max_lines:
            break
    return "\n".join(out).strip()


# Strategies

def _run_baseline(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Baseline (no prompting technique).

    Note: This is intentionally not attributed to a specific technique paper.
    It uses the standardized wrapper + mandatory output requirements.
    """
    trace: list[str] = []
    prompt = _final_prompt(domain_prompt, problem_prompt)
    raw = _call(client, model=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens)

    trace.append("=== baseline: prompt ===")
    trace.append(prompt)
    trace.append("=== baseline: raw ===")
    trace.append(raw)

    return StrategyOutput(final_text=_ensure_single_codeblock(raw), trace="\n".join(trace))


def _run_zero_shot_cot(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Zero-shot Chain-of-Thought (CoT).

    "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022).
    Adaptation for this benchmark: include the trigger phrase but require
    reasoning to be kept internal to satisfy the plan-only output contract.
    """
    trace: list[str] = []
    technique = _join(
        "Let's think step by step.",
        "Think step-by-step internally, but DO NOT output your reasoning.",
    )
    prompt = _final_prompt(domain_prompt, problem_prompt, technique=technique)

    raw = _call(client, model=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens)

    trace.append("=== zero_shot_cot: prompt ===")
    trace.append(prompt)
    trace.append("=== zero_shot_cot: raw ===")
    trace.append(raw)

    return StrategyOutput(final_text=_ensure_single_codeblock(raw), trace="\n".join(trace))


def _run_plan_and_solve(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Plan-and-Solve prompting.

    "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models"
    (Wang et al., 2023).

    Procedure:
      (1) produce a high-level plan / subgoals (aux step; no actions)
      (2) carry out the plan into an action sequence (final step; output contract)
    """
    trace: list[str] = []

    plan_instruction = _join(
        "PLAN-AND-SOLVE (Step 1/2):",
        "First, understand the problem and devise a plan to solve it.",
        "Output ONLY 4-10 bullet points (subgoals / ordering constraints).",
        "Do NOT output actions.",
    )
    plan_prompt = _aux_prompt(domain_prompt, problem_prompt, instruction=plan_instruction)

    high_level = _call(
        client,
        model=model_name,
        prompt=plan_prompt,
        temperature=temperature,
        max_tokens=min(1024, max_tokens),
    )

    trace.append("=== plan_and_solve: plan_prompt ===")
    trace.append(plan_prompt)
    trace.append("=== plan_and_solve: high_level ===")
    trace.append(high_level)

    extra = _join(
        "PLAN-AND-SOLVE (Step 2/2): Carry out the high-level plan below to produce the final action sequence.",
        "High-level plan:",
        high_level,
    )
    solve_prompt = _final_prompt(domain_prompt, problem_prompt, extra=extra)

    final_raw = _call(
        client,
        model=model_name,
        prompt=solve_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    trace.append("=== plan_and_solve: solve_prompt ===")
    trace.append(solve_prompt)
    trace.append("=== plan_and_solve: final_raw ===")
    trace.append(final_raw)

    return StrategyOutput(final_text=_ensure_single_codeblock(final_raw), trace="\n".join(trace))


def _run_step_back(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Step-Back prompting.

    "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"
    (Zheng et al., 2023; ICLR 2024).

    Procedure:
      (1) abstract instance into principles/constraints (aux step; no actions)
      (2) solve using those principles (final step; output contract)
    """
    trace: list[str] = []

    abstract_instruction = _join(
        "STEP-BACK (Step 1/2):",
        "Before solving, step back and write high-level constraints/invariants relevant to this instance.",
        "Output 6-12 concise bullet points.",
        "Do NOT output actions.",
    )
    abstract_prompt = _aux_prompt(domain_prompt, problem_prompt, instruction=abstract_instruction)

    principles = _call(
        client,
        model=model_name,
        prompt=abstract_prompt,
        temperature=temperature,
        max_tokens=min(1024, max_tokens),
    )

    trace.append("=== step_back: abstract_prompt ===")
    trace.append(abstract_prompt)
    trace.append("=== step_back: principles ===")
    trace.append(principles)

    extra = _join(
        "STEP-BACK (Step 2/2): Solve using the principles/constraints below as guidance.",
        "Principles/constraints:",
        principles,
    )
    solve_prompt = _final_prompt(domain_prompt, problem_prompt, extra=extra)

    final_raw = _call(
        client,
        model=model_name,
        prompt=solve_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    trace.append("=== step_back: solve_prompt ===")
    trace.append(solve_prompt)
    trace.append("=== step_back: final_raw ===")
    trace.append(final_raw)

    return StrategyOutput(final_text=_ensure_single_codeblock(final_raw), trace="\n".join(trace))


def _run_self_refine(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Self-Refine.

    "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023).

    Adaptation for this benchmark:
      - draft is produced in final format (code block)
      - critic step is auxiliary (no output requirements) and temperature=0 for stability
      - revise step is final-format again
    """
    trace: list[str] = []

    # Draft (final-format)
    draft_prompt = _final_prompt(domain_prompt, problem_prompt, extra="Generate an initial candidate plan.")
    draft_raw = _call(
        client,
        model=model_name,
        prompt=draft_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    current = _ensure_single_codeblock(draft_raw)

    trace.append("=== self_refine: draft_prompt ===")
    trace.append(draft_prompt)
    trace.append("=== self_refine: draft_raw ===")
    trace.append(draft_raw)
    trace.append("=== self_refine: current_codeblock ===")
    trace.append(current)

    max_rounds = 3
    for r in range(1, max_rounds + 1):
        critic_instruction = _join(
            "SELF-REFINE CRITIC:",
            "You are a strict plan critic.",
            "Do NOT output any code blocks.",
            "Identify ONLY concrete issues you can verify from DOMAIN BRIEF / PROBLEM PROMPT / the candidate plan text:",
            "- invalid action names",
            "- invalid object names",
            "- wrong argument count or wrong argument order vs DOMAIN BRIEF",
            "- formatting violations (numbering/commentary/multiple plans/blank lines)",
            "If there are NO issues, output exactly: NO_ISSUES",
            "Otherwise output a short numbered list. For each issue, quote the offending line.",
        )
        critic_extra = _join("Candidate plan:", current)
        critic_prompt = _aux_prompt(domain_prompt, problem_prompt, instruction=critic_instruction, extra=critic_extra)

        fb = _call(
            client,
            model=model_name,
            prompt=critic_prompt,
            temperature=0.0,
            max_tokens=min(1024, max_tokens),
        ).strip()

        trace.append(f"=== self_refine: critic_prompt_round_{r} ===")
        trace.append(critic_prompt)
        trace.append(f"=== self_refine: feedback_round_{r} ===")
        trace.append(fb)

        if fb == "NO_ISSUES":
            break

        revise_extra = _join(
            "Revise the candidate plan to address ALL issues in the feedback.",
            "Feedback:",
            fb,
            "Previous plan:",
            current,
        )
        revise_prompt = _final_prompt(domain_prompt, problem_prompt, extra=revise_extra)

        revised_raw = _call(
            client,
            model=model_name,
            prompt=revise_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        revised = _ensure_single_codeblock(revised_raw)

        trace.append(f"=== self_refine: revise_prompt_round_{r} ===")
        trace.append(revise_prompt)
        trace.append(f"=== self_refine: revised_raw_round_{r} ===")
        trace.append(revised_raw)
        trace.append(f"=== self_refine: revised_codeblock_round_{r} ===")
        trace.append(revised)

        if revised.strip() == current.strip():
            break
        current = revised

    return StrategyOutput(final_text=current, trace="\n".join(trace))


# Tree of Thoughts

@dataclass
class _ToTNode:
    actions_prefix_text: str  # plain text (action lines only, no fences)
    score: float


_SCORE_RE = re.compile(r"^\s*(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")

def _parse_scores(text: str, n: int) -> list[float]:
    scores = [0.0] * n
    for ln in (text or "").splitlines():
        m = _SCORE_RE.match(ln)
        if not m:
            continue
        i = int(m.group(1))
        s = float(m.group(2))
        if 1 <= i <= n:
            scores[i - 1] = s
    return scores


def _run_tree_of_thought(
    client: OpenAICompatClient,
    model_name: str,
    domain_prompt: str,
    problem_prompt: str,
    temperature: float,
    max_tokens: int,
) -> StrategyOutput:
    """
    Tree of Thoughts (ToT).
    "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023).

    Faithful structural alignment with the original repo's BFS-style ToT:
      - PROPOSE: generate candidate continuations
      - EVALUATE: separate scoring call (analogous to value/vote prompting)
      - SELECT: beam/greedy select top prefixes
      - COMPLETE: produce a final plan satisfying the output contract
    """
    trace: list[str] = []

    beam_width = 2
    branching = 3
    max_depth = 6

    propose_temp = max(temperature, 0.7)
    eval_temp = 0.0

    beam: list[_ToTNode] = [_ToTNode(actions_prefix_text="", score=0.0)]

    trace.append("=== tree_of_thought: CONFIG ===")
    trace.append(f"beam_width={beam_width}, branching={branching}, max_depth={max_depth}")
    trace.append(f"propose_temp={propose_temp}, eval_temp={eval_temp}")

    for depth in range(max_depth):
        new_nodes: list[_ToTNode] = []
        trace.append(f"=== tree_of_thought: DEPTH {depth} ===")

        for b_idx, node in enumerate(beam):
            prefix_block = f"```\n{node.actions_prefix_text}\n```" if node.actions_prefix_text.strip() else "```\n\n```"

            propose_instruction = _join(
                "TREE-OF-THOUGHTS (PROPOSE):",
                f"Propose EXACTLY {branching} candidates for the next small chunk of actions.",
                "Each candidate must be a fenced code block with 1-3 action lines.",
                "Do NOT include any scores in this step.",
                "Return EXACTLY this structure:",
                "CANDIDATE 1",
                "```",
                "action_name arg1 arg2 ...",
                "```",
                "CANDIDATE 2",
                "```",
                "action_name arg1 arg2 ...",
                "```",
                "CANDIDATE 3",
                "```",
                "action_name arg1 arg2 ...",
                "```",
            )
            propose_extra = _join("PREFIX:", prefix_block)
            propose_prompt = _aux_prompt(domain_prompt, problem_prompt, instruction=propose_instruction, extra=propose_extra)

            proposed_raw = _call(
                client,
                model=model_name,
                prompt=propose_prompt,
                temperature=propose_temp,
                max_tokens=min(900, max_tokens),
            )

            trace.append(f"[beam={b_idx}] prefix_score={node.score}")
            trace.append("=== propose_prompt ===")
            trace.append(propose_prompt)
            trace.append("=== proposed_raw ===")
            trace.append(proposed_raw)

            # Take first N code blocks as candidate chunks
            blocks = _CODEBLOCK_RE.findall(proposed_raw or "")
            candidates = [_clean_chunk(b, max_lines=3) for b in blocks[:branching]]
            candidates = [c for c in candidates if c]

            if not candidates:
                continue

            eval_instruction = _join(
                "TREE-OF-THOUGHTS (EVALUATE):",
                "Score each candidate chunk from 0 to 10.",
                "Higher is better: likely valid (per DOMAIN BRIEF), uses valid objects, and makes progress toward the goal.",
                "Do NOT output code blocks.",
                "Output EXACTLY:",
                "SCORES:",
                "1: <0-10>",
                "2: <0-10>",
                "3: <0-10>",
            )
            cand_text = []
            for i, c in enumerate(candidates, start=1):
                cand_text.append(f"CANDIDATE {i}:\n```\n{c}\n```")
            eval_extra = _join(
                "PREFIX:",
                prefix_block,
                "CANDIDATES:",
                "\n\n".join(cand_text),
            )
            eval_prompt = _aux_prompt(domain_prompt, problem_prompt, instruction=eval_instruction, extra=eval_extra)

            eval_raw = _call(
                client,
                model=model_name,
                prompt=eval_prompt,
                temperature=eval_temp,
                max_tokens=min(400, max_tokens),
            )

            scores = _parse_scores(eval_raw, n=len(candidates))

            trace.append("=== eval_prompt ===")
            trace.append(eval_prompt)
            trace.append("=== eval_raw ===")
            trace.append(eval_raw)
            trace.append(f"=== parsed_scores ===\n{scores}")

            for sc, chunk in zip(scores, candidates):
                new_prefix = _merge_action_text(node.actions_prefix_text, chunk)
                new_nodes.append(_ToTNode(actions_prefix_text=new_prefix, score=node.score + float(sc)))

        if not new_nodes:
            trace.append("=== tree_of_thought: NO NEW NODES; STOPPING ===")
            break

        new_nodes.sort(key=lambda n: n.score, reverse=True)
        beam = new_nodes[:beam_width]

    best = max(beam, key=lambda n: n.score)
    prefix_block = f"```\n{best.actions_prefix_text}\n```" if best.actions_prefix_text.strip() else "```\n\n```"

    complete_extra = _join(
        "TREE-OF-THOUGHTS (COMPLETE):",
        "Complete a full goal-reaching plan starting from the prefix below.",
        "You may minimally adjust the prefix only if necessary.",
        "PREFIX:",
        prefix_block,
    )
    complete_prompt = _final_prompt(domain_prompt, problem_prompt, extra=complete_extra)

    final_raw = _call(
        client,
        model=model_name,
        prompt=complete_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    trace.append("=== tree_of_thought: selected_prefix ===")
    trace.append(prefix_block)
    trace.append("=== tree_of_thought: complete_prompt ===")
    trace.append(complete_prompt)
    trace.append("=== tree_of_thought: final_raw ===")
    trace.append(final_raw)

    return StrategyOutput(final_text=_ensure_single_codeblock(final_raw), trace="\n".join(trace))


# Registry

STRATEGIES: dict[str, StrategySpec] = {
    "baseline": StrategySpec(
        name="baseline",
        run=_run_baseline,
        description="No technique; standardized wrapper + mandatory output requirements.",
    ),
    "zero_shot_cot": StrategySpec(
        name="zero_shot_cot",
        run=_run_zero_shot_cot,
        description="Zero-shot CoT.",
    ),
    "plan_and_solve": StrategySpec(
        name="plan_and_solve",
        run=_run_plan_and_solve,
        description="Plan-and-Solve : subgoals (aux) then execute into plan (final).",
    ),
    "step_back": StrategySpec(
        name="step_back",
        run=_run_step_back,
        description="Step-Back: abstract constraints (aux) then solve (final).",
    ),
    "self_refine": StrategySpec(
        name="self_refine",
        run=_run_self_refine,
        description="Self-Refine: draft -> critique (aux) -> revise (final).",
    ),
    "tree_of_thought": StrategySpec(
        name="tree_of_thought",
        run=_run_tree_of_thought,
        description="Tree of Thoughts: propose + eval + beam, then complete (final).",
    ),
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
