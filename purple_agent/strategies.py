from __future__ import annotations
import re, json, random
from typing import List, Tuple, Dict, Optional
from .llm_registry import LLMClient

_ACTION_LINE_RE = re.compile(r"""^\s*(?:\d+\s*[:.]\s*)?(?:\d+\.\d+\s*:\s*)?\(\s*([a-zA-Z0-9_-]+)(?:\s+[^)]*)?\)\s*(?:;.*)?$""", re.VERBOSE | re.IGNORECASE)

def _only_actions_codeblock(text: str) -> str:
    """Extract action lines; if none, return text as-is. Always wrap final in ``` ... ```."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    acts = [ln[ln.index("("):].strip() if "(" in ln else ln.strip()
            for ln in lines if _ACTION_LINE_RE.match(ln)]
    body = "\n".join(acts) if acts else text.strip()
    if "```" in body:
        return body.strip()
    return f"```\n{body}\n```"

def _score_prompt(problem_nl: str, plan: str) -> str:
    return f"""You are a strict PDDL plan judge.
Problem (natural language description):
{problem_nl}

Candidate plan:
````

{plan}

```

Score this plan for reaching the goal with plausible preconditions/effects and minimal redundancy.
Return ONLY:
SCORE: <number 0..100>
REASONS: <one-line rationale>
"""

def judge_score(judge: LLMClient, problem_nl: str, plan_text: str) -> float:
    out = judge.generate(_score_prompt(problem_nl, plan_text), system="You judge PDDL plans. Output 'SCORE: N' and 'REASONS: ...' only.")
    m = re.search(r"SCORE\s*:\s*([0-9]+(?:\.[0-9]+)?)", out, re.IGNORECASE)
    try:
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0

# -------------------- Strategies --------------------

def run_direct(planner: LLMClient, problem_nl: str) -> str:
    prompt = f"""{problem_nl}

Output ONLY the final plan in a single code block with one PDDL action per line."""
    out = planner.generate(prompt)
    return _only_actions_codeblock(out)

def run_cot(planner: LLMClient, problem_nl: str) -> str:
    prompt = f"""{problem_nl}

Think step by step about preconditions and effects, then output ONLY one final code block containing the plan (one action per line)."""
    out = planner.generate(prompt)
    return _only_actions_codeblock(out)

def run_cot_sc(planner: LLMClient, judge: LLMClient, problem_nl: str,
               *, samples: int = 6, temperature: float = 0.7) -> str:
    cands: List[Tuple[float, str]] = []
    for _ in range(max(1, samples)):
        out = planner.generate(
            f"{problem_nl}\n\nThink step by step. Then output ONLY the plan as one code block.",
            temperature=temperature
        )
        plan = _only_actions_codeblock(out)
        score = judge_score(judge, problem_nl, plan)
        cands.append((score, plan))
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]

def run_ltm(planner: LLMClient, problem_nl: str) -> str:
    prompt = f"""{problem_nl}

Use Least-to-Most decomposition:
1) List subgoals in order.
2) Solve each subgoal.
3) Finally, output ONLY one code block containing the complete plan."""
    out = planner.generate(prompt)
    return _only_actions_codeblock(out)

def run_react(controller: LLMClient, problem_nl: str, *, max_steps: int = 16) -> str:
    """
    Internal loop: Thought -> Action -> 'ok' ... Finish
    (single 'generate_plan' call scope; no human/green messaging)
    """
    history = f"""You will reason and build a PDDL plan via a ReAct loop.
    Rules:
    - At each step, write:
    Thought: <short reasoning>
    Action: (<operator args>)
    - I will always reply 'Observation: ok'.
    - When the goal is reachable with your accumulated actions, write:
    Finish:
    ```

    (<action1>)
    (<action2>)
    ...

    ```
    """
    state = f"{history}\n\nProblem:\n{problem_nl}\n\nBegin.\nThought:"
    plan_lines: List[str] = []
    for _ in range(max_steps):
        out = controller.generate(state)
        if re.search(r"(?im)^\s*finish\s*:", out):
            return _only_actions_codeblock(out)
        m = re.search(r"(?im)^\s*action\s*:\s*(\([^)]+\))", out)
        if m:
            act = m.group(1).strip()
            plan_lines.append(act)
            state += f"\nAction: {act}\nObservation: ok\nThought:"
        else:
            state += "\n[Hint: propose next as 'Action: (<op> <args>)']\nThought:"
    return _only_actions_codeblock("\n".join(plan_lines) if plan_lines else out)

def run_debate(planner_a: LLMClient, planner_b: LLMClient, judge: LLMClient, problem_nl: str) -> str:
    plan_a = _only_actions_codeblock(planner_a.generate(
        f"{problem_nl}\n\nPropose a complete plan. Output ONLY the code block."))
    plan_b = _only_actions_codeblock(planner_b.generate(
        f"{problem_nl}\n\nPropose a different valid plan if possible. Output ONLY the code block."))

    score_a = judge_score(judge, problem_nl, plan_a)
    score_b = judge_score(judge, problem_nl, plan_b)
    return plan_a if score_a >= score_b else plan_b

def run_verifier(planner: LLMClient, verifier: LLMClient, problem_nl: str) -> str:
    plan = _only_actions_codeblock(planner.generate(
        f"{problem_nl}\n\nPropose a plan. Output ONLY one code block with the plan."))

    verify_prompt = f"""You are a plan verifier for PDDL-like actions.
Check the plan for unsatisfied preconditions, wrong arity, or object/type mismatch given this problem description.
If you find issues, produce a corrected plan. Else, repeat the original plan.

Problem:
{problem_nl}

Plan:
{plan}

Return ONLY one code block with the corrected/final plan."""
    corrected = verifier.generate(verify_prompt, system="Be strict and minimal; fix only what's needed.")
    return _only_actions_codeblock(corrected)

def run_tot(planner: LLMClient, judge: LLMClient, problem_nl: str,
          *, depth: int = 4, branch: int = 3, beam: int = 3) -> str:
    """
    Small Beam/ToT search over partial plans using the judge as a heuristic.
    """
    Frontier = List[Tuple[List[str], float]]  # (partial_plan, score)
    frontier: Frontier = [([], 0.0)]
    for _ in range(max(1, depth)):
        candidates: Frontier = []
        for partial, _score in frontier:
            prefix = "\n".join(partial)
            prompt = f"""Problem:
                        {problem_nl}

                        Partial plan so far:
                        ```

                        {prefix}

                        ````

                        Propose up to {branch} plausible NEXT actions only (no commentary).
                        Return them as a code block with one action per line.
                        """
        out = planner.generate(prompt)
        actions = [ln.strip() for ln in out.splitlines() if _ACTION_LINE_RE.match(ln)]
        actions = actions[:branch] if actions else []
        if not actions:  # try a single action fallback
            single = planner.generate(f"{problem_nl}\n\nGiven partial plan:\n{prefix}\n\nPropose one next action only.")
            m = re.search(r"\(([^)]+)\)", single)
            if m:
                actions = [f"({m.group(1)})"]

        for act in actions:
            cand_plan = partial + [act]
            cand_text = "```\n" + "\n".join(cand_plan) + "\n```"
            s = judge_score(judge, problem_nl, cand_text)
            candidates.append((cand_plan, s))
        # beam prune
        candidates.sort(key=lambda t: t[1], reverse=True)
        frontier = candidates[:max(1, beam)] if candidates else frontier
    # choose best
    if not frontier:
        return "```\n```"
    best = max(frontier, key=lambda t: t[1])[0]
    return _only_actions_codeblock("\n".join(best))

def run_ensemble(planner: LLMClient, synth: LLMClient, judge: Optional[LLMClient],
                 problem_nl: str, *, n: int = 6, temp: float = 0.8) -> str:
    cands = []
    for _ in range(max(1, n)):
        out = planner.generate(
            f"{problem_nl}\n\nPropose a diverse valid plan. Output ONLY one code block.",
            temperature=temp
        )
        cands.append(_only_actions_codeblock(out))

    # ✅ precompute the joined candidates to avoid backslashes inside f-string braces
    cands_text = "\n\n".join(cands)

    synth_prompt = f"""You will receive several candidate PDDL plans. Merge them into a single best plan
that reaches the goal with minimal redundancy and no contradictory steps.
Return ONLY one code block with the final plan.

Candidates:
{cands_text}
"""
    merged = _only_actions_codeblock(synth.generate(synth_prompt))
    if judge:
        # optional final selection: merged vs best individual
        best_ind = max(cands, key=lambda p: judge_score(judge, problem_nl, p))
        s_merged = judge_score(judge, problem_nl, merged)
        s_best_ind = judge_score(judge, problem_nl, best_ind)
        return merged if s_merged >= s_best_ind else best_ind
    return merged