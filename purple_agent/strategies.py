from __future__ import annotations
from typing import Optional, Dict, Any, List

# Utilities
def _extract_codeblock(text: str) -> str:
    import re
    m = re.search(r"```[a-zA-Z0-9_-]*\n(.*?)\n```", text, re.DOTALL)
    return (m.group(1).strip() if m else text.strip())

def _only_actions_codeblock(text: str) -> str:
    body = _extract_codeblock(text)
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    return "\n".join(lines) + ("\n" if lines else "")

# ---------------------------
# Baseline / CoT / LtM / SC
# ---------------------------

def run_base(planner: "LLMClient", problem_nl: str, *, temperature: float = 0.0) -> str:
    prompt = f"""You are a PDDL planning agent.
Task: Produce a valid PDDL plan that solves the problem.
Constraints:
- Output ONLY the final plan inside one fenced code block.
- One action per line in PDDL syntax, e.g.: (move a b)
- No explanations inside the code block.

Problem:
{problem_nl}
"""
    out = planner.generate(prompt, temperature=temperature)
    return _only_actions_codeblock(out)

def run_cot(planner: "LLMClient", problem_nl: str, *, temperature: float = 0.2) -> str:
    prompt = f"""You are a PDDL planning agent.
Think step-by-step about preconditions/effects, then return ONLY the final plan in a single fenced code block.
Do not include explanations inside the code block.

Problem:
{problem_nl}
"""
    out = planner.generate(prompt, temperature=temperature)
    return _only_actions_codeblock(out)

def run_ltm(planner: "LLMClient", problem_nl: str, *, temperature: float = 0.2) -> str:
    prompt = f"""You are a PDDL planning agent.
Use Least-to-Most:
1) Decompose into ordered subgoals.
2) Solve subgoals sequentially, carrying forward facts.
Finally, output ONLY the consolidated final PDDL plan in ONE fenced code block (no explanations inside).

Problem:
{problem_nl}
"""
    out = planner.generate(prompt, temperature=temperature)
    return _only_actions_codeblock(out)

def run_cot_self_consistency(planner: "LLMClient", judge: Optional["LLMClient"],
                             problem_nl: str, *, samples: int = 3, temp: float = 0.7) -> str:
    cands: List[str] = []
    cot_prompt = f"""You are a PDDL planning agent.
Think step-by-step, then return ONLY the final plan in one fenced code block.
Problem:
{problem_nl}
"""
    for _ in range(max(1, samples)):
        out = planner.generate(cot_prompt, temperature=temp)
        cands.append(_only_actions_codeblock(out))

    # majority by normalized text
    from collections import Counter
    normalized = ["\n".join(s.strip().split()) for s in cands]
    counts = Counter(normalized)
    best_text = max(counts.items(), key=lambda kv: kv[1])[0]
    best_plan = cands[normalized.index(best_text)]

    if judge:
        def judge_score(p: str) -> float:
            jprompt = f"""Rate this PDDL plan from 0 (invalid) to 1 (likely valid & minimal):
Plan:
````

{p}

```"""
            js = judge.generate(jprompt, temperature=0.0).strip()
            import re
            m = re.search(r"\b0(\.\d+)?\b|\b1(\.0+)?\b", js)
            try:
                return float(m.group(0)) if m else 0.0
            except Exception:
                return 0.0

        scored = [(p, judge_score(p)) for p in set(cands)]
        best_plan = max(scored, key=lambda t: t[1])[0]

    return best_plan

# ------------------------------
# Popular single-turn extras
# ------------------------------

def run_few_shot(planner: "LLMClient", problem_nl: str, *,
                 examples: List[Dict[str, str]] | None = None,
                 temperature: float = 0.0) -> str:
    """
    Few-shot prompting: prepend k exemplars (mini NL description + plan).
    examples: list of {"desc": "...", "plan": "(act ...)\n(...)"}
    """
    examples = examples or []
    exemplars = []
    for ex in examples:
        desc = (ex.get("desc") or "").strip()
        plan = (ex.get("plan") or "").strip()
        if plan:
            exemplars.append(f"""Example:
Description: {desc or "(omitted)"}
Plan:
```

{plan}

```""")
    ex_block = "\n\n".join(exemplars)

    prompt = f"""{ex_block}

Now produce a valid PDDL plan for the following problem.
Output ONLY the final plan in ONE fenced code block (no explanations inside).

Problem:
{problem_nl}
"""
    out = planner.generate(prompt, temperature=temperature)
    return _only_actions_codeblock(out)

def run_self_refine(planner: "LLMClient", problem_nl: str, *, temperature: float = 0.2) -> str:
    """
    Self-Refine (single-turn): draft -> critique -> revise; return only the final code block.
    """
    prompt = f"""You are a careful PDDL planning agent.

1) Draft a candidate plan.
2) Critique it for common PDDL errors (unsatisfied preconditions, wrong arguments, redundant steps).
3) Revise it accordingly.
4) Return ONLY the final plan in ONE fenced code block (no explanations inside).

Problem:
{problem_nl}
"""
    out = planner.generate(prompt, temperature=temperature)
    return _only_actions_codeblock(out)

def run_deliberate(planner: "LLMClient", problem_nl: str, *,
                   drafts: int = 3, temperature: float = 0.5) -> str:
    """
    Deliberate: generate a few short internal 'deliberations' then synthesize one final plan.
    Single model call; model is instructed to only output the final plan code block at the end.
    """
    prompt = f"""You are a PDDL planning agent.

Deliberate protocol:
- Think through {max(2, drafts)} brief alternative reasoning sketches (internally).
- Then synthesize ONE best plan.
- Output ONLY the final plan in ONE fenced code block (no explanations inside).

Problem:
{problem_nl}
"""
    out = planner.generate(prompt, temperature=temperature)
    return _only_actions_codeblock(out)

# -------------
# Dispatcher
# -------------
def run_strategy(name: str, *, planner: "LLMClient", judge: Optional["LLMClient"], problem_nl: str,
                 settings: Optional[Dict[str, Any]] = None) -> str:
    settings = settings or {}
    if name == "base":
        return run_base(planner, problem_nl, temperature=settings.get("base", {}).get("temperature", 0.0))
    if name == "cot":
        return run_cot(planner, problem_nl, temperature=settings.get("cot", {}).get("temperature", 0.2))
    if name == "ltm":
        return run_ltm(planner, problem_nl, temperature=settings.get("ltm", {}).get("temperature", 0.2))
    if name == "cot_sc":
        sc = settings.get("cot_sc", {})
        return run_cot_self_consistency(planner, judge, problem_nl,
                                        samples=int(sc.get("samples", 3)),
                                        temp=float(sc.get("temperature", 0.7)))
    if name == "few_shot":
        fs = settings.get("few_shot", {})
        return run_few_shot(planner, problem_nl,
                            examples=fs.get("examples"),
                            temperature=float(fs.get("temperature", 0.0)))
    if name == "self_refine":
        sr = settings.get("self_refine", {})
        return run_self_refine(planner, problem_nl, temperature=float(sr.get("temperature", 0.2)))
    if name == "deliberate":
        dl = settings.get("deliberate", {})
        return run_deliberate(planner, problem_nl,
                              drafts=int(dl.get("drafts", 3)),
                              temperature=float(dl.get("temperature", 0.5)))
    raise ValueError(f"Unknown strategy: {name}")

# --- Back-compat alias (some callers may still import this) ---
def run_direct(*, name: str, planner, judge, problem_nl: str, settings=None) -> str:
    """Deprecated alias for run_strategy to preserve older callers."""
    return run_strategy(name, planner=planner, judge=judge, problem_nl=problem_nl, settings=settings or {})

__all__ = [
    "run_strategy",
    "run_direct",
    "run_base",
    "run_cot",
    "run_ltm",
    "run_cot_self_consistency",
    "run_few_shot",
    "run_self_refine",
    "run_deliberate",
]