from __future__ import annotations

import dspy

# === 1) Task decomposition (hierarchical planning) =========================

class DecomposeTaskSignature(dspy.Signature):
    """
    Decompose a complex planning problem into subgoals for an LLM planning agent.
    Mirrors the task-decomposer / planner role in ReAct variants, Plan-and-Solve,
    CoReaAgents-style architectures, etc.
    """
    problem_nl: str = dspy.InputField(
        desc="Natural language description of the planning problem and constraints."
    )
    subgoals: str = dspy.OutputField(
        desc="Numbered or bulleted list of subgoals / phases that make solving the problem easier."
    )


_decompose_task = dspy.ChainOfThought(DecomposeTaskSignature)


def llm_decompose_task(problem_nl: str) -> str:
    """
    LLM-backed tool:
    Given a planning problem, propose a set of subgoals / phases.
    """
    pred = _decompose_task(problem_nl=problem_nl)
    return pred.subgoals.strip()


# === 2) Plan outline generator ============================================

class PlanOutlineSignature(dspy.Signature):
    """
    Turn a problem description + (optional) subgoals into a high-level plan outline.
    Classic planner step before we fill in low-level actions.
    """
    problem_nl: str = dspy.InputField(
        desc="Problem description the plan must solve."
    )
    subgoals: str = dspy.InputField(
        desc="Optional subgoals / phases to respect. Can be empty if not known."
    )
    plan_outline: str = dspy.OutputField(
        desc="High-level plan: coarse steps in execution order, one per line."
    )


_plan_outline = dspy.ChainOfThought(PlanOutlineSignature)


def llm_generate_plan_outline(problem_nl: str, subgoals: str | None = "") -> str:
    """
    LLM-backed tool:
    Emit a high-level plan outline given the problem (and optionally subgoals).
    """
    pred = _plan_outline(problem_nl=problem_nl, subgoals=subgoals or "")
    return pred.plan_outline.strip()


# === 3) Plan critic / reflector ===========================================

class CritiquePlanSignature(dspy.Signature):
    """
    Self-critique / reflection on a candidate plan.
    Reflects Reflexion-style agents, CoReaAgents' Reflect Agent,
    and general self-check patterns in agentic systems.
    """
    problem_nl: str = dspy.InputField(
        desc="Original planning problem and constraints."
    )
    plan: str = dspy.InputField(
        desc="Candidate plan to evaluate."
    )
    critique: str = dspy.OutputField(
        desc="Short critique: missing steps, constraint violations, obvious errors."
    )


_critique_plan = dspy.ChainOfThought(CritiquePlanSignature)


def llm_critique_plan(problem_nl: str, plan: str) -> str:
    """
    LLM-backed tool:
    Inspect a candidate plan and return a critique / list of issues.
    """
    pred = _critique_plan(problem_nl=problem_nl, plan=plan)
    return pred.critique.strip()


# === 4) Plan refiner / repair =============================================

class RefinePlanSignature(dspy.Signature):
    """
    Refine or repair a plan given a critique.
    This is the self-correction pattern used all over modern agentic setups.
    """
    problem_nl: str = dspy.InputField(
        desc="Planning problem and constraints."
    )
    plan: str = dspy.InputField(
        desc="Current candidate plan."
    )
    critique: str = dspy.InputField(
        desc="Known issues with the current plan (may be empty to let the LM self-critique)."
    )
    improved_plan: str = dspy.OutputField(
        desc="Rewritten plan that addresses the critique and better fits the problem."
    )


_refine_plan = dspy.ChainOfThought(RefinePlanSignature)


def llm_refine_plan(problem_nl: str, plan: str, critique: str | None = "") -> str:
    """
    LLM-backed tool:
    Repair / refine a plan, optionally conditioned on an explicit critique.
    """
    pred = _refine_plan(problem_nl=problem_nl, plan=plan, critique=critique or "")
    return pred.improved_plan.strip()


# === 5) Context summarizer =================================================

class SummarizeContextSignature(dspy.Signature):
    """
    Summarize long problem or interaction context so the planning agent can
    keep a compact view of what matters. Usual memory / condensation helper
    in long-horizon agents.
    """
    context: str = dspy.InputField(
        desc="Free-form context: prior steps, tool outputs, notes, etc."
    )
    summary: str = dspy.OutputField(
        desc="Short summary that preserves key constraints, goals, and decisions."
    )


_summarize_context = dspy.ChainOfThought(SummarizeContextSignature)


def llm_summarize_context(context: str) -> str:
    """
    LLM-backed tool:
    Summarize arbitrary context into a compressed, planning-relevant state.
    """
    pred = _summarize_context(context=context)
    return pred.summary.strip()


# Convenience: list of all tool callables this module defines.
AGENTIC_PLANNING_TOOLS = [
    llm_decompose_task,
    llm_generate_plan_outline,
    llm_critique_plan,
    llm_refine_plan,
    llm_summarize_context,
]
