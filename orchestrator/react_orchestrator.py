# ruff: noqa
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import dspy
import mlflow
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# -----------------------------------------------------------------------------
# Import your backend (we support both "new names" and older *_nl names)
# -----------------------------------------------------------------------------

# This orchestrator is intentionally OUTSIDE green_agent, but imports it.
# Adjust import only if your package layout differs.
from green_agent import tools_backend as tb  # noqa: E402


def _pick_attr(mod: Any, *names: str) -> Callable[..., Any]:
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"None of these exist on {mod.__name__}: {names}")


# Prefer your "agent-facing" names if present; otherwise fall back to *_nl.
RESET_EPISODE = _pick_attr(tb, "reset_episode", "reset_episode_nl")
UNDO = _pick_attr(tb, "undo", "undo_nl")

# These may be either context-free or require (domain, index, ...)
GET_TASK_OVERVIEW = _pick_attr(tb, "get_task_overview", "get_task_overview_nl")
LIST_OBJECTS = _pick_attr(tb, "list_objects", "list_objects_nl")
DESCRIBE_OBJECT = _pick_attr(tb, "describe_object", "describe_object_nl")

# Naming drift: list_actions vs list_actions_nl vs list_action_types_nl
LIST_ACTIONS = _pick_attr(tb, "list_actions", "list_actions_nl", "list_action_types_nl")

# describe_action is stable in your file
DESCRIBE_ACTION = _pick_attr(tb, "describe_action")

# Stateful tools (usually context-free)
GET_STATE = _pick_attr(tb, "get_state", "get_state_nl")
ACT = _pick_attr(tb, "act", "act_nl")
GET_HISTORY = _pick_attr(tb, "get_history", "get_history_nl")

# Submit should ONLY validate the current episode prefix (no text submission).
SUBMIT = _pick_attr(tb, "submit", "submit_episode", "submit_episode_nl")


# -----------------------------------------------------------------------------
# Repro / Git helpers
# -----------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def get_git_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        meta["git_commit"] = _run(["git", "rev-parse", "HEAD"])
    except Exception:
        meta["git_commit"] = "unknown"

    try:
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        staged_dirty = subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0
        meta["git_dirty"] = bool(dirty or staged_dirty)
    except Exception:
        meta["git_dirty"] = None

    # Optional: store patch if dirty (useful for exact repro)
    if meta.get("git_dirty"):
        try:
            meta["git_diff"] = _run(["git", "diff"])
            meta["git_diff_cached"] = _run(["git", "diff", "--cached"])
        except Exception:
            pass

    return meta


# -----------------------------------------------------------------------------
# Domain/problem discovery
# -----------------------------------------------------------------------------

_PROBLEM_RE = re.compile(r"^problem(\d+)\.pddl$", re.IGNORECASE)


def discover_domains(examples_dir: Path) -> list[str]:
    out: list[str] = []
    if not examples_dir.exists():
        return out
    for p in sorted(examples_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        if (p / "domain.pddl").exists() and (p / "prompts.json").exists() and (p / "problems_pddl").exists():
            out.append(p.name)
    return out


def discover_problem_indices(examples_dir: Path, domain: str) -> list[int]:
    probs_dir = examples_dir / domain / "problems_pddl"
    if not probs_dir.exists():
        return []
    idxs: list[int] = []
    for f in probs_dir.iterdir():
        if not f.is_file():
            continue
        m = _PROBLEM_RE.match(f.name)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))


# -----------------------------------------------------------------------------
# Tool wrappers (agent never passes domain/index)
# -----------------------------------------------------------------------------

class InvalidActionError(RuntimeError):
    pass


def _call_flexible(fn: Callable[..., Any], *preferred_args: Any, **preferred_kwargs: Any) -> Any:
    """
    Calls `fn` with the provided args if possible; if that fails due to signature mismatch,
    retries with progressively fewer context args.
    """
    try:
        return fn(*preferred_args, **preferred_kwargs)
    except TypeError:
        # Retry without any "context" args; common if you moved to fully stateful tools.
        return fn(**preferred_kwargs) if preferred_kwargs else fn()


@dataclass
class EpisodeTools:
    domain: str
    index: int
    strict_invalid_action: bool = True

    def reset(self, *, val_path: str | None = None, tolerance: float = 0.001) -> str:
        return _call_flexible(RESET_EPISODE, self.domain, self.index, val_path=val_path, tolerance=tolerance)

    # --- Agent-facing tools (names unchanged) ---

    def get_task_overview(self) -> str:
        """Signature: -> overview: str. Return the task description + initial situation + goal for this episode."""
        return _call_flexible(GET_TASK_OVERVIEW, self.domain, self.index)

    def list_objects(self, kind: str | None = None) -> str:
        """Signature: kind: str | None -> objects: str. List episode objects, optionally filtered by type/kind."""
        return _call_flexible(LIST_OBJECTS, self.domain, self.index, kind=kind)

    def describe_object(self, name: str) -> str:
        """Signature: name: str -> description: str. Describe one object by exact name."""
        return _call_flexible(DESCRIBE_OBJECT, self.domain, self.index, name)

    def list_actions(self) -> str:
        """Signature: -> actions: str. List available grounded action *types* and their parameters/preconditions/effects."""
        # Some variants are domain-only.
        try:
            return _call_flexible(LIST_ACTIONS, self.domain)
        except TypeError:
            return _call_flexible(LIST_ACTIONS, self.domain, self.index)

    def describe_action(self, action_name: str) -> str:
        """Signature: action_name: str -> description: str. Explain an action schema by name."""
        return _call_flexible(DESCRIBE_ACTION, self.domain, action_name)

    def get_state(self, max_facts: int = 200) -> str:
        """Signature: max_facts: int -> state: str. Show the current simulated state (debug/inspection)."""
        return _call_flexible(GET_STATE, max_facts=max_facts)

    def get_history(self) -> str:
        """Signature: -> history: str. Return the executed action history so far (the current plan prefix)."""
        return _call_flexible(GET_HISTORY)

    def act(self, step_text: str) -> str:
        """
        Signature: step_text: str -> result: str.
        Execute exactly ONE action (natural language / PDDL-ish). If invalid, the EPISODE ENDS immediately.
        """
        out = _call_flexible(ACT, step_text)
        if self.strict_invalid_action and isinstance(out, str) and out.strip().startswith("Executed: NO"):
            raise InvalidActionError(out)
        return out

    def submit(self) -> str:
        """
        Signature: -> verdict: str.
        Validate the current episode prefix as a full plan (goal satisfaction enforced).
        """
        return _call_flexible(SUBMIT)

    def undo(self, to_step: int) -> str:
        """Signature: to_step: int -> result: str. Revert history to the given step index (0 clears)."""
        return _call_flexible(UNDO, to_step)


# -----------------------------------------------------------------------------
# DSPy ReAct program
# -----------------------------------------------------------------------------

class PlannerSig(dspy.Signature):
    """
    You are solving an interactive planning task via tools.

    Rules:
    - Start by calling get_task_overview() and list_actions().
    - Propose ONE grounded action at a time using act(step_text).
    - If act fails (invalid action), the episode ends immediately.
    - When you believe the goal is achieved, call submit() once.
    - If submit() says Accepted: YES, return the final plan using get_history().
    - Do NOT invent actions; only use those shown by list_actions()/describe_action().
    """

    objective: str = dspy.InputField(desc="A short instruction like 'Solve the current episode'.")
    final_plan: str = dspy.OutputField(desc="Return the final plan (from get_history()) or a short failure note.")


class ReActPlanner(dspy.Module):
    def __init__(self, tools: list[Callable[..., Any]], max_iters: int = 40):
        super().__init__()
        # DSPy ReAct: signature + tools + iteration budget :contentReference[oaicite:3]{index=3}
        self.react = dspy.ReAct(PlannerSig, tools=tools, max_iters=max_iters)

    def forward(self, objective: str) -> dspy.Prediction:
        return self.react(objective=objective)


# -----------------------------------------------------------------------------
# MLflow + DSPy tracing setup
# -----------------------------------------------------------------------------

def setup_mlflow(tracking_uri: str | None, experiment: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment)

    # DSPy <-> MLflow tracing integration: enable autolog for DSPy calls :contentReference[oaicite:4]{index=4}
    if hasattr(mlflow, "dspy") and hasattr(mlflow.dspy, "autolog"):
        mlflow.dspy.autolog()
    else:
        raise RuntimeError(
            "Your MLflow install does not expose mlflow.dspy.autolog(). "
            "Upgrade MLflow to a version that supports DSPy tracing (see MLflow tracing docs)."
        )


def _parse_submit_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"accepted": None, "raw": text}
    if not isinstance(text, str):
        return out
    m = re.search(r"Accepted:\s*(YES|NO)", text)
    if m:
        out["accepted"] = (m.group(1) == "YES")
    m = re.search(r"Plan length:\s*(\d+)", text)
    if m:
        out["plan_length"] = int(m.group(1))
    m = re.search(r"Plan cost/value:\s*([0-9.+-]+)", text)
    if m:
        try:
            out["plan_cost"] = float(m.group(1))
        except Exception:
            pass
    return out


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-dir", default="examples")
    ap.add_argument("--domains", nargs="*", default=None, help="If omitted, auto-discover all domains in examples/.")
    ap.add_argument("--problems", nargs="*", type=int, default=None, help="If omitted, auto-discover per-domain problems.")
    ap.add_argument("--models", nargs="+", required=True, help="DSPy model IDs like: openai/gpt-4o-mini")
    ap.add_argument("--experiment", default="DSPy_ReAct_Planning")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--tolerance", type=float, default=0.001)
    ap.add_argument("--val-path", default=None)
    args = ap.parse_args()

    console = Console()
    examples_dir = Path(args.examples_dir).resolve()

    # Repro knobs
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")

    git_meta = get_git_metadata()

    setup_mlflow(args.tracking_uri, args.experiment)

    domains = args.domains or discover_domains(examples_dir)
    if not domains:
        raise SystemExit(f"No domains found under: {examples_dir}")

    # Build episode list (domain x problems)
    episodes: list[tuple[str, int]] = []
    for d in domains:
        idxs = args.problems or discover_problem_indices(examples_dir, d)
        for i in idxs:
            episodes.append((d, i))

    if not episodes:
        raise SystemExit("No (domain, problem) episodes discovered.")

    total_runs = len(args.models) * len(episodes)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    overall_task = progress.add_task("ALL runs", total=total_runs)

    with progress:
        # Parent MLflow run (sweep)
        with mlflow.start_run(run_name=f"sweep_{git_meta.get('git_commit','unknown')[:8]}"):
            mlflow.set_tags(
                {
                    "git_commit": git_meta.get("git_commit", "unknown"),
                    "git_dirty": str(git_meta.get("git_dirty")),
                    "runner": "orchestrators/dspy_react_sweep.py",
                    "dspy_program": "ReActPlanner",
                }
            )
            mlflow.log_params(
                {
                    "models": json.dumps(args.models),
                    "domains": json.dumps(domains),
                    "episodes_n": len(episodes),
                    "max_iters": args.max_iters,
                    "tolerance": args.tolerance,
                    "val_path": args.val_path or "",
                }
            )

            if git_meta.get("git_dirty") and git_meta.get("git_diff"):
                mlflow.log_text(git_meta["git_diff"], "repro/git_diff.patch")
            if git_meta.get("git_dirty") and git_meta.get("git_diff_cached"):
                mlflow.log_text(git_meta["git_diff_cached"], "repro/git_diff_cached.patch")

            # Sweep loops
            for model_id in args.models:
                # Configure DSPy LM (temperature forced to 0 for reproducibility; DSPy uses dspy.LM + dspy.configure) :contentReference[oaicite:5]{index=5}
                lm = dspy.LM(model_id, temperature=0)
                dspy.configure(lm=lm)

                model_task = progress.add_task(f"Model: {model_id}", total=len(episodes))

                for (domain, index) in episodes:
                    run_name = f"{model_id.replace('/','_')}__{domain}__p{index:02d}"
                    episode_task = progress.add_task(f"{domain} p{index:02d}", total=1)

                    start = time.time()
                    tools = EpisodeTools(domain=domain, index=index, strict_invalid_action=True)

                    # Reset episode context once; agent uses context-free tools thereafter.
                    reset_msg = tools.reset(val_path=args.val_path, tolerance=args.tolerance)

                    # Prepare the tool callables (names unchanged)
                    tool_fns: list[Callable[..., Any]] = [
                        tools.get_task_overview,
                        tools.list_actions,
                        tools.list_objects,
                        tools.describe_action,
                        tools.describe_object,
                        tools.get_state,
                        tools.get_history,
                        tools.act,
                        tools.submit,
                    ]

                    planner = ReActPlanner(tools=tool_fns, max_iters=args.max_iters)

                    accepted = False
                    invalid_action = None
                    final_plan = ""
                    submit_text = ""
                    try:
                        # Nested run per episode for clean MLflow UI grouping
                        with mlflow.start_run(run_name=run_name, nested=True):
                            mlflow.set_tags(
                                {
                                    "model": model_id,
                                    "domain": domain,
                                    "problem_index": str(index),
                                }
                            )

                            mlflow.log_text(reset_msg, "episode/reset.txt")

                            pred = planner(objective="Solve the current episode.")
                            final_plan = getattr(pred, "final_plan", "") if pred is not None else ""

                            # Always validate at the end (even if agent forgot)
                            submit_text = tools.submit()
                            parsed = _parse_submit_text(submit_text)
                            accepted = bool(parsed.get("accepted"))

                            # Log core outputs
                            mlflow.log_text(final_plan or "", "episode/final_plan.txt")
                            mlflow.log_text(tools.get_history(), "episode/history.txt")
                            mlflow.log_text(submit_text, "episode/submit.txt")

                            # Metrics
                            mlflow.log_metric("accepted", 1.0 if accepted else 0.0)
                            if parsed.get("plan_length") is not None:
                                mlflow.log_metric("plan_length", float(parsed["plan_length"]))
                            if parsed.get("plan_cost") is not None:
                                mlflow.log_metric("plan_cost", float(parsed["plan_cost"]))

                    except InvalidActionError as e:
                        invalid_action = str(e)
                        # Create a nested run even on invalid action for visibility
                        with mlflow.start_run(run_name=run_name, nested=True):
                            mlflow.set_tags(
                                {
                                    "model": model_id,
                                    "domain": domain,
                                    "problem_index": str(index),
                                    "terminated": "invalid_action",
                                }
                            )
                            mlflow.log_text(reset_msg, "episode/reset.txt")
                            mlflow.log_text(invalid_action, "episode/invalid_action.txt")
                            mlflow.log_text(tools.get_history(), "episode/history.txt")
                            mlflow.log_metric("accepted", 0.0)

                    finally:
                        dur = time.time() - start
                        # Parent run metrics aggregation can be added later if you want.
                        progress.update(episode_task, advance=1)
                        progress.remove_task(episode_task)
                        progress.update(model_task, advance=1)
                        progress.update(overall_task, advance=1)

                        # Lightweight console status line
                        status = "OK" if accepted else ("INVALID" if invalid_action else "FAIL")
                        console.print(f"[{status}] {model_id} • {domain} • p{index:02d} • {dur:.2f}s")

                progress.remove_task(model_task)


if __name__ == "__main__":
    main()
