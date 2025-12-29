# ruff: noqa
from __future__ import annotations

import argparse
import json
import os
import re  # <--- FIXED: Added missing import
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import dspy
import mlflow
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from green_agent import tools_backend as tb

# -----------------------------------------------------------------------------
# 1. Backend Wrappers (Raw functions for verification)
# -----------------------------------------------------------------------------

def _pick_attr(mod: Any, *names: str) -> Callable[..., Any]:
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"None of these exist on {mod.__name__}: {names}")

# Raw backend functions (Main process uses these for verification)
RESET_EPISODE = _pick_attr(tb, "reset_episode", "reset_episode_nl")
GET_TASK_OVERVIEW = _pick_attr(tb, "get_task_overview", "get_task_overview_nl")
LIST_OBJECTS = _pick_attr(tb, "list_objects", "list_objects_nl")
DESCRIBE_OBJECT = _pick_attr(tb, "describe_object", "describe_object_nl")
LIST_ACTIONS = _pick_attr(tb, "list_actions", "list_actions_nl", "list_action_types_nl")
DESCRIBE_ACTION = _pick_attr(tb, "describe_action")
GET_STATE = _pick_attr(tb, "get_state", "get_state_nl")
ACT = _pick_attr(tb, "act", "act_nl")
GET_HISTORY = _pick_attr(tb, "get_history", "get_history_nl")
SUBMIT = _pick_attr(tb, "submit", "submit_episode", "submit_episode_nl")
UNDO = _pick_attr(tb, "undo", "undo_nl")

# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def get_git_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        meta["git_commit"] = _run(["git", "rev-parse", "HEAD"])
    except Exception:
        meta["git_commit"] = "unknown"
    return meta

def count_history_steps(history_text: str) -> int:
    if not history_text or "No actions executed yet" in history_text:
        return 0
    return len([line for line in history_text.splitlines() if line.strip()])

def _parse_submit_text(text: str) -> dict[str, Any]:
    out = {"accepted": None, "raw": text}
    if isinstance(text, str):
        m = re.search(r"Accepted:\s*(YES|NO)", text)
        if m: out["accepted"] = (m.group(1) == "YES")
        m = re.search(r"Plan length:\s*(\d+)", text)
        if m: out["plan_length"] = int(m.group(1))
    return out

# -----------------------------------------------------------------------------
# 3. Episode Tools (Agent Interface)
# -----------------------------------------------------------------------------

@dataclass
class EpisodeTools:
    domain: str
    index: int
    strict_invalid_action: bool = True
    
    invalid_action_count: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)

    def _track(self, name: str):
        self.tool_usage[name] = self.tool_usage.get(name, 0) + 1

    def _call(self, fn: Callable, *args, **kwargs):
        try: return fn(*args, **kwargs)
        except TypeError: return fn(**kwargs) if kwargs else fn()

    def reset(self, *, val_path: str = None, tolerance: float = 0.001) -> str:
        # Internal reset, not tracked
        return self._call(RESET_EPISODE, self.domain, self.index, val_path=val_path, tolerance=tolerance)

    def get_task_overview(self) -> str: 
        self._track("get_task_overview")
        return self._call(GET_TASK_OVERVIEW, self.domain, self.index)

    def list_objects(self, kind: str = None) -> str: 
        self._track("list_objects")
        return self._call(LIST_OBJECTS, self.domain, self.index, kind=kind)

    def describe_object(self, name: str) -> str: 
        self._track("describe_object")
        return self._call(DESCRIBE_OBJECT, self.domain, self.index, name)
    
    def list_actions(self) -> str:
        self._track("list_actions")
        try: return self._call(LIST_ACTIONS, self.domain)
        except TypeError: return self._call(LIST_ACTIONS, self.domain, self.index)

    def describe_action(self, action_name: str) -> str: 
        self._track("describe_action")
        return self._call(DESCRIBE_ACTION, self.domain, action_name)

    def get_state(self, max_facts: int = 200) -> str: 
        self._track("get_state")
        return self._call(GET_STATE, max_facts=max_facts)

    def get_history(self) -> str: 
        self._track("get_history")
        return self._call(GET_HISTORY)

    def submit(self) -> str: 
        self._track("submit")
        return self._call(SUBMIT)

    def undo(self, to_step: int) -> str: 
        self._track("undo")
        return self._call(UNDO, to_step)

    def act(self, step_text: str) -> str:
        self._track("act")
        out = self._call(ACT, step_text)
        if isinstance(out, str) and "Executed: NO" in out:
            self.invalid_action_count += 1
            if self.strict_invalid_action:
                lower = out.lower()
                is_syntax = any(x in lower for x in ["arg", "expect", "found", "got", "parameter", "signature", "missing"])
                if not is_syntax:
                    raise AgentDeath(out)
        return out

class AgentDeath(Exception):
    pass

# -----------------------------------------------------------------------------
# 4. DSPy Modules
# -----------------------------------------------------------------------------

class PlannerSig(dspy.Signature):
    """
    You are solving an interactive planning task via tools.
    Strategy:
    1. INSPECT: Call get_task_overview() and list_actions().
    2. PLAN: Think about the sequence of actions needed.
    3. ACT: Propose ONE grounded action at a time using act(step_text).
    4. VERIFY: Call get_state() if needed.
    5. TERMINATE: Call submit() when done.
    """
    objective: str = dspy.InputField(desc="A short instruction.")
    final_plan: str = dspy.OutputField(desc="Return the final plan (from get_history()) or a failure note.")

class StrictReAct(dspy.ReAct):
    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", getattr(self, "max_iters", 5))

        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
            except ValueError: break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            tool_name = pred.next_tool_name.strip()
            if tool_name.lower() in ["submit", "finish", "submit_episode"]:
                tool_name = "submit"
            elif tool_name not in self.tools and tool_name.lower() in self.tools:
                tool_name = tool_name.lower()

            if tool_name == "submit":
                try:
                    tool_fn = self.tools[tool_name]
                    observation = tool_fn(**pred.next_tool_args)
                    trajectory[f"observation_{idx}"] = observation
                    if "Accepted: YES" in str(observation):
                        return dspy.Prediction(trajectory=trajectory, final_plan=str(observation))
                except Exception as err:
                    trajectory[f"observation_{idx}"] = f"Error: {err}"
                    return dspy.Prediction(trajectory=trajectory, final_plan=f"Submit Failed: {err}")
                trajectory["stop_reason"] = "submit_break"
                break

            try:
                if tool_name not in self.tools:
                     observation = f"Error: Tool '{tool_name}' not found."
                else:
                    tool_fn = self.tools[tool_name]
                    observation = tool_fn(**pred.next_tool_args)
                trajectory[f"observation_{idx}"] = observation
                if "Accepted: YES" in str(observation) or "Success" in str(observation):
                    return dspy.Prediction(trajectory=trajectory, final_plan=str(observation))
            except AgentDeath as e:
                trajectory[f"observation_{idx}"] = f"FATAL DOMAIN ERROR: {e}"
                trajectory["fatal_error_log"] = str(e)
                return dspy.Prediction(trajectory=trajectory, final_plan=f"Failed: {e}")
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {tool_name}: {err}"

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

# -----------------------------------------------------------------------------
# 5. Main Orchestrator (Sequential)
# -----------------------------------------------------------------------------

def discover_domains(examples_dir: Path) -> list[str]:
    out = []
    if examples_dir.exists():
        for p in sorted(examples_dir.iterdir()):
            if p.is_dir() and not p.name.startswith((".", "_")) and (p / "domain.pddl").exists():
                out.append(p.name)
    return out

def discover_problem_indices(examples_dir: Path, domain: str) -> list[int]:
    probs_dir = examples_dir / domain / "problems_pddl"
    if not probs_dir.exists(): return []
    idxs = []
    _PROBLEM_RE = re.compile(r"^problem(\d+)\.pddl$", re.IGNORECASE)
    for f in probs_dir.iterdir():
        if f.is_file():
            m = _PROBLEM_RE.match(f.name)
            if m: idxs.append(int(m.group(1)))
    return sorted(set(idxs))

def load_costs(domain_dir: Path) -> dict[int, int]:
    path = domain_dir / "prompts.json"
    costs = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
            for p in data.get("problems", []):
                pid = p.get("id", "")
                m = re.search(r"p(\d+)", pid, re.IGNORECASE)
                if m and "optimal_cost" in p: costs[int(m.group(1))] = int(p["optimal_cost"])
        except Exception: pass
    return costs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-dir", default="examples")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--problems", nargs="*", type=int, default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--experiment", default="DSPy_ReAct_Sequential")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    ap.add_argument("--max-iters", type=int, default=40)
    args = ap.parse_args()

    console = Console()
    examples_dir = Path(args.examples_dir).resolve()
    git_meta = get_git_metadata()
    
    domains = args.domains or discover_domains(examples_dir)
    episodes = []
    for d in domains:
        idxs = args.problems or discover_problem_indices(examples_dir, d)
        for i in idxs: episodes.append((d, i))

    if not episodes: raise SystemExit("No episodes found.")

    # Setup MLflow Autologging (Works in sequential!)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    if hasattr(mlflow, "dspy"):
        mlflow.dspy.autolog()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
        console=console
    )

    with progress:
        task_id = progress.add_task(f"Running {len(episodes) * len(args.models)} episodes", total=len(episodes) * len(args.models), status="")

        for model_id in args.models:
            # Re-init LM per model loop
            lm = dspy.LM(model_id, temperature=1, cache=False)
            dspy.configure(lm=lm)

            for domain, index in episodes:
                progress.update(task_id, description=f"Running {domain} p{index:02d}")
                
                costs = load_costs(examples_dir / domain)
                optimal_cost = costs.get(index, 0)
                
                # --- START EPISODE ---
                timestamp = int(time.time())
                run_name = f"{model_id.replace('/','_')}__{domain}__p{index:02d}__{timestamp}"
                
                start_time = time.time()
                
                # 1. Init Tools (Metrics enabled)
                tools = EpisodeTools(domain=domain, index=index, strict_invalid_action=True)
                reset_msg = tools.reset() # Reset backend state

                tool_fns = [
                    tools.get_task_overview, tools.list_actions, tools.list_objects,
                    tools.describe_action, tools.describe_object, tools.get_state,
                    tools.get_history, tools.act, tools.submit,
                ]
                
                planner = StrictReAct(PlannerSig, tools=tool_fns, max_iters=args.max_iters)

                is_success = False
                is_valid = True
                error_msg = ""
                submit_text = ""
                steps_taken = 0
                
                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({
                        "model": model_id,
                        "domain": domain,
                        "problem_index": str(index),
                        "git_commit": git_meta.get("git_commit", "unknown")
                    })
                    mlflow.log_text(reset_msg, "episode/reset.txt")

                    try:
                        # 2. Run Agent
                        pred = planner(objective="Solve the current episode.")
                        
                        if hasattr(pred, "trajectory") and pred.trajectory.get("fatal_error_log"):
                            raise AgentDeath(pred.trajectory["fatal_error_log"])

                        # 3. Verify
                        final_plan = getattr(pred, "final_plan", "")
                        if "Accepted: YES" in final_plan or "Success" in final_plan:
                            submit_text = final_plan
                        else:
                            # Use RAW backend to avoid metrics pollution
                            hist_check = GET_HISTORY()
                            submit_text = hist_check if "Accepted: YES" in hist_check else SUBMIT()

                        parsed = _parse_submit_text(submit_text)
                        is_success = bool(parsed.get("accepted"))
                        
                        if parsed.get("plan_length"):
                            steps_taken = parsed["plan_length"]
                        else:
                            steps_taken = count_history_steps(GET_HISTORY())

                        mlflow.log_text(final_plan, "episode/final_plan.txt")
                        mlflow.log_text(submit_text, "episode/submit.txt")
                        if hasattr(pred, "trajectory"):
                            mlflow.log_dict(pred.trajectory, "episode/trajectory.json")

                    except AgentDeath as e:
                        is_valid = False
                        error_msg = str(e)
                        mlflow.set_tag("terminated", "invalid_action")
                        mlflow.log_text(error_msg, "episode/invalid_action.txt")
                        steps_taken = count_history_steps(GET_HISTORY())
                        if hasattr(pred, "trajectory"): mlflow.log_dict(pred.trajectory, "episode/trajectory.json")

                    except Exception as e:
                        is_valid = False
                        error_msg = str(e)
                        mlflow.set_tag("terminated", "crash")
                        mlflow.log_text(error_msg, "episode/crash.txt")
                        steps_taken = count_history_steps(GET_HISTORY())

                    # 4. Metrics & Logging
                    duration = time.time() - start_time
                    score = 0.0
                    if is_success and steps_taken > 0:
                        score = optimal_cost / steps_taken
                        if score > 1.0: score = 1.0

                    mlflow.log_metric("is_valid", 1.0 if is_valid else 0.0)
                    mlflow.log_metric("is_success", 1.0 if is_success else 0.0)
                    mlflow.log_metric("steps_taken", float(steps_taken))
                    mlflow.log_metric("optimal_steps", float(optimal_cost))
                    mlflow.log_metric("score", score)
                    mlflow.log_metric("wall_time", duration)
                    mlflow.log_metric("invalid_actions", float(tools.invalid_action_count))
                    
                    for k, v in tools.tool_usage.items():
                        mlflow.log_metric(f"tool_usage_{k}", float(v))

                    mlflow.log_text(GET_HISTORY(), "episode/history.txt")

                    # Console Feedback
                    status_str = f"[bold green]OK {score:.2f}[/]" if is_success else f"[bold red]FAIL[/]"
                    if not is_valid: status_str = "[bold red]CRASH[/]"
                    progress.update(task_id, advance=1, status=f"Last: {domain} p{index} -> {status_str}")
                    
                    # Panel
                    grid = f"Steps: {steps_taken}/{optimal_cost} | Time: {duration:.1f}s | Score: {score:.2f}"
                    if error_msg: grid += f"\nError: {error_msg[:100]}"
                    console.print(Panel(grid, title=f"{domain} p{index:02d} - {status_str}", border_style="green" if is_success else "red"))

if __name__ == "__main__":
    main()