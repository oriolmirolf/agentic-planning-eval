# ruff: noqa
from __future__ import annotations

import argparse
import json
import os
import time
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import dspy
import mlflow
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn  # <--- Added this missing import
)
from rich import box

from green_agent import tools_backend as tb

# -----------------------------------------------------------------------------
# 1. Backend Wrappers
# -----------------------------------------------------------------------------

def _pick_attr(mod: Any, *names: str) -> Callable[..., Any]:
    for n in names:
        if hasattr(mod, n): return getattr(mod, n)
    raise AttributeError(f"None of these exist on {mod.__name__}: {names}")

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
# 2. Episode Tools (Agent Interface)
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
                if not is_syntax: raise AgentDeath(out)
        return out

class AgentDeath(Exception): pass

# -----------------------------------------------------------------------------
# 3. DSPy Modules
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
    def forward(self, status_updater=None, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", getattr(self, "max_iters", 5))

        for idx in range(max_iters):
            if status_updater: status_updater(f"Step {idx+1}/{max_iters}")

            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError: break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            tool_name = pred.next_tool_name.strip()
            if tool_name.lower() in ["submit", "finish", "submit_episode"]: tool_name = "submit"
            elif tool_name not in self.tools and tool_name.lower() in self.tools: tool_name = tool_name.lower()

            if status_updater: status_updater(f"Step {idx+1}: {tool_name}")

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
# 4. Worker Logic
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    model_id: str
    domain: str
    index: int
    optimal_cost: int
    max_iters: int

@dataclass
class RunResult:
    config: RunConfig
    success: bool
    valid: bool
    score: float
    steps: int
    duration: float
    metrics: Dict[str, float]
    history: str
    final_plan: str
    submit_text: str
    trajectory: dict | None
    llm_history: List[dict] | None
    error_msg: str = ""

def _parse_submit_text(text: str) -> dict[str, Any]:
    out = {"accepted": None, "raw": text}
    if isinstance(text, str):
        m = re.search(r"Accepted:\s*(YES|NO)", text)
        if m: out["accepted"] = (m.group(1) == "YES")
        m = re.search(r"Plan length:\s*(\d+)", text)
        if m: out["plan_length"] = int(m.group(1))
    return out

def count_history_steps(history_text: str) -> int:
    if not history_text or "No actions executed yet" in history_text: return 0
    return len([line for line in history_text.splitlines() if line.strip()])

def run_one_episode(cfg: RunConfig, status_dict: Dict) -> RunResult:
    pid = os.getpid()
    def update_status(msg: str):
        status_dict[pid] = f"[bold blue]{cfg.domain} p{cfg.index:02d}[/]: {msg}"

    update_status("Initializing...")
    lm = dspy.LM(cfg.model_id, temperature=1, cache=False)
    lm.history = [] 
    dspy.configure(lm=lm)

    start_time = time.time()
    tools = EpisodeTools(domain=cfg.domain, index=cfg.index, strict_invalid_action=True)
    tools.reset()

    tool_fns = [
        tools.get_task_overview, tools.list_actions, tools.list_objects,
        tools.describe_action, tools.describe_object, tools.get_state,
        tools.get_history, tools.act, tools.submit,
    ]
    planner = StrictReAct(PlannerSig, tools=tool_fns, max_iters=cfg.max_iters)

    is_success = False
    is_valid = True
    error_msg = ""
    submit_text = ""
    final_plan = ""
    traj_data = None

    try:
        pred = planner(objective="Solve the current episode.", status_updater=update_status)
        if hasattr(pred, "trajectory"):
            traj_data = pred.trajectory
            if pred.trajectory.get("fatal_error_log"):
                raise AgentDeath(pred.trajectory["fatal_error_log"])

        update_status("Verifying...")
        final_plan = getattr(pred, "final_plan", "")
        if "Accepted: YES" in final_plan or "Success" in final_plan:
            submit_text = final_plan
        else:
            hist_check = GET_HISTORY()
            submit_text = hist_check if "Accepted: YES" in hist_check else SUBMIT()

        parsed = _parse_submit_text(submit_text)
        is_success = bool(parsed.get("accepted"))
        steps_taken = parsed.get("plan_length") or count_history_steps(GET_HISTORY())

    except AgentDeath as e:
        is_valid = False
        error_msg = str(e)
        steps_taken = count_history_steps(GET_HISTORY())
        if hasattr(pred, "trajectory"): traj_data = pred.trajectory

    except Exception as e:
        is_valid = False
        error_msg = str(e)
        steps_taken = count_history_steps(GET_HISTORY())

    duration = time.time() - start_time
    score = 0.0 if not is_success or steps_taken == 0 else min(1.0, cfg.optimal_cost / steps_taken)

    metrics = {
        "wall_time": duration,
        "invalid_actions": float(tools.invalid_action_count),
    }
    for k, v in tools.tool_usage.items(): metrics[f"tool_usage_{k}"] = float(v)

    if pid in status_dict: del status_dict[pid]

    return RunResult(cfg, is_success, is_valid, score, steps_taken, duration, metrics, GET_HISTORY(), final_plan, submit_text, traj_data, lm.history, error_msg)

# -----------------------------------------------------------------------------
# 5. Main Orchestrator
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
    ap.add_argument("--experiment", default="DSPy_ReAct_Parallel")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    console = Console()
    examples_dir = Path(args.examples_dir).resolve()
    
    domains = args.domains or discover_domains(examples_dir)
    episodes = []
    for d in domains:
        idxs = args.problems or discover_problem_indices(examples_dir, d)
        for i in idxs: episodes.append((d, i))

    if not episodes: raise SystemExit("No episodes found.")

    tasks = []
    for model_id in args.models:
        for d, i in episodes:
            costs = load_costs(examples_dir / d)
            tasks.append(RunConfig(model_id, d, i, costs.get(i, 0), args.max_iters))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # UI Components
    progress = Progress(
        SpinnerColumn(), 
        TextColumn("[bold]{task.description}"), 
        BarColumn(), 
        TaskProgressColumn(), 
        TimeElapsedColumn(), 
        TextColumn("[bold cyan]OK: {task.fields[success_count]}[/] | [bold red]ERR: {task.fields[fail_count]}[/]"),
    )
    task_tracker = progress.add_task(f"Total: {len(tasks)}", total=len(tasks), success_count=0, fail_count=0)
    results_log = []

    def get_layout(worker_status):
        w_table = Table(title="Active Workers", box=box.SIMPLE, show_header=False, width=80)
        w_table.add_column("Worker", style="dim", width=10)
        w_table.add_column("Status")
        
        if not worker_status:
            w_table.add_row("-", "[dim]Idle[/]")
        else:
            # Sort by PID for stable display
            for pid in sorted(worker_status.keys()):
                w_table.add_row(f"PID {pid}", worker_status[pid])

        r_table = Table(title="Recent Results", box=box.ROUNDED, width=80)
        r_table.add_column("Domain", style="bold")
        r_table.add_column("Result")
        r_table.add_column("Score", justify="right")
        r_table.add_column("Time", justify="right")
        
        for r in results_log[-5:]:
            r_table.add_row(*r)

        return Group(
            Panel(progress, border_style="blue"),
            w_table,
            r_table
        )

    success_count = 0
    fail_count = 0
    
    with multiprocessing.Manager() as manager:
        status_dict = manager.dict()
        
        with Live(get_layout(status_dict), refresh_per_second=4, console=console) as live:
            
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {executor.submit(run_one_episode, t, status_dict): t for t in tasks}
                futures = list(future_to_task.keys())
                
                while futures:
                    # Manually poll futures to keep UI alive
                    done_futures = [f for f in futures if f.done()]
                    
                    live.update(get_layout(status_dict))
                    
                    for f in done_futures:
                        futures.remove(f)
                        t = future_to_task[f]
                        try:
                            res = f.result()
                            if res.success: success_count += 1
                            else: fail_count += 1
                            progress.update(task_tracker, advance=1, success_count=success_count, fail_count=fail_count)

                            run_name = f"{t.model_id.replace('/','_')}__{t.domain}__p{t.index:02d}__{int(time.time())}"
                            with mlflow.start_run(run_name=run_name):
                                mlflow.set_tags({
                                    "model": t.model_id, "domain": t.domain, "problem_index": str(t.index),
                                    "terminated": "crash" if res.error_msg else ("invalid" if not res.valid else "success")
                                })
                                mlflow.log_metric("is_success", 1.0 if res.success else 0.0)
                                mlflow.log_metric("steps", float(res.steps))
                                mlflow.log_metric("score", res.score)
                                for k, v in res.metrics.items(): mlflow.log_metric(k, v)
                                mlflow.log_text(res.history, "episode/history.txt")
                                mlflow.log_text(res.final_plan, "episode/final_plan.txt")
                                if res.trajectory: mlflow.log_dict(res.trajectory, "episode/trajectory.json")
                                if res.llm_history: mlflow.log_dict(res.llm_history, "episode/llm_raw_trace.json")

                            color = "green" if res.success else "red"
                            icon = "✅" if res.success else ("💀" if not res.valid else "❌")
                            results_log.append((f"{t.domain} p{t.index:02d}", f"[{color}]{icon}[/]", f"{res.score:.2f}", f"{res.duration:.1f}s"))

                        except Exception as exc:
                            results_log.append((f"{t.domain} p{t.index:02d}", "[bold red]CRASH[/]", "0.0", "-"))
                            fail_count += 1
                            progress.update(task_tracker, advance=1, success_count=success_count, fail_count=fail_count)
                    
                    time.sleep(0.1)

if __name__ == "__main__":
    main()