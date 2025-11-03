# green_agent/runner.py
from __future__ import annotations
import os, time, json
from typing import Optional, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .config import EvalConfig
from .plan_parser import extract_plan
from .metrics import compute_metrics

from purple_agent.openai_agent import OpenAIPurpleAgent
from purple_agent.a2a_agent import A2APurpleAgent


console = Console()

def load_text(path: Optional[str]) -> str:
    if not path: return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_purple(kind: str, *, model: Optional[str], a2a_url: Optional[str]):
    if kind == "openai":
        return OpenAIPurpleAgent(model=model)
    if kind == "a2a":
        if not a2a_url:
            raise SystemExit("Missing purple_url for 'a2a' purple.")
        return A2APurpleAgent(url=a2a_url)

def _make_run_dir(base_out: str, domain_path: str) -> str:
    example = Path(domain_path).parent.name or "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_out) / f"{example}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

def evaluate_once(cfg: EvalConfig) -> Dict[str, Any]:
    run_dir = _make_run_dir(cfg.out_dir, cfg.domain_path)
    problem_nl = load_text(cfg.prompt_path)

    purple = build_purple(cfg.purple_kind, model=cfg.openai_model, a2a_url=cfg.purple_url)

    t0 = time.time()
    plan_raw = purple.generate_plan(problem_nl=problem_nl)
    t1 = time.time()

    raw_path = os.path.join(run_dir, "purple_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(plan_raw)

    extracted = extract_plan(plan_raw)
    plan_txt = extracted.to_val_plan_text()
    plan_path = os.path.join(run_dir, "purple.plan")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_txt)

    flags = (*cfg.val_flags, "-t", str(cfg.tolerance))
    metrics = compute_metrics(domain=cfg.domain_path, problem=cfg.problem_path, plan_text=plan_txt, val_path=cfg.val_path, flags=flags, check_redundancy=cfg.check_redundancy)

    # Persist logs and structured trace
    val_stdout_path = os.path.join(run_dir, "val_stdout.txt")
    val_stderr_path = os.path.join(run_dir, "val_stderr.txt")
    with open(val_stdout_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(metrics.val_stdout or "")
    with open(val_stderr_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(metrics.val_stderr or "")

    trace_path = os.path.join(run_dir, "val_trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "time": st.time,
                    "action": st.action,
                    "adds": st.adds,
                    "deletes": st.deletes,
                    "failed": st.failed,
                    "failure_detail": st.failure_detail,
                } for st in metrics.steps
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Rich table
    table = Table(title="Green Agent — Plan Evaluation")
    table.add_column("Metric"); table.add_column("Value")
    table.add_row("Valid", str(metrics.valid))
    if not metrics.valid:
        table.add_row("Reason", metrics.failure_reason or "unknown")
    table.add_row("Length", str(metrics.length))
    table.add_row("Cost/Value", str(metrics.cost_value))
    table.add_row("First failure at", str(metrics.first_failure_at))
    table.add_row("First failed action", metrics.first_failed_action or "—")
    table.add_row("First failure reason", metrics.first_failure_reason or "—")
    table.add_row("First failure details", metrics.first_failure_detail or "—")
    table.add_row("Unsat preconds", str(metrics.unsat_count))
    if metrics.redundant_indices:
        table.add_row("Redundant steps", ", ".join(map(str, metrics.redundant_indices)) or "—")
    table.add_row("Advice fixes", str(metrics.advice_count))
    table.add_row("Advice top preds", ", ".join(f"{p}:{c}" for p,c in metrics.advice_top_predicates) or "—")
    table.add_row("LLM Latency (s)", f"{t1 - t0:.2f}")
    console.print(table)

    record = {
        "domain": cfg.domain_path,
        "problem": cfg.problem_path,
        "valid": metrics.valid,
        "length": metrics.length,
        "cost_value": metrics.cost_value,
        "first_failure_at": metrics.first_failure_at,
        "first_failed_action": metrics.first_failed_action,
        "first_failure_reason": metrics.first_failure_reason,
        "first_failure_detail": metrics.first_failure_detail,
        "unsat_count": metrics.unsat_count,
        "redundant_indices": metrics.redundant_indices,
        "advice_count": metrics.advice_count,
        "advice_top_predicates": metrics.advice_top_predicates,
        "raw_plan_path": raw_path,
        "norm_plan_path": plan_path,
        "val_stdout_path": val_stdout_path,
        "val_stderr_path": val_stderr_path,
        "val_trace_path": trace_path,
        "failure_reason": metrics.failure_reason,
        "run_dir": run_dir,
    }

    return record
