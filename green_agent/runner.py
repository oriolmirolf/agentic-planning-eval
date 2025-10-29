# green_agent/runner.py
from __future__ import annotations
import os, json, time
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table

from .config import EvalConfig
from .plan_parser import extract_plan, pretty
from .pddl_actions import actions_from_domain, semantics_from_domain   # <-- add semantics
from .metrics import compute_metrics

from purple_agent.openai_agent import OpenAIPurpleAgent
from purple_agent.http_agent import HTTPPurpleAgent
from purple_agent.file_agent import FilePurpleAgent

console = Console()

FORMAT_INSTRUCTIONS = """
Output ONLY the plan in a single fenced code block. One grounded action per line.
Use parentheses, lowercase action names, and the object names from the problem text.
Do NOT include explanations. Example format:
```
(action-name arg1 arg2)
(another-action x y z)
```
""".strip()

def load_text(path: Optional[str]) -> str:
    if not path: return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_purple(kind: str, *, url: Optional[str], model: Optional[str], temperature: float):
    if kind == "openai": return OpenAIPurpleAgent(model=model, temperature=temperature)
    if kind == "http":
        if not url: raise SystemExit("--purple=http requires --purple-url http://...")
        return HTTPPurpleAgent(url)
    if kind == "file":
        if not url: raise SystemExit("--purple=file requires --purple-url <path to plan file>")
        return FilePurpleAgent(url)
    raise SystemExit(f"Unknown purple kind: {kind}")

# fresh per-run folder (already in your version)
import time as _time
from pathlib import Path
def _make_run_dir(base_out: str, domain_path: str) -> str:
    example = Path(domain_path).parent.name or "run"
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_out) / f"{example}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

def evaluate_once(cfg: EvalConfig) -> Dict[str, Any]:
    # fresh run dir
    run_dir = _make_run_dir(cfg.out_dir, cfg.domain_path)

    # base prompt
    problem_nl = load_text(cfg.prompt_path).strip()

    # purple agent
    purple = build_purple(cfg.purple_kind, url=cfg.purple_url, model=cfg.openai_model, temperature=cfg.temperature)

    # call purple with the combined prompt; we send an empty 'actions_nl' to avoid duplication
    t0 = time.time()
    plan_raw = purple.generate_plan(problem_nl=problem_nl)
    t1 = time.time()

    raw_path = os.path.join(run_dir, "purple_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f: f.write(plan_raw)

    extracted = extract_plan(plan_raw)
    plan_txt = extracted.to_val_plan_text()
    plan_path = os.path.join(run_dir, "purple.plan")
    with open(plan_path, "w", encoding="utf-8") as f: f.write(plan_txt)

    metrics = compute_metrics(domain=cfg.domain_path, problem=cfg.problem_path, plan_text=plan_txt, val_path=cfg.val_path)

    table = Table(title="Green Agent — Plan Evaluation")
    table.add_column("Metric"); table.add_column("Value")
    table.add_row("Valid", str(metrics.valid))
    table.add_row("Coherent", str(metrics.coherence))
    table.add_row("Length", str(metrics.length))
    table.add_row("Cost/Value", str(metrics.cost_value))
    table.add_row("Unsat preconds", str(metrics.unsat_count))
    table.add_row("First failure at", str(metrics.first_failure_at))
    table.add_row("Redundant steps", ", ".join(map(str, metrics.redundant_indices)) or "—")
    table.add_row("Minimality", f"{metrics.minimality_ratio:.2f}")
    table.add_row("LLM Latency (s)", f"{t1 - t0:.2f}")
    console.print(table)

    record = {
        "domain": cfg.domain_path,
        "problem": cfg.problem_path,
        "valid": metrics.valid,
        "coherence": metrics.coherence,
        "length": metrics.length,
        "cost_value": metrics.cost_value,
        "unsat_count": metrics.unsat_count,
        "first_failure_at": metrics.first_failure_at,
        "redundant_indices": metrics.redundant_indices,
        "minimality_ratio": metrics.minimality_ratio,
        "raw_plan_path": raw_path,
        "norm_plan_path": plan_path,
        "run_dir": run_dir,
    }
    return record
