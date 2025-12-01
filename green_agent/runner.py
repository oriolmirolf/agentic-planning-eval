# green_agent/runner.py
from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from purple_agent.react_dspy.react_agent import ReActDSPyPurpleAgent

from .config import EvalConfig
from .metrics import compute_metrics
from .plan_parser import extract_plan

console = Console()


def load_text(path: str | None) -> str:
    if not path:
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def build_purple(
    kind: str,
    *,
    model: str | None,
    a2a_url: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
    strategy_name: str | None = None,
    strategy_params: dict | None = None,
):
    if kind == "openai":
        from purple_agent.openai_agent import OpenAIPurpleAgent

        return OpenAIPurpleAgent(model=model, base_url=base_url, api_key=api_key)
    if kind == "a2a":
        if not a2a_url:
            raise SystemExit("Missing purple_url for 'a2a' purple.")
        from purple_agent.a2a_agent import A2APurpleAgent

        return A2APurpleAgent(url=a2a_url)
    if kind in ("strategy", "composite"):
        from purple_agent.strategy_agent import StrategyPurpleAgent

        params = strategy_params or {}
        roles = params.get("roles", {})
        settings = params.get("settings", {})
        if not (strategy_name and roles):
            raise SystemExit(
                "strategy kind requires strategy_name and strategy_params.roles"
            )
        return StrategyPurpleAgent(
            strategy_name=strategy_name, roles=roles, settings=settings
        )
    if kind == "react_dspy":
        return ReActDSPyPurpleAgent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
        )
    raise SystemExit(f"Unknown purple kind: {kind!r}")


def _make_run_dir(base_out: str, domain_path: str) -> str:
    example = Path(domain_path).parent.name or "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_out) / f"{example}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _score_from(metrics, optimal_cost: float | None) -> float | None:
    if not metrics.valid:
        return 0.0
    oc = optimal_cost
    cost = metrics.cost_value
    if oc is not None and cost is not None and cost > 0:
        return float(oc) / float(cost)
    return None


def evaluate_once(cfg: EvalConfig) -> dict[str, Any]:
    # Respect explicit run_dir (used by whole-domain batch);
    # otherwise create a new stamped folder.
    run_dir = cfg.run_dir or _make_run_dir(cfg.out_dir, cfg.domain_path)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Prefer inline prompt_text (new flow). Fallback to legacy prompt_path.
    problem_nl = (cfg.prompt_text or load_text(cfg.prompt_path) or "").strip()

    purple = build_purple(
        cfg.purple_kind,
        model=cfg.openai_model,
        a2a_url=cfg.purple_url,
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        strategy_name=cfg.strategy_name,
        strategy_params=cfg.strategy_params,
    )

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
    metrics = compute_metrics(
        domain=cfg.domain_path,
        problem=cfg.problem_path,
        plan_text=plan_txt,
        val_path=cfg.val_path,
        flags=flags,
        check_redundancy=cfg.check_redundancy,
    )

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
                }
                for st in metrics.steps
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Rich table (conditionally printed)
    if cfg.print_card:
        table = Table(title="Green Agent — Plan Evaluation")
        table.add_column("Metric")
        table.add_column("Value")
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
            table.add_row(
                "Redundant steps", ", ".join(map(str, metrics.redundant_indices)) or "—"
            )
        table.add_row("Advice fixes", str(metrics.advice_count))
        table.add_row(
            "Advice top preds",
            ", ".join(f"{p}:{c}" for p, c in metrics.advice_top_predicates) or "—",
        )
        table.add_row("LLM Latency (s)", f"{t1 - t0:.2f}")
        table.add_row("VAL attempts", str(getattr(metrics, "val_attempts", 1)))
        if getattr(metrics, "val_warning", None):
            table.add_row("Warning", metrics.val_warning)
        console.print(table)

    score = _score_from(metrics, cfg.optimal_cost)

    record = {
        "domain": cfg.domain_path,
        "problem": cfg.problem_path,
        "valid": metrics.valid,
        "length": metrics.length,
        "cost_value": metrics.cost_value,
        "optimal_cost": cfg.optimal_cost,
        "score": score,
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
        "val_attempts": getattr(metrics, "val_attempts", 1),
        "val_warning": getattr(metrics, "val_warning", None),
    }
    return record


# ---------------- Whole-domain evaluation with parallel LLM ----------------
def evaluate_domain(
    cfg_base: EvalConfig,
    *,
    start: int | None = None,
    end: int | None = None,
    print_cards: bool = False,
    llm_workers: int = 4,  # parallelize LLM calls
    val_workers: int = 1,  # default: serialize VAL (avoid stampede)
) -> dict[str, Any]:
    """
    Two-phase pipeline:
      1) Parallel: generate plans with the LLM / purple agent (llm_workers threads).
      2) Then validate plans (VAL). Default sequential (val_workers=1).
         You can increase val_workers if you want some parallelism here too.

    Layout:
      <out>/<domain>-<stamp>/
        domain_summary.json
        results.jsonl
        scores.csv
        p01/  (raw+plan+val logs per problem)
        p02/
        ...
    """
    domain_dir = Path(cfg_base.domain_path).parent
    problems_dir = domain_dir / "problems_pddl"
    with open(domain_dir / "prompts.json", encoding="utf-8") as f:
        data = json.load(f)
    domain_prompt = (data.get("domain_prompt") or "").strip()
    problems = data.get("problems", [])

    # filter problems by range
    items: list[dict] = []
    for item in problems:
        pid = str(item.get("id", "")).strip()
        try:
            idx = int(pid[1:]) if pid.lower().startswith("p") else int(pid)
        except Exception:
            continue
        if start is not None and idx < start:
            continue
        if end is not None and idx > end:
            continue
        items.append({"pid": pid, "idx": idx, "entry": item})

    # Create main (batch) run folder
    stamp = time.strftime("%Y%m%d-%H%M%S")
    batch_root = Path(cfg_base.out_dir) / f"{domain_dir.name}-{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1: parallel LLM plan generation ----------
    def _gen_job(job: dict) -> dict[str, Any]:
        """Generate plan for a single problem and write purple_raw.txt / purple.plan."""
        pid = job["pid"]
        idx = job["idx"]
        entry = job["entry"]
        prob_dir = batch_root / pid
        prob_dir.mkdir(parents=True, exist_ok=True)

        problem_pddl = problems_dir / f"problem{idx}.pddl"
        prompt_text = (
            domain_prompt + "\n\n" + str(entry.get("prompt", "")).strip()
        ).strip()
        oc = entry.get("optimal_cost")

        # Build a per-problem config just for the purple generation (no VAL yet)
        cfg = EvalConfig(
            domain_path=cfg_base.domain_path,
            problem_path=str(problem_pddl),
            out_dir=str(batch_root),  # not used since run_dir is set
            run_dir=str(prob_dir),  # write artifacts in pXX/
            val_path=cfg_base.val_path,  # passed later
            val_flags=cfg_base.val_flags,  # passed later
            tolerance=cfg_base.tolerance,  # passed later
            purple_kind=cfg_base.purple_kind,
            purple_url=cfg_base.purple_url,
            prompt_text=prompt_text,
            openai_model=cfg_base.openai_model,
            check_redundancy=False,  # redundancy is a VAL concern; skip here
            llm_base_url=cfg_base.llm_base_url,
            llm_api_key=cfg_base.llm_api_key,
            optimal_cost=oc,
            print_card=False,  # quiet per-problem in batch by default
        )

        # Generate plan (like evaluate_once but stop before VAL)
        purple = build_purple(
            cfg.purple_kind,
            model=cfg.openai_model,
            a2a_url=cfg.purple_url,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            strategy_name=cfg.strategy_name,
            strategy_params=cfg.strategy_params,
        )
        t0 = time.time()
        plan_raw = purple.generate_plan(problem_nl=cfg.prompt_text or "")
        # small jitter to avoid lockstep bursts with some providers
        time.sleep(random.uniform(0.05, 0.15))
        t1 = time.time()

        raw_path = os.path.join(cfg.run_dir, "purple_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(plan_raw)

        extracted = extract_plan(plan_raw)
        plan_txt = extracted.to_val_plan_text()
        plan_path = os.path.join(cfg.run_dir, "purple.plan")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan_txt)

        return {
            "pid": pid,
            "idx": idx,
            "problem_path": str(problem_pddl),
            "optimal_cost": oc,
            "run_dir": str(prob_dir),
            "norm_plan_path": plan_path,
            "raw_plan_path": raw_path,
            "llm_latency": t1 - t0,
        }

    llm_jobs = items
    llm_results: list[dict[str, Any]] = []

    progress_columns = [
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA "),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns, console=console) as progress:
        t_llm = progress.add_task("Generating plans (LLM)", total=len(llm_jobs))
        with ThreadPoolExecutor(max_workers=max(1, llm_workers)) as pool:
            futures = [pool.submit(_gen_job, job) for job in llm_jobs]
            for fut in as_completed(futures):
                llm_results.append(fut.result())
                progress.update(t_llm, advance=1)

        # ---------- Phase 2: VAL validation (default sequential) ----------
        def _val_job(r: dict[str, Any]) -> dict[str, Any]:
            run_dir = r["run_dir"]
            plan_path = r["norm_plan_path"]
            flags = (*cfg_base.val_flags, "-t", str(cfg_base.tolerance))
            with open(plan_path, encoding="utf-8") as f:
                plan_txt = f.read()

            metrics = compute_metrics(
                domain=cfg_base.domain_path,
                problem=r["problem_path"],
                plan_text=plan_txt,
                val_path=cfg_base.val_path,
                flags=flags,
                check_redundancy=cfg_base.check_redundancy,
            )

            # Persist logs
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
                        }
                        for st in metrics.steps
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            score = _score_from(metrics, r.get("optimal_cost"))
            return {
                **r,
                "valid": metrics.valid,
                "length": metrics.length,
                "cost_value": metrics.cost_value,
                "score": score,
                "first_failure_at": metrics.first_failure_at,
                "first_failed_action": metrics.first_failed_action,
                "first_failure_reason": metrics.first_failure_reason,
                "first_failure_detail": metrics.first_failure_detail,
                "unsat_count": metrics.unsat_count,
                "redundant_indices": metrics.redundant_indices,
                "advice_count": metrics.advice_count,
                "advice_top_predicates": metrics.advice_top_predicates,
                "val_stdout_path": val_stdout_path,
                "val_stderr_path": val_stderr_path,
                "val_trace_path": trace_path,
                "failure_reason": metrics.failure_reason,
                "val_attempts": getattr(metrics, "val_attempts", 1),
                "val_warning": getattr(metrics, "val_warning", None),
            }

        t_val = progress.add_task("Validating plans (VAL)", total=len(llm_results))
        final_results: list[dict[str, Any]] = []

        if max(1, val_workers) == 1:
            # sequential VAL
            for r in llm_results:
                final_results.append(_val_job(r))
                progress.update(t_val, advance=1)
        else:
            with ThreadPoolExecutor(max_workers=max(1, val_workers)) as pool:
                futures = [pool.submit(_val_job, r) for r in llm_results]
                for fut in as_completed(futures):
                    final_results.append(fut.result())
                    progress.update(t_val, advance=1)

    # ---------- Aggregate & write batch artifacts ----------
    results = []
    for rec in final_results:
        # Normalize paths relative to the batch root
        for k in (
            "raw_plan_path",
            "norm_plan_path",
            "val_stdout_path",
            "val_stderr_path",
            "val_trace_path",
            "run_dir",
        ):
            if rec.get(k):
                try:
                    rec[k] = str(Path(rec[k]).relative_to(batch_root))
                except Exception:
                    pass
        rec["problem_id"] = rec["pid"]
        results.append(rec)

    # results.jsonl
    jsonl_path = batch_root / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate numbers
    n = len(results)
    # New: counts by reason (including "valid")
    counts_by_reason: defaultdict[str, int] = defaultdict(int)
    for r in results:
        if r.get("valid"):
            counts_by_reason["valid"] += 1
        else:
            reason = r.get("failure_reason") or "unknown_failure"
            counts_by_reason[reason] += 1
    # Deterministic ordering for JSON/console
    counts_by_reason = dict(sorted(counts_by_reason.items(), key=lambda kv: kv[0]))

    # Score aggregation (unchanged)
    scores = [
        r.get("score") for r in results if isinstance(r.get("score"), (int, float))
    ]
    total_score = sum(scores) if scores else 0.0

    summary = {
        "domain": str(domain_dir),
        "root_dir": str(batch_root),
        "results_path": str(jsonl_path),
        "count": n,
        "total_score": total_score,
        "scores": {r["problem_id"]: r.get("score") for r in results},
        "counts_by_reason": counts_by_reason,  # <--- NEW primary aggregate
    }

    with open(batch_root / "domain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV (kept the same columns; reason present per row)
    csv_path = batch_root / "scores.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "problem_id",
                "valid",
                "optimal_cost",
                "cost_value",
                "score",
                "failure_reason",
                "val_attempts",
                "val_warning",
                "llm_latency_s",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.get("problem_id"),
                    r.get("valid"),
                    r.get("optimal_cost"),
                    r.get("cost_value"),
                    r.get("score"),
                    r.get("failure_reason"),
                    r.get("val_attempts"),
                    r.get("val_warning"),
                    f"{r.get('llm_latency', 0.0):.2f}",
                ]
            )

    # Console summary: show counts_by_reason instead of valid/invalid
    table = Table(title=f"Domain Summary — {domain_dir.name}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Problems", str(n))
    table.add_row("Total Score", f"{total_score:.4f}")
    console.print(table)

    if counts_by_reason:
        console.print("Outcome breakdown (reason -> count):")
        for reason, count in counts_by_reason.items():
            console.print(f"  - {reason}: {count}")

    console.print(f"[results.jsonl]  {jsonl_path}")
    console.print(f"[summary.json]   {batch_root / 'domain_summary.json'}")
    console.print(f"[scores.csv]     {csv_path}")

    return summary
