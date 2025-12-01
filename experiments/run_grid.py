# /Oriol-TFM/experiments/run_prompting_grid_parallel.py
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

from green_agent.cli import _resolve_paths
from green_agent.metrics import compute_metrics
from green_agent.plan_parser import extract_plan
from purple_agent.strategy_agent import StrategyPurpleAgent

from .mlflow_utils import (
    log_artifacts as mlflow_log_artifacts,
    log_dataset_for_domain,
    log_metrics as mlflow_log_metrics,
    log_params as mlflow_log_params,
    mlflow_run,
)

console = Console()

# ---------- model loading ----------


def _load_models(py_path: Path) -> dict[str, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("exp_models", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp_models"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    MODELS = getattr(mod, "MODELS", None)
    if not isinstance(MODELS, dict):
        raise RuntimeError(f"{py_path} must define dict MODELS")
    out = {}
    for k, v in MODELS.items():
        v = dict(v)
        if "api_key" not in v:
            env = v.get("api_key_env")
            if env:
                v["api_key"] = os.getenv(env)
        out[k] = v
    return out


# ---------- job & helpers ----------


@dataclass
class Job:
    domain: str
    index: int
    model_id: str
    technique: str
    domain_path: str
    problem_path: str
    prompt_text: str
    optimal_cost: float | None
    planner_cfg: dict[str, Any]
    judge_cfg: dict[str, Any] | None
    run_dir: Path | None = None
    plan_path: Path | None = None
    raw_path: Path | None = None


def _roles_for(job: Job) -> dict[str, dict[str, Any]]:
    if job.technique == "cot_sc":
        return {"planner": job.planner_cfg, "judge": (job.judge_cfg or job.planner_cfg)}
    return {"planner": job.planner_cfg}


def _providers_for(job: Job) -> list[str]:
    provs = set()
    provs.add(job.planner_cfg.get("provider"))
    if job.technique == "cot_sc" and job.judge_cfg:
        provs.add(job.judge_cfg.get("provider"))
    return sorted(p for p in provs if p)


def _fmt_inflight(d: dict[str, int]) -> str:
    return (
        "In-flight LLM: {total} "
        f"[OpenAI {d.get('openai', 0)} | Anthropic {d.get('anthropic', 0)} | "
        f"Google {d.get('google', 0)} | Local {d.get('openai_compat', 0)}]"
    ).format(**d)


# ---------- phases ----------


def _llm_generate(
    job: Job,
    batch_root: Path,
    *,
    semaphores: dict[str, threading.BoundedSemaphore],
    progress: Progress | None,
    inflight: dict[str, int],
    lock: threading.Lock,
    status_task: int,
    strategy_settings: dict[str, Any],
) -> Job:
    provs = _providers_for(job)
    # Acquire provider semaphores in sorted order to avoid deadlocks
    locks = [semaphores[p] for p in provs if p in semaphores]
    for s in locks:
        s.acquire()

    # mark inflight counts
    if progress is not None:
        with lock:
            inflight["total"] += 1
            for p in provs:
                inflight[p] = inflight.get(p, 0) + 1
            progress.update(status_task, description=_fmt_inflight(inflight))

    try:
        run_dir = batch_root / f"{job.technique}-{job.model_id}-p{job.index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        roles = _roles_for(job)
        agent = StrategyPurpleAgent(
            strategy_name=job.technique, roles=roles, settings=strategy_settings
        )

        t0 = time.time()
        raw_out = agent.generate_plan(problem_nl=job.prompt_text or "")
        dt = time.time() - t0

        raw_path = run_dir / "purple_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        plan_txt = extract_plan(raw_out).to_val_plan_text()
        plan_path = run_dir / "purple.plan"
        plan_path.write_text(plan_txt, encoding="utf-8")

        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "domain": job.domain,
                    "index": job.index,
                    "model_id": job.model_id,
                    "technique": job.technique,
                    "elapsed_s": dt,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        job.run_dir, job.plan_path, job.raw_path = run_dir, plan_path, raw_path
        return job
    finally:
        # unmark inflight & release
        if progress is not None:
            with lock:
                inflight["total"] -= 1
                for p in provs:
                    inflight[p] -= 1
                progress.update(status_task, description=_fmt_inflight(inflight))
        for s in reversed(locks):
            s.release()


def _val_validate(
    job: Job, *, val_path: str | None, tolerance: float = 0.001
) -> dict[str, Any]:
    assert job.plan_path and job.run_dir
    plan_txt = job.plan_path.read_text(encoding="utf-8")
    flags = ("-v", "-t", str(tolerance))
    metrics = compute_metrics(
        domain=job.domain_path,
        problem=job.problem_path,
        plan_text=plan_txt,
        val_path=val_path,
        flags=flags,
        check_redundancy=False,
    )
    val_stdout_path = job.run_dir / "val_stdout.txt"
    val_stderr_path = job.run_dir / "val_stderr.txt"
    val_trace_path = job.run_dir / "val_trace.json"

    val_stdout_path.write_text(
        metrics.val_stdout or "", encoding="utf-8", errors="replace"
    )
    val_stderr_path.write_text(
        metrics.val_stderr or "", encoding="utf-8", errors="replace"
    )
    val_trace_path.write_text(
        json.dumps(
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
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    score = (
        float(job.optimal_cost) / float(metrics.cost_value)
        if (job.optimal_cost and metrics.cost_value and metrics.cost_value > 0)
        else None
    )
    return {
        "domain": job.domain_path,
        "problem": job.problem_path,
        "valid": metrics.valid,
        "length": metrics.length,
        "cost_value": metrics.cost_value,
        "optimal_cost": job.optimal_cost,
        "score": score,
        "first_failure_at": metrics.first_failure_at,
        "first_failed_action": metrics.first_failed_action,
        "first_failure_reason": metrics.first_failure_reason,
        "first_failure_detail": metrics.first_failure_detail,
        "unsat_count": metrics.unsat_count,
        "advice_count": metrics.advice_count,
        "advice_top_predicates": metrics.advice_top_predicates,
        "failure_reason": metrics.failure_reason,
        "run_dir": str(job.run_dir),
        "raw_plan_path": str(job.raw_path),
        "norm_plan_path": str(job.plan_path),
        "val_stdout_path": str(val_stdout_path),
        "val_stderr_path": str(val_stderr_path),
        "val_trace_path": str(val_trace_path),
        "val_attempts": getattr(metrics, "val_attempts", 1),
        "val_warning": getattr(metrics, "val_warning", None),
    }


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--models-file", default="experiments/models.py")
    ap.add_argument(
        "--judge-model",
        default=None,
        help="ID from MODELS to use as CoT-SC judge; default: same as planner",
    )
    ap.add_argument(
        "--techniques",
        default="base,cot,ltm",
        help="Comma-separated subset of {base,cot,ltm,cot_sc}. "
        "Default: base,cot,ltm (faster).",
    )
    ap.add_argument("--out", default="out/prompting-par")
    ap.add_argument("--val-path", default=None)

    # global worker caps
    ap.add_argument(
        "--llm-workers", type=int, default=16, help="Total threads for LLM generation"
    )
    ap.add_argument(
        "--val-workers", type=int, default=1, help="Threads for VAL (1 == safest)"
    )

    # per-provider caps (defaults favor higher local throughput)
    ap.add_argument("--openai-par", type=int, default=8)
    ap.add_argument("--anthropic-par", type=int, default=4)
    ap.add_argument("--google-par", type=int, default=4)
    ap.add_argument("--local-par", type=int, default=16)  # openai_compat

    args = ap.parse_args()

    # Parse techniques (speed knob: default excludes cot_sc)
    selected_techs = tuple(t.strip() for t in args.techniques.split(",") if t.strip())
    valid = {"base", "cot", "ltm", "cot_sc"}
    for t in selected_techs:
        if t not in valid:
            raise SystemExit(f"Unknown technique '{t}'. Choose from {sorted(valid)}")

    MODELS = _load_models(Path(args.models_file).resolve())

    # problem range
    start = int(args.start)
    if args.end is None:
        probs_dir = Path("examples") / args.domain / "problems_pddl"
        ids = sorted(
            [
                int(re.search(r"(\d+)", p.stem).group(1))
                for p in probs_dir.glob("problem*.pddl")
            ]
        )
        end = ids[-1] if ids else start
    else:
        end = int(args.end)

    # batch root
    stamp = time.strftime("%Y%m%d-%H%M%S")
    batch_root = Path(args.out) / f"{args.domain}-{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)

    # build jobs
    jobs: list[Job] = []
    counts_by_tech: dict[str, int] = {t: 0 for t in selected_techs}
    for idx in range(start, end + 1):
        auto = _resolve_paths(args.domain, idx)
        if not auto["domain"] or not auto["problem"]:
            console.print(
                f"[yellow][SKIP][/yellow] Could not resolve paths for p{idx:02d}"
            )
            continue
        prompt_text = (auto["prompt_text"] or "").strip()
        for model_id, mcfg in MODELS.items():
            for tech in selected_techs:
                jobs.append(
                    Job(
                        domain=args.domain,
                        index=idx,
                        model_id=model_id,
                        technique=tech,
                        domain_path=auto["domain"],
                        problem_path=auto["problem"],
                        prompt_text=prompt_text,
                        optimal_cost=auto.get("optimal_cost"),
                        planner_cfg=mcfg,
                        judge_cfg=(
                            MODELS.get(args.judge_model)
                            if (tech == "cot_sc" and args.judge_model)
                            else None
                        ),
                    )
                )
                counts_by_tech[tech] += 1

    total_jobs = len(jobs)
    if total_jobs == 0:
        console.print("[red]No jobs to run.[/red]")
        return

    # SPEED KNOB: shuffle job order to avoid long head-of-line batches
    random.shuffle(jobs)

    # semaphores per provider
    prov_caps = {
        "openai": max(0, args.openai_par),
        "anthropic": max(0, args.anthropic_par),
        "google": max(0, args.google_par),
        "openai_compat": max(0, args.local_par),
    }
    semaphores = {
        p: threading.BoundedSemaphore(c if c > 0 else 9999)
        for p, c in prov_caps.items()
    }

    # inflight counters for status line
    inflight = {
        "total": 0,
        "openai": 0,
        "anthropic": 0,
        "google": 0,
        "openai_compat": 0,
    }
    inflight_lock = threading.Lock()

    progress_cols = [
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA "),
        TimeRemainingColumn(),
    ]

    strategy_settings: dict[str, Any] = {}
    if "cot_sc" in selected_techs:
        strategy_settings.setdefault("cot_sc", {})["samples"] = (
            3  # reduce self-consistency samples
        )

    console.print(
        "[green]Phase 1:[/green] Generating plans (LLM) "
        "with {max(1, args.llm_workers)} workers"
    )
    with Progress(*progress_cols, console=console) as progress:
        # high-level tasks
        t_queued = progress.add_task("Queued (LLM)", total=total_jobs)
        t_llm_done = progress.add_task("Finished (LLM)", total=total_jobs)
        # per-technique tasks
        t_by_tech = {
            tech: progress.add_task(
                f"{tech} (LLM done)", total=counts_by_tech.get(tech, 0)
            )
            for tech in selected_techs
        }
        # status line for inflight
        t_status = progress.add_task(_fmt_inflight(inflight), total=1, completed=0)

        llm_done: list[Job] = []
        # submit all jobs to pool; update "Queued (LLM)" as we submit
        with ThreadPoolExecutor(max_workers=max(1, args.llm_workers)) as pool:
            future_map = {}
            for job in jobs:
                fut = pool.submit(
                    _llm_generate,
                    job,
                    batch_root,
                    semaphores=semaphores,
                    progress=progress,
                    inflight=inflight,
                    lock=inflight_lock,
                    status_task=t_status,
                    strategy_settings=strategy_settings,
                )
                future_map[fut] = job
                progress.update(t_queued, advance=1)

            # consume results as they complete (no submission-order blocking)
            for fut in as_completed(future_map):
                j = fut.result()
                llm_done.append(j)
                progress.update(t_llm_done, advance=1)
                progress.update(t_by_tech[j.technique], advance=1)

    console.print(
        "[green]Phase 2:[/green] Validating plans (VAL) with "
        "{max(1, args.val_workers)} worker(s)"
    )
    with Progress(*progress_cols, console=console) as progress:
        t_val_done = progress.add_task("Finished (VAL)", total=len(jobs))
        t_val_by_tech = {
            tech: progress.add_task(
                f"{tech} (VAL done)", total=counts_by_tech.get(tech, 0)
            )
            for tech in selected_techs
        }

        results: list[tuple[Job, dict[str, Any]]] = []
        if max(1, args.val_workers) == 1:
            for job in llm_done:
                rec = _val_validate(job, val_path=args.val_path)
                results.append((job, rec))
                progress.update(t_val_done, advance=1)
                progress.update(t_val_by_tech[j.technique], advance=1)
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.val_workers)) as pool:
                future_map = {
                    pool.submit(_val_validate, job, val_path=args.val_path): job
                    for job in llm_done
                }
                for fut in as_completed(future_map):
                    job = future_map[fut]
                    rec = fut.result()
                    results.append((job, rec))
                    progress.update(t_val_done, advance=1)
                    progress.update(t_val_by_tech[j.technique], advance=1)

    # write outputs
    csv_path = batch_root / "experiments.csv"
    jsonl_path = batch_root / "records.jsonl"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts",
                    "domain",
                    "index",
                    "model_id",
                    "technique",
                    "valid",
                    "cost_value",
                    "length",
                    "unsat_count",
                    "failure_reason",
                    "score",
                    "optimal_cost",
                    "val_attempts",
                    "val_warning",
                    "run_dir",
                    "raw_plan",
                    "norm_plan",
                    "val_stdout",
                    "val_stderr",
                    "val_trace",
                ]
            )

    for job, rec in results:
        # --- file outputs ---
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(
                json.dumps(
                    {
                        **rec,
                        "domain_name": job.domain,
                        "index": job.index,
                        "model_id": job.model_id,
                        "technique": job.technique,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    job.domain,
                    job.index,
                    job.model_id,
                    job.technique,
                    rec.get("valid"),
                    rec.get("cost_value"),
                    rec.get("length"),
                    rec.get("unsat_count"),
                    rec.get("failure_reason"),
                    rec.get("score"),
                    rec.get("optimal_cost"),
                    rec.get("val_attempts"),
                    rec.get("val_warning"),
                    rec.get("run_dir"),
                    rec.get("raw_plan_path"),
                    rec.get("norm_plan_path"),
                    rec.get("val_stdout_path"),
                    rec.get("val_stderr_path"),
                    rec.get("val_trace_path"),
                ]
            )

        # --- MLflow logging (one run per job; no-op if MLflow is disabled) ---
        planner_cfg = job.planner_cfg or {}
        judge_cfg = job.judge_cfg or {}

        model_name = planner_cfg.get("model") or job.model_id
        dataset_name = job.domain

        run_name = f"{dataset_name}-p{job.index:02d}-{model_name}-{job.technique}"

        ml_tags = {
            "runner": "experiments.run_prompting_grid_parallel",
            "stage": "evaluation",
            "technique": job.technique,
            "dataset": dataset_name,
            "model": model_name,
        }

        with mlflow_run(run_name=run_name, tags=ml_tags) as run:
            if run is None:
                continue

            # Dataset = whole domain (blocks/logistics/...), logged once per run
            log_dataset_for_domain(
                domain=job.domain,
                domain_path=job.domain_path,
            )

            # Extra params (pipeline config / VAL config, etc.)
            ml_params = {
                "domain": job.domain,
                "problem_index": job.index,
                "problem_path": job.problem_path,
                "model_id": job.model_id,
                "technique": job.technique,
                "optimal_cost": rec.get("optimal_cost"),
                "planner.provider": planner_cfg.get("provider"),
                "planner.model": planner_cfg.get("model"),
                "planner.base_url": planner_cfg.get("base_url"),
                "judge.provider": judge_cfg.get("provider"),
                "judge.model": judge_cfg.get("model"),
                "judge.base_url": judge_cfg.get("base_url"),
                "val_path": args.val_path,
            }
            mlflow_log_params(ml_params)

            # Core metrics
            ml_metrics = {
                "valid": 1.0 if rec.get("valid") else 0.0,
                "cost_value": rec.get("cost_value"),
                "length": rec.get("length"),
                "score": rec.get("score"),
                "unsat_count": rec.get("unsat_count") or 0,
                "val_attempts": rec.get("val_attempts") or 0,
            }
            mlflow_log_metrics(ml_metrics)

            # Artifacts: full run dir (plans + VAL logs / trace)
            if job.run_dir:
                mlflow_log_artifacts(
                    str(job.run_dir),
                    artifact_path=f"{dataset_name}/p{job.index:02d}/{job.model_id}/{job.technique}",
                )

    console.print(
        f"\n[bold green]OK[/bold green] • Wrote:\n- {csv_path}\n- {jsonl_path}"
    )
    console.print(f"[dim]Batch root: {batch_root}[/dim]")


if __name__ == "__main__":
    main()
