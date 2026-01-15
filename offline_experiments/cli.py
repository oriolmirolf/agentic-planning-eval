from __future__ import annotations

import argparse
import os
import sys
import time
import statistics
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow  # type: ignore
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .data import DomainSpec, discover_domains, load_domain
from .mlflow_utils import get_git_info, init_mlflow, set_required_git_tag
from .openai_client import LLMRequest, OpenAICompatClient, try_list_models
from .prompting import STRATEGIES, run_strategy
from .vecinf_adapter import VecInfLauncher

# -----------------------------
# Model catalog / knobs
# -----------------------------

THINKING_MODELS = {
    "Kimi-K2-Thinking",
    "Qwen3-VL-30B-A3B-Thinking",
    "gpt-5-nano-thinking", 
    "gemini-3-flash-thinking",
    "gemini-3-pro-thinking",
}

DEFAULT_MODELS = [
    "Kimi-K2-Thinking",
    "Qwen3-Coder-480B-A35B-Instruct",
    "Qwen2.5-72B-Instruct",
    "Qwen3-VL-30B-A3B-Thinking",
    "gpt-5-nano",
    "gemini-3-flash",
    "gemini-3-flash-thinking", 
    "gemini-3-pro",
]

DEFAULT_STRATEGIES = [
    "baseline",
    "zero_shot_cot",
    "plan_and_solve",
    "step_back",
    "self_refine",
    "tree_of_thought"
]

DEFAULT_BASE_URL = (
    os.getenv("GREEN_BENCH_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "http://localhost:5679/v1"
)


def is_thinking_model(model_name: str) -> bool:
    """Detects if a model is a reasoning/thinking model based on name."""
    mn = (model_name or "").lower()
    return (model_name in THINKING_MODELS) or ("thinking" in mn) or ("reasoner" in mn) or ("o1" in mn)


def max_tokens_for(model_name: str) -> int:
    # Reasoning models usually require significantly more output tokens for the CoT
    return 16384 if is_thinking_model(model_name) else 4096


# -----------------------------
# VecInf lifecycle
# -----------------------------


@dataclass(frozen=True)
class ModelEndpoint:
    model_name: str
    job_id: str
    base_url: str


def _parse_base_url_map(raw: str | None) -> dict[str, str]:
    """Parse a mapping like: "modelA=http://.../v1,modelB=http://.../v1"."""
    if not raw:
        return {}
    out: dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid --base-url-map entry {chunk!r}. Expected 'MODEL=URL'."
            )
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(
                f"Invalid --base-url-map entry {chunk!r}. Expected 'MODEL=URL'."
            )
        out[k] = v
    return out


@contextmanager
def launched_model(
    launcher: VecInfLauncher,
    model_name: str,
    *,
    on_phase: Callable[[str], None] | None = None,
) -> Iterator[ModelEndpoint]:
    job_id: str | None = None
    try:
        if on_phase:
            on_phase(f"Launching {model_name} via vec_inf…")
        job_id = launcher.launch(model_name)

        if on_phase:
            on_phase(f"Waiting until ready: {model_name}…")
        base_url = launcher.wait_until_ready(job_id)

        yield ModelEndpoint(model_name=model_name, job_id=job_id, base_url=base_url)
    finally:
        if job_id:
            try:
                if on_phase:
                    on_phase(f"Shutting down {model_name}…")
                launcher.shutdown(job_id)
            except Exception as e:
                print(
                    f"[WARN] Failed to shutdown model job_id={job_id}: {e}",
                    file=sys.stderr,
                )


@contextmanager
def manual_model(
    model_name: str,
    base_url: str,
    *,
    on_phase: Callable[[str], None] | None = None,
) -> Iterator[ModelEndpoint]:
    """Use a pre-launched OpenAI-compatible endpoint (e.g., SSH tunnel)."""
    if on_phase:
        on_phase(f"Using manual endpoint for {model_name}: {base_url}")
    yield ModelEndpoint(model_name=model_name, job_id="manual", base_url=base_url)


# -----------------------------
# MLflow helpers
# -----------------------------


@contextmanager
def mlflow_run(name: str, *, nested: bool) -> Iterator[str]:
    with mlflow.start_run(run_name=name, nested=nested) as run:
        yield run.info.run_id


def log_run_common_tags(*, git_commit: str, repo_dirty: bool) -> None:
    set_required_git_tag(git_commit)
    mlflow.set_tag("repo_dirty", str(bool(repo_dirty)).lower())


def _log_aggregate_metrics(results: list[dict[str, float]], prefix: str = "") -> None:
    """Calculates and logs aggregate stats (avg success, avg score) to current run."""
    if not results:
        return

    count = len(results)
    # Safely extract with defaults
    successes = sum(r.get("is_success", 0.0) for r in results)
    scores = sum(r.get("score", 0.0) for r in results)
    valid_runs = sum(r.get("is_valid", 0.0) for r in results)
    
    # Calculate averages
    avg_success = successes / count
    avg_score = scores / count
    validity_rate = valid_runs / count
    
    # Duration stats (if available)
    durations = [r.get("wall_time", 0.0) for r in results]
    avg_duration = statistics.mean(durations) if durations else 0.0

    metrics_to_log = {
        f"{prefix}agg_success_rate": avg_success,
        f"{prefix}agg_avg_score": avg_score,
        f"{prefix}agg_validity_rate": validity_rate,
        f"{prefix}agg_avg_duration_s": avg_duration,
        f"{prefix}count": float(count),
    }
    
    mlflow.log_metrics(metrics_to_log)


# -----------------------------
# Core runner
# -----------------------------


def _load_domains(examples_dir: Path, domain_names: Sequence[str]) -> list[DomainSpec]:
    return [load_domain(examples_dir, dn) for dn in domain_names]


def _ensure_repo_on_syspath(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def run_suite(
    *,
    examples_dir: Path,
    domains: Sequence[str],
    models: Sequence[str],
    strategies: Sequence[str],
    out_dir: Path,
    val_path: str | None,
    val_flags: tuple[str, ...],
    tolerance: float,
    mlflow_tracking_uri: str | None,
    experiment_name: str,
    openai_api_key: str,
    request_timeout_s: float,
    launch_mode: str,
    base_url: str,
    base_url_map: dict[str, str],
    allow_shared_base_url: bool,
    problem_indices: Sequence[int] | None = None,
    limit_problems: int | None = None,
) -> None:
    console = Console(stderr=True)

    git = get_git_info()
    if git.dirty:
        console.print(
            f"[yellow][WARN][/yellow] Git repo is dirty (uncommitted changes). commit={git.commit}"
        )

    _ensure_repo_on_syspath(git.repo_root)

    # Lazy import
    from .evaluator import EvalResult, evaluate_with_val, parse_plan, write_artifacts

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    session_out = out_dir / f"planning_benchmark_{stamp}"
    session_out.mkdir(parents=True, exist_ok=True)

    domain_specs = _load_domains(examples_dir, domains)

    # --- Progress totals ---
    def n_problems(dom: DomainSpec) -> int:
        selected = dom.problems
        if problem_indices:
            want = set(int(x) for x in problem_indices)
            selected = [p for p in dom.problems if int(p.index) in want]
        if limit_problems:
            selected = selected[:limit_problems]
        return len(selected)

    problems_by_domain: dict[str, int] = {d.name: n_problems(d) for d in domain_specs}
    total_per_model = sum(
        problems_by_domain[d.name] * len(strategies) for d in domain_specs
    )
    grand_total = total_per_model * len(list(models))

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA "),
        TimeRemainingColumn(),
    ]

    def phase_update(progress: Progress, task_id: int, msg: str) -> None:
        progress.update(task_id, description=f"Phase: {msg}")

    with Progress(*progress_columns, console=console) as progress:
        t_phase = progress.add_task("Phase: idle", total=None)
        t_all = progress.add_task("ALL runs", total=grand_total)
        t_model = progress.add_task("Model", total=0)
        t_domain = progress.add_task("Domain", total=0)
        t_strategy = progress.add_task("Strategy", total=0)

        mode = (launch_mode or "auto").strip().lower()
        if mode not in {"auto", "vecinf", "manual"}:
            raise ValueError("launch_mode must be one of: auto | vecinf | manual")

        launcher: VecInfLauncher | None = None
        effective_mode = mode
        if mode in {"auto", "vecinf"}:
            try:
                launcher = VecInfLauncher()
                effective_mode = "vecinf"
            except ImportError as e:
                if mode == "vecinf":
                    raise
                console.print(
                    "[yellow][WARN][/yellow] vec_inf not importable; falling back to manual endpoint."
                )
                effective_mode = "manual"

        if effective_mode == "manual" and not (base_url or base_url_map):
            raise ValueError("Manual mode requires --base-url or --base-url-map.")

        if (
            effective_mode == "manual"
            and not base_url_map
            and len(list(models)) > 1
            and not bool(allow_shared_base_url)
        ):
            raise ValueError(
                "Manual mode with single URL cannot safely run multiple models without --allow-shared-base-url."
            )

        # Container for aggregation at Session level
        session_results: list[dict[str, float]] = []

        with mlflow_run(f"session_{stamp}", nested=False):
            log_run_common_tags(git_commit=git.commit, repo_dirty=git.dirty)
            mlflow.log_param("examples_dir", str(examples_dir))
            mlflow.log_param("val_path", str(val_path) if val_path else "")
            mlflow.log_param("val_flags", " ".join(val_flags))
            mlflow.log_param("tolerance", float(tolerance))
            mlflow.set_tag("endpoint_mode", effective_mode)

            for model_name in models:
                progress.reset(t_model, total=total_per_model)
                progress.update(t_model, completed=0, description=f"Model: {model_name}")
                phase_update(progress, t_phase, f"Preparing endpoint for {model_name}…")

                # Container for aggregation at Model level
                model_results: list[dict[str, float]] = []

                with mlflow_run(f"model={model_name}", nested=True):
                    log_run_common_tags(git_commit=git.commit, repo_dirty=git.dirty)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("temperature", 0.0)
                    mlflow.log_param("max_tokens", max_tokens_for(model_name))

                    if effective_mode == "vecinf":
                        assert launcher is not None
                        ep_ctx = launched_model(
                            launcher,
                            model_name,
                            on_phase=lambda s: phase_update(progress, t_phase, s),
                        )
                    else:
                        if base_url_map and model_name in base_url_map:
                            url = base_url_map[model_name]
                        elif base_url_map:
                             raise ValueError(f"Missing base URL for model {model_name!r}")
                        else:
                            url = base_url
                        ep_ctx = manual_model(
                            model_name,
                            url,
                            on_phase=lambda s: phase_update(progress, t_phase, s),
                        )

                    with ep_ctx as ep:
                        mlflow.set_tag("vecinf_job_id", ep.job_id)
                        mlflow.set_tag("base_url", ep.base_url)
                        mlflow.set_tag("endpoint_mode", effective_mode)
                        
                        client = OpenAICompatClient(
                            base_url=ep.base_url,
                            api_key=openai_api_key,
                            request_timeout_s=request_timeout_s,
                        )

                        for dom in domain_specs:
                            problem_specs = dom.problems
                            if problem_indices:
                                want = set(int(x) for x in problem_indices)
                                problem_specs = [p for p in dom.problems if int(p.index) in want]
                            if limit_problems:
                                problem_specs = problem_specs[:limit_problems]

                            domain_total = len(problem_specs) * len(strategies)
                            progress.reset(t_domain, total=domain_total)
                            progress.update(t_domain, completed=0, description=f"Domain: {dom.name}")

                            # Container for aggregation at Domain level
                            domain_results: list[dict[str, float]] = []

                            with mlflow_run(f"domain={dom.name}", nested=True):
                                log_run_common_tags(git_commit=git.commit, repo_dirty=git.dirty)
                                mlflow.log_param("domain", dom.name)
                                mlflow.log_param("prompts_path", str(dom.prompts_path))

                                for strat in strategies:
                                    if strat not in STRATEGIES:
                                        raise ValueError(
                                            f"Unknown strategy: {strat}. Known: {sorted(STRATEGIES)}"
                                        )

                                    progress.reset(t_strategy, total=len(problem_specs))
                                    progress.update(t_strategy, completed=0, description=f"Strategy: {strat}")

                                    # Container for aggregation at Strategy level
                                    strat_results: list[dict[str, float]] = []

                                    with mlflow_run(f"strategy={strat}", nested=True):
                                        log_run_common_tags(git_commit=git.commit, repo_dirty=git.dirty)
                                        mlflow.log_param("strategy", strat)

                                        for pr in problem_specs:
                                            phase_update(progress, t_phase, f"Running {model_name} • {dom.name} • {strat} • {pr.problem_id}")

                                            run_name = f"{dom.name}/{pr.problem_id}"
                                            with mlflow_run(run_name, nested=True):
                                                log_run_common_tags(git_commit=git.commit, repo_dirty=git.dirty)
                                                max_toks = max_tokens_for(model_name)
                                                
                                                mlflow.log_params({
                                                    "model_name": model_name,
                                                    "strategy": strat,
                                                    "temperature": 0.0,
                                                    "max_tokens": max_toks,
                                                    "problem_id": pr.problem_id,
                                                    "problem_index": pr.index,
                                                })

                                                mlflow.set_tag("domain", dom.name)
                                                mlflow.set_tag("strategy", strat)
                                                mlflow.set_tag("problem_id", pr.problem_id)
                                                if pr.difficulty:
                                                    mlflow.set_tag("difficulty", pr.difficulty)

                                                # Strategy execution
                                                t0 = time.time()
                                                raw_text = ""
                                                trace_text: str | None = None
                                                llm_error: str | None = None
                                                
                                                try:
                                                    out = run_strategy(
                                                        strat,
                                                        client=client,
                                                        model_name=model_name,
                                                        domain_prompt=dom.domain_prompt,
                                                        problem_prompt=pr.prompt,
                                                        temperature=0.0,
                                                        max_tokens=max_toks,
                                                    )
                                                    raw_text = out.final_text
                                                    trace_text = out.trace
                                                except Exception as e:
                                                    llm_error = str(e)
                                                    raw_text = f"[LLM_ERROR]\n{llm_error}\n"
                                                    trace_text = f"=== LLM_ERROR ===\n{llm_error}\n"
                                                    print(f"[LLM_ERROR] {llm_error}", file=sys.stderr)

                                                t1 = time.time()

                                                # Parse & Eval
                                                parse = parse_plan(dom.name, raw_text)
                                                
                                                # Determine Solvability
                                                optimal_cost_val = float(pr.optimal_cost) if pr.optimal_cost is not None else 0.0
                                                is_ground_truth_solvable = (optimal_cost_val != -1.0)

                                                cleaned_plan = parse.plan_text.strip().upper()
                                                model_claims_unsolvable = (
                                                    "UNSOLVABLE" in cleaned_plan 
                                                    or "UNSOLVABLE" in raw_text.strip().upper().splitlines()
                                                )

                                                metrics = None
                                                eval_error: str | None = None
                                                t2 = t1
                                                
                                                if parse.plan_text.strip() and not model_claims_unsolvable:
                                                    try:
                                                        metrics = evaluate_with_val(
                                                            domain_pddl=dom.domain_pddl,
                                                            problem_pddl=dom.problems_dir / f"problem{pr.index}.pddl",
                                                            plan_text=parse.plan_text,
                                                            val_path=val_path,
                                                            val_flags=val_flags,
                                                            tolerance=tolerance,
                                                            check_redundancy=False,
                                                            is_ground_truth_solvable=is_ground_truth_solvable, 
                                                        )
                                                    except Exception as e:
                                                        eval_error = str(e)
                                                        print(f"[LLM_ERROR] {e}")
                                                    t2 = time.time()
                                                else:
                                                    t2 = time.time()

                                                # Artifacts
                                                prob_out = (
                                                    session_out
                                                    / f"model={model_name}"
                                                    / f"domain={dom.name}"
                                                    / f"strategy={strat}"
                                                    / f"{pr.problem_id}"
                                                )
                                                arts = write_artifacts(
                                                    out_dir=prob_out,
                                                    raw_text=raw_text,
                                                    plan_text=parse.plan_text,
                                                    metrics=metrics,
                                                )
                                                if trace_text:
                                                    prob_out.mkdir(parents=True, exist_ok=True)
                                                    trace_path = prob_out / f"trace.txt"
                                                    trace_path.write_text(trace_text, encoding="utf-8")
                                                    mlflow.log_artifact(str(trace_path))

                                                mlflow.log_artifact(str(arts.raw_response_path))
                                                mlflow.log_artifact(str(arts.plan_path))
                                                mlflow.log_artifact(str(arts.val_stdout_path))
                                                mlflow.log_artifact(str(arts.val_stderr_path))
                                                mlflow.log_artifact(str(arts.val_trace_path))

                                                # Scoring Logic
                                                llm_dur = t1 - t0
                                                eval_dur = t2 - t1
                                                total_dur = t2 - t0
                                                is_valid_run = 1.0 if (llm_error is None and eval_error is None) else 0.0
                                                steps_taken = 0.0
                                                is_success = 0.0
                                                score = 0.0

                                                if model_claims_unsolvable:
                                                    if not is_ground_truth_solvable:
                                                        is_success = 1.0
                                                        score = 1.0
                                                        is_valid_run = 1.0
                                                    else:
                                                        is_success = 0.0
                                                        score = 0.0
                                                elif metrics:
                                                    steps_taken = float(metrics.length)
                                                    if not is_ground_truth_solvable:
                                                        is_success = 0.0
                                                        score = 0.0
                                                    else:
                                                        is_success = 1.0 if metrics.valid else 0.0
                                                        if is_success > 0.0 and optimal_cost_val > 0.0 and steps_taken > 0.0:
                                                            score = optimal_cost_val / steps_taken
                                                            if score > 1.0: score = 1.0
                                                        elif is_success > 0.0:
                                                            score = 1.0

                                                result = EvalResult(
                                                    success=int(is_success),
                                                    llm_duration_seconds=llm_dur,
                                                    eval_duration_seconds=eval_dur,
                                                    total_duration_seconds=total_dur,
                                                    parse=parse,
                                                    metrics=metrics,
                                                    failure=llm_error or eval_error,
                                                    artifacts=arts,
                                                )

                                                final_metrics = result.to_mlflow_metrics()
                                                final_metrics.update({
                                                    "is_valid": is_valid_run,
                                                    "is_success": is_success,
                                                    "steps_taken": steps_taken,
                                                    "optimal_steps": optimal_cost_val,
                                                    "score": score,
                                                    "wall_time": float(total_dur),
                                                    "is_ground_truth_solvable": 1.0 if is_ground_truth_solvable else 0.0
                                                })

                                                mlflow.log_metrics(final_metrics)
                                                
                                                # Accumulate for parent aggregation
                                                strat_results.append(final_metrics)

                                                # --- Extra scalar params/metrics (Restored) ---
                                                if pr.optimal_cost is not None:
                                                    mlflow.log_param("optimal_cost", pr.optimal_cost)
                                                    if (
                                                        metrics
                                                        and metrics.valid
                                                        and metrics.cost_value
                                                        and metrics.cost_value > 0
                                                    ):
                                                        mlflow.log_metric(
                                                            "score_opt_over_cost",
                                                            float(pr.optimal_cost) / float(metrics.cost_value),
                                                        )

                                                # --- Error visibility (Restored) ---
                                                if llm_error:
                                                    mlflow.set_tag("llm_error", llm_error[:5000])
                                                if eval_error:
                                                    mlflow.set_tag("eval_error", eval_error[:5000])
                                                if parse.parse_errors:
                                                    mlflow.set_tag(
                                                        "parse_errors",
                                                        "\n".join(parse.parse_errors)[:5000],
                                                    )
                                                if metrics and metrics.failure_reason:
                                                    mlflow.set_tag(
                                                        "val_failure_reason",
                                                        metrics.failure_reason,
                                                    )

                                            progress.advance(t_strategy, 1)
                                            progress.advance(t_domain, 1)
                                            progress.advance(t_model, 1)
                                            progress.advance(t_all, 1)

                                        # End Strategy Loop -> Log Aggregate
                                        _log_aggregate_metrics(strat_results)
                                        # Pass results up to domain container
                                        domain_results.extend(strat_results)

                                # End Domain Loop -> Log Aggregate
                                _log_aggregate_metrics(domain_results)
                                # Pass results up to model container
                                model_results.extend(domain_results)
                    
                    # End Model Loop -> Log Aggregate
                    _log_aggregate_metrics(model_results)
                    # Pass results up to session container
                    session_results.extend(model_results)

            # End Session Loop -> Log Global Aggregate
            _log_aggregate_metrics(session_results)
            phase_update(progress, t_phase, "Idle")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an AllxAll planning benchmark against green_agent."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "vecinf", "manual"],
        help="Endpoint mode.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--base-url-map",
        type=str,
        default=None,
        help="Mapping for manual mode e.g. 'ModelA=url1,ModelB=url2'.",
    )
    parser.add_argument(
        "--allow-shared-base-url",
        action="store_true",
        help="Allow multiple models on one URL in manual mode.",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default=None,
        help="Path to examples/.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domain names.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategies.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out_bench",
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to VAL binary.",
    )
    parser.add_argument(
        "--val-flags",
        type=str,
        default="-v",
        help="VAL flags.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="VAL numeric tolerance.",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="Planning_Benchmark_v1",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for OpenAI SDK.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=180.0,
        help="Timeout for LLM request.",
    )
    parser.add_argument(
        "--limit-problems",
        type=int,
        default=None,
        help="Only run first N problems per domain.",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        type=int,
        default=None,
        help="Run only these problem indices.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    git = get_git_info()
    examples_dir = Path(
        args.examples_dir) if args.examples_dir else Path(
        git.repo_root) / "examples"

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    else:
        domains = discover_domains(examples_dir)

    base_url_map = _parse_base_url_map(getattr(args, "base_url_map", None))
    base_url = str(getattr(args, "base_url", DEFAULT_BASE_URL)).strip()

    mode = str(getattr(args, "mode", "auto")).strip().lower()
    effective_mode_for_discovery = mode
    if mode in {"auto", "vecinf"}:
        try:
            _ = VecInfLauncher()
        except ImportError:
            if mode == "auto":
                effective_mode_for_discovery = "manual"

    models_str = str(args.models)
    models = [m.strip() for m in models_str.split(",") if m.strip()]

    # Auto-detect if manual mode and default models are unchanged
    if (
        effective_mode_for_discovery == "manual"
        and not base_url_map
        and models_str.strip() == ",".join(DEFAULT_MODELS)
    ):
        detected = try_list_models(
            base_url=base_url,
            api_key=str(args.openai_api_key),
            request_timeout_s=min(30.0, float(args.request_timeout_s)),
        )
        if detected:
            print(
                f"[INFO] Detected served model(s) at {base_url}: {detected}. "
                "Using those instead of default --models.",
                file=sys.stderr,
            )
            models = detected

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    val_flags = tuple([x for x in args.val_flags.split() if x.strip()])

    run_suite(
        examples_dir=examples_dir,
        domains=domains,
        models=models,
        strategies=strategies,
        out_dir=out_dir,
        val_path=args.val_path,
        val_flags=val_flags,
        tolerance=float(args.tolerance),
        mlflow_tracking_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        openai_api_key=args.openai_api_key,
        request_timeout_s=float(args.request_timeout_s),
        launch_mode=str(args.mode),
        base_url=base_url,
        base_url_map=base_url_map,
        allow_shared_base_url=bool(args.allow_shared_base_url),
        problem_indices=args.problems,
        limit_problems=args.limit_problems,
    )


if __name__ == "__main__":
    main()