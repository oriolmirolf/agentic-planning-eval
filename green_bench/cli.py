from __future__ import annotations

import argparse
import os
import sys
import time
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
}

DEFAULT_MODELS = [
    "Kimi-K2-Thinking",
    "Qwen3-Coder-480B-A35B-Instruct",
    "Qwen2.5-72B-Instruct",
    "Qwen3-VL-30B-A3B-Thinking",
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
    return model_name in THINKING_MODELS or "thinking" in (
        model_name or "").lower()


def max_tokens_for(model_name: str) -> int:
    return 8192 if is_thinking_model(model_name) else 4096


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
                f"Invalid --base-url-map entry {
                    chunk!r}. Expected 'MODEL=URL'."
            )
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(
                f"Invalid --base-url-map entry {
                    chunk!r}. Expected 'MODEL=URL'."
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
    """Use a pre-launched OpenAI-compatible endpoint (e.g., SSH tunnel).

    This mode does NOT attempt to import or use vec_inf.
    """
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
    # Hard constraint: git commit hash must be logged as tag 'git_commit'
    set_required_git_tag(git_commit)
    mlflow.set_tag("repo_dirty", str(bool(repo_dirty)).lower())


# -----------------------------
# Core runner
# -----------------------------


def _load_domains(examples_dir: Path,
                  domain_names: Sequence[str]) -> list[DomainSpec]:
    return [load_domain(examples_dir, dn) for dn in domain_names]


def _ensure_repo_on_syspath(repo_root: str) -> None:
    # Allows importing green_agent even if user runs this from outside repo
    # root.
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

    # Resolve git + ensure green_agent importability
    git = get_git_info()
    if git.dirty:
        console.print(
            f"[yellow][WARN][/yellow] Git repo is dirty (uncommitted changes). commit={
                git.commit}"
        )

    _ensure_repo_on_syspath(git.repo_root)

    # Lazy import: these depend on green_agent being importable
    from .evaluator import EvalResult, evaluate_with_val, parse_plan, write_artifacts

    # MLflow setup
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)


    stamp = time.strftime("%Y%m%d-%H%M%S")
    session_out = out_dir / f"planning_benchmark_{stamp}"
    session_out.mkdir(parents=True, exist_ok=True)

    # Resolve domains
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

    problems_by_domain: dict[str, int] = {
        d.name: n_problems(d) for d in domain_specs}
    total_per_model = sum(
        problems_by_domain[d.name] * len(strategies) for d in domain_specs)
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
        t_phase = progress.add_task("Phase: idle", total=None)  # indeterminate
        t_all = progress.add_task("ALL runs", total=grand_total)
        t_model = progress.add_task("Model", total=0)
        t_domain = progress.add_task("Domain", total=0)
        t_strategy = progress.add_task("Strategy", total=0)

        # Decide how endpoints are provided.
        # - vecinf: launch/wait/shutdown via vec_inf
        # - manual: assume a pre-launched OpenAI-compatible endpoint (e.g., SSH tunnel)
        # - auto: try vecinf, fall back to manual if vec_inf isn't importable
        mode = (launch_mode or "auto").strip().lower()
        if mode not in {"auto", "vecinf", "manual"}:
            raise ValueError(
                "launch_mode must be one of: auto | vecinf | manual")

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
                    "[yellow][WARN][/yellow] vec_inf not importable in this Python environment; "  # noqa: E501
                    "falling back to manual endpoint mode. "
                    "Pass --mode manual --base-url http://localhost:5679/v1 to silence this warning."  # noqa: E501
                )
                console.print(
                    f"[yellow][WARN][/yellow] vec_inf import error: {e}")
                effective_mode = "manual"

        if effective_mode == "manual" and not (base_url or base_url_map):
            raise ValueError(
                "Manual mode requires --base-url or --base-url-map "
                "(or set GREEN_BENCH_BASE_URL)."
            )

        if (
            effective_mode == "manual"
            and not base_url_map
            and len(list(models)) > 1
            and not bool(allow_shared_base_url)
        ):
            raise ValueError(
                "Manual endpoint mode with a single --base-url cannot safely run multiple models. "  # noqa: E501
                "Either: (1) pass --models <served-model-name> to run just the currently launched model, "  # noqa: E501
                "or (2) provide --base-url-map 'MODEL=URL,...' for multiple tunnels/endpoints, "  # noqa: E501
                "or (3) pass --allow-shared-base-url if you truly host multiple models behind one endpoint."  # noqa: E501
            )

        with mlflow_run(f"session_{stamp}", nested=False):
            log_run_common_tags(git_commit=git.commit, repo_dirty=git.dirty)
            mlflow.log_param("examples_dir", str(examples_dir))
            mlflow.log_param("val_path", str(val_path) if val_path else "")
            mlflow.log_param("val_flags", " ".join(val_flags))
            mlflow.log_param("tolerance", float(tolerance))
            mlflow.set_tag("endpoint_mode", effective_mode)

            for model_name in models:
                # Reset per-model progress
                progress.reset(t_model, total=total_per_model)
                progress.update(
                    t_model,
                    completed=0,
                    description=f"Model: {model_name}")
                phase_update(
                    progress,
                    t_phase,
                    f"Preparing endpoint for {model_name}…")

                with mlflow_run(f"model={model_name}", nested=True):
                    log_run_common_tags(
                        git_commit=git.commit, repo_dirty=git.dirty)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("temperature", 0.0)
                    mlflow.log_param("max_tokens", max_tokens_for(model_name))

                    # Resolve a model endpoint (either launched via vec_inf or
                    # pre-launched)
                    if effective_mode == "vecinf":
                        assert launcher is not None
                        ep_ctx = launched_model(
                            launcher,
                            model_name,
                            on_phase=lambda s: phase_update(
                                progress, t_phase, s),
                        )
                    else:
                        if base_url_map:
                            if model_name not in base_url_map:
                                raise ValueError(
                                    f"Missing base URL for model {
                                        model_name!r} in --base-url-map. "
                                    "Either add it or pass a single --base-url and a single --models."  # noqa: E501
                                )
                            url = base_url_map[model_name]
                        else:
                            url = base_url
                        ep_ctx = manual_model(
                            model_name,
                            url,
                            on_phase=lambda s: phase_update(
                                progress, t_phase, s),
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

                        # Per-domain loop
                        for dom in domain_specs:
                            problem_specs = dom.problems
                            if problem_indices:
                                want = set(int(x) for x in problem_indices)
                                problem_specs = [p for p in dom.problems if int(p.index) in want]
                            if limit_problems:
                                problem_specs = problem_specs[:limit_problems]

                            # Reset per-domain progress for this model
                            domain_total = len(problem_specs) * len(strategies)
                            progress.reset(t_domain, total=domain_total)
                            progress.update(
                                t_domain,
                                completed=0,
                                description=f"Domain: {
                                    dom.name}"
                            )

                            with mlflow_run(f"domain={dom.name}", nested=True):
                                log_run_common_tags(
                                    git_commit=git.commit, repo_dirty=git.dirty)
                                mlflow.log_param("domain", dom.name)
                                mlflow.log_param(
                                    "prompts_path", str(
                                        dom.prompts_path))

                                for strat in strategies:
                                    if strat not in STRATEGIES:
                                        raise ValueError(
                                            f"Unknown strategy: {strat}. Known: {sorted(STRATEGIES)}"
                                        )

                                    # Reset per-strategy progress (within this
                                    # domain)
                                    progress.reset(
                                        t_strategy, total=len(problem_specs))
                                    progress.update(
                                        t_strategy,
                                        completed=0,
                                        description=f"Strategy: {strat}",
                                    )

                                    with mlflow_run(f"strategy={strat}", nested=True):
                                        log_run_common_tags(
                                            git_commit=git.commit, repo_dirty=git.dirty)
                                        mlflow.log_param("strategy", strat)

                                        for pr in problem_specs:
                                            phase_update(
                                                progress,
                                                t_phase,
                                                f"Running {model_name} • {
                                                    dom.name} • {strat} • {
                                                    pr.problem_id}",
                                            )

                                            run_name = f"{dom.name}/{pr.problem_id}"
                                            with mlflow_run(run_name, nested=True):
                                                log_run_common_tags(
                                                    git_commit=git.commit,
                                                    repo_dirty=git.dirty,
                                                )

                                                max_toks = max_tokens_for(
                                                    model_name)

                                                # Required params
                                                mlflow.log_params(
                                                    {
                                                        "model_name": model_name,
                                                        "strategy": strat,
                                                        "temperature": 0.0,
                                                        "max_tokens": max_toks,
                                                        "problem_id": pr.problem_id,
                                                        "problem_index": pr.index,
                                                    }
                                                )

                                                # Helpful tags for filtering
                                                mlflow.set_tag(
                                                    "domain", dom.name)
                                                mlflow.set_tag(
                                                    "strategy", strat)
                                                mlflow.set_tag(
                                                    "problem_id", pr.problem_id)
                                                if pr.difficulty:
                                                    mlflow.set_tag(
                                                        "difficulty", pr.difficulty)

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

                                                # Parse plan
                                                parse = parse_plan(
                                                    dom.name, raw_text)
                                                success = 1 if parse.plan_text.strip() else 0  # noqa: E501

                                                # VAL evaluation
                                                t2 = t1
                                                metrics = None
                                                eval_error: str | None = None
                                                if parse.plan_text.strip():
                                                    try:
                                                        metrics = evaluate_with_val(
                                                            domain_pddl=dom.domain_pddl,
                                                            problem_pddl=dom.problems_dir /  # noqa: E501
                                                            f"problem{
                                                                pr.index}.pddl",
                                                            plan_text=parse.plan_text,
                                                            val_path=val_path,
                                                            val_flags=val_flags,
                                                            tolerance=tolerance,
                                                            check_redundancy=False,
                                                        )
                                                    except Exception as e:
                                                        eval_error = str(e)
                                                        print(f"[LLM_ERROR] {e}")
                                                    t2 = time.time()
                                                else:
                                                    t2 = time.time()

                                                # Write artifacts locally
                                                # (always include raw response)
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

                                                # Log required artifact (raw
                                                # response)
                                                mlflow.log_artifact(
                                                    str(arts.raw_response_path))

                                                # Useful additional artifacts
                                                mlflow.log_artifact(
                                                    str(arts.plan_path))
                                                mlflow.log_artifact(
                                                    str(arts.val_stdout_path))
                                                mlflow.log_artifact(
                                                    str(arts.val_stderr_path))
                                                mlflow.log_artifact(
                                                    str(arts.val_trace_path))

                                                # Metrics (required + extras)
                                                llm_dur = t1 - t0
                                                eval_dur = (
                                                    t2 - t1) if parse.plan_text.strip() else 0.0  # noqa: E501
                                                total_dur = t2 - t0

                                                result = EvalResult(
                                                    success=success,
                                                    llm_duration_seconds=llm_dur,
                                                    eval_duration_seconds=eval_dur,
                                                    total_duration_seconds=total_dur,
                                                    parse=parse,
                                                    metrics=metrics,
                                                    failure=llm_error or eval_error,
                                                    artifacts=arts,
                                                )
                                                mlflow.log_metrics(
                                                    result.to_mlflow_metrics())

                                                is_valid = 1.0 if (llm_error is None and eval_error is None) else 0.0
                                                optimal_steps = float(pr.optimal_cost or 0.0)

                                                if metrics is not None:
                                                    steps_taken = float(metrics.length)
                                                    is_success = 1.0 if bool(metrics.valid) else 0.0
                                                else:
                                                    steps_taken = 0.0
                                                    is_success = 0.0

                                                score = 0.0
                                                if is_success > 0.0 and steps_taken > 0.0 and optimal_steps > 0.0:
                                                    score = optimal_steps / steps_taken
                                                    if score > 1.0:
                                                        score = 1.0

                                                wall_time = float(total_dur)
                                                mlflow.log_metrics(
                                                    {
                                                        "is_valid": is_valid,
                                                        "is_success": is_success,
                                                        "steps_taken": steps_taken,
                                                        "optimal_steps": optimal_steps,
                                                        "score": score,
                                                        "wall_time": wall_time,
                                                    }
                                                )

                                                # Extra scalar params/metrics
                                                if pr.optimal_cost is not None:
                                                    mlflow.log_param(
                                                        "optimal_cost", pr.optimal_cost)
                                                    if (
                                                        metrics
                                                        and metrics.valid
                                                        and metrics.cost_value
                                                        and metrics.cost_value > 0
                                                    ):
                                                        mlflow.log_metric(
                                                            "score_opt_over_cost",
                                                            float(
                                                                pr.optimal_cost) / float(metrics.cost_value),  # noqa: E501
                                                        )

                                                # Error visibility
                                                if llm_error:
                                                    mlflow.set_tag(
                                                        "llm_error", llm_error[:5000])
                                                if eval_error:
                                                    mlflow.set_tag(
                                                        "eval_error", eval_error[:5000])
                                                if parse.parse_errors:
                                                    mlflow.set_tag(
                                                        "parse_errors",
                                                        "\n".join(parse.parse_errors)[
                                                            :5000],
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

                phase_update(progress, t_phase, "Idle")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an AllxAll planning benchmark "
            "(models x prompting strategies x domains x problems) "
            "against your green_agent (VAL-based plan validation)."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "vecinf", "manual"],
        help=(
            "How to obtain model endpoints: 'vecinf' launches via vec_inf; "
            "'manual' uses a pre-launched OpenAI-compatible endpoint "
            "(e.g., SSH tunnel); 'auto' tries vec_inf then falls back to manual."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=(
            "OpenAI-compatible base URL for manual mode (default: %(default)s). "
            "You can set GREEN_BENCH_BASE_URL to override."
        ),
    )
    parser.add_argument(
        "--base-url-map",
        type=str,
        default=None,
        help=(
            "Optional per-model endpoint mapping for manual mode, e.g. "
            "'Qwen2.5-72B-Instruct=http://localhost:5679/v1,"
            "Kimi-K2-Thinking=http://localhost:5680/v1'."
        ),
    )
    parser.add_argument(
        "--allow-shared-base-url",
        action="store_true",
        help=(
            "Allow running multiple --models against a single --base-url in "
            "manual mode. "
            "Not recommended unless your endpoint truly hosts multiple models."
        ),
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default=None,
        help="Path to the repo's examples/ directory (default: <git-root>/examples).",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domain names. Default: auto-discover under examples/.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated vec_inf model names.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help=f"Comma-separated strategies. Known: {sorted(STRATEGIES)}",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out_bench",
        help="Local directory for artifacts (raw responses, plans, VAL logs).",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help=(
            "Optional path to the VAL Validate binary. If omitted, uses VAL_PATH env "
            "or PATH lookup."
        ),
    )
    parser.add_argument(
        "--val-flags",
        type=str,
        default="-v",
        help=(
            "VAL flags, space-separated (default: '-v'). Avoid '-e' for stability "
            "unless you know it's safe."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="VAL numeric tolerance (passed via -t).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="Optional MLflow tracking URI. If omitted, uses MLflow defaults/env.",
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
        help="API key for OpenAI SDK. For local servers, 'EMPTY' is usually fine.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=180.0,
        help="Timeout for a single LLM request (seconds).",
    )
    parser.add_argument(
        "--limit-problems",
        type=int,
        default=None,
        help="Optional: only run the first N problems per domain (smoke test).",
    )

    parser.add_argument(
        "--problems",
        nargs="*",
        type=int,
        default=None,
        help="Optional: run only these problem indices (per domain), e.g. --problems 1 2 5",
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

    # If running in manual mode (or auto-mode without vec_inf installed),
    # attempt to auto-detect the served model(s) at the endpoint.
    mode = str(getattr(args, "mode", "auto")).strip().lower()
    effective_mode_for_discovery = mode
    if mode in {"auto", "vecinf"}:
        try:
            _ = VecInfLauncher()  # just to test import availability
        except ImportError:
            if mode == "auto":
                effective_mode_for_discovery = "manual"

    models_str = str(args.models)
    models = [m.strip() for m in models_str.split(",") if m.strip()]

    # Only auto-detect if user did not override --models AND we are
    # effectively manual.
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
