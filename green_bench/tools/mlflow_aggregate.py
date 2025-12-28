#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mlflow
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggSpec:
    score_col_candidates: tuple[str, ...] = (
        "metrics.score_opt_over_cost",
        "metrics.plan_cost_value",  # fallback if you use this as "score"
    )
    duration_col_candidates: tuple[str, ...] = (
        "metrics.total_duration_seconds",
        "metrics.duration_seconds",
    )


def _pick_first_existing(df: pd.DataFrame, cols: Iterable[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _coalesce_cols(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    series = None
    for c in candidates:
        if c in df.columns:
            series = df[c] if series is None else series.combine_first(df[c])
    df[out_col] = series if series is not None else np.nan


def _quantile(s: pd.Series, q: float) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return float("nan")
    return float(s2.quantile(q))


def _aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    score_raw_col: str,
    score_valid_only_col: str,
    score_with_failures_col: str,
    dur_col: str,
) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)

    def _sum01(x: pd.Series) -> float:
        # x already numeric 0/1; sum is fine even if empty
        return float(np.nansum(x.to_numpy(dtype=float)))

    out = g.agg(
        n_runs=("run_id", "count"),
        n_success=("success_bin", _sum01),
        n_valid=("valid_bin", _sum01),
        success_rate=("success_bin", "mean"),
        valid_rate=("valid_bin", "mean"),
        # Explicitly separate:
        avg_score_valid_only=(score_valid_only_col, "mean"),   # mean over valid only (NaNs dropped)
        avg_score_with_failures=(score_with_failures_col, "mean"),  # invalid/failed => 0 included
        avg_score_raw=(score_raw_col, "mean"),  # legacy-style mean over raw score col (NaNs dropped)
        avg_total_s=(dur_col, "mean"),
        score_missing_for_valid_rate=("score_missing_for_valid", "mean"),
    ).reset_index()

    # quantiles (p50/p95) on duration
    p50 = g[dur_col].apply(lambda s: _quantile(s, 0.50)).reset_index(name="p50_total_s")
    p95 = g[dur_col].apply(lambda s: _quantile(s, 0.95)).reset_index(name="p95_total_s")
    out = out.merge(p50, on=group_cols, how="left").merge(p95, on=group_cols, how="left")

    # stable sorting
    out = out.sort_values(["n_runs"] + group_cols, ascending=[False] + [True] * len(group_cols))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help=(
            "Experiment names to include (e.g. Planning_Benchmark_v1 Planning_Benchmark_v2). "
            "If omitted, uses $MLFLOW_EXPERIMENT_NAME or 'Planning_Benchmark_v1'."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="mlflow_reports",
        help="Directory to write CSV reports into.",
    )
    ap.add_argument(
        "--child-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only child problem runs (requires tags.problem_id or params.problem_id). Default: true.",
    )
    args = ap.parse_args()

    # Resolve experiment names
    exp_names = args.experiments
    if not exp_names:
        exp_names = [os.environ.get("MLFLOW_EXPERIMENT_NAME", "Planning_Benchmark_v1")]

    # Resolve experiment IDs
    exp_ids: list[str] = []
    for name in exp_names:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            raise SystemExit(
                f"Experiment not found: {name}\n"
                f"Tip: check with: python -c \"import mlflow; print([e.name for e in mlflow.search_experiments()])\""
            )
        exp_ids.append(exp.experiment_id)

    runs = mlflow.search_runs(experiment_ids=exp_ids, output_format="pandas", max_results=200000)
    if runs.empty:
        raise SystemExit(f"No runs found for experiments: {exp_names} (ids={exp_ids})")

    # Normalize dimensions (prefer params, fallback to tags)
    _coalesce_cols(runs, "model_name", ["params.model_name", "tags.model_name"])
    _coalesce_cols(runs, "strategy", ["tags.strategy", "params.strategy"])
    _coalesce_cols(runs, "domain", ["tags.domain"])
    _coalesce_cols(runs, "problem_id", ["tags.problem_id", "params.problem_id"])

    # Filter to child runs (problem runs)
    if args.child_only:
        runs = runs[runs["problem_id"].notna()]

    spec = AggSpec()
    score_raw_col = _pick_first_existing(runs, spec.score_col_candidates) or "metrics.score_opt_over_cost"
    dur_col = _pick_first_existing(runs, spec.duration_col_candidates) or "metrics.total_duration_seconds"

    # Ensure columns exist (create NaNs if missing)
    for col in [score_raw_col, dur_col, "metrics.success", "metrics.valid"]:
        if col not in runs.columns:
            runs[col] = np.nan

    # Convert numeric columns
    runs[score_raw_col] = pd.to_numeric(runs[score_raw_col], errors="coerce")
    runs[dur_col] = pd.to_numeric(runs[dur_col], errors="coerce")

    # IMPORTANT: treat missing success/valid as 0 (so they count as failures)
    runs["success_bin"] = pd.to_numeric(runs["metrics.success"], errors="coerce").fillna(0.0).clip(0, 1)
    runs["valid_bin"] = pd.to_numeric(runs["metrics.valid"], errors="coerce").fillna(0.0).clip(0, 1)

    # Score variants:
    # - valid-only: only keep score where valid==1 & success==1; otherwise NaN (so mean is over valid subset)
    runs["score_valid_only"] = runs[score_raw_col].where((runs["success_bin"] == 1.0) & (runs["valid_bin"] == 1.0))

    # - score with failures: if (success==1 & valid==1) use score (missing score -> 0), else 0
    runs["score_with_failures"] = np.where(
        (runs["success_bin"] == 1.0) & (runs["valid_bin"] == 1.0),
        runs[score_raw_col].fillna(0.0),
        0.0,
    )

    # Detect “valid but missing score”
    runs["score_missing_for_valid"] = np.where(
        (runs["success_bin"] == 1.0) & (runs["valid_bin"] == 1.0) & (runs[score_raw_col].isna()),
        1.0,
        0.0,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "by_model": _aggregate(runs, ["model_name"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
        "by_strategy": _aggregate(runs, ["strategy"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
        "by_domain": _aggregate(runs, ["domain"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
        "by_domain_strategy": _aggregate(runs, ["domain", "strategy"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
        "by_model_strategy": _aggregate(runs, ["model_name", "strategy"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
        "by_model_domain": _aggregate(runs, ["model_name", "domain"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
        "by_model_domain_strategy": _aggregate(runs, ["model_name", "domain", "strategy"], score_raw_col, "score_valid_only", "score_with_failures", dur_col),
    }

    for name, df in tables.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))
        df.to_csv(out_dir / f"{name}.csv", index=False)

    print(f"\nWrote reports to: {out_dir}/")
    print(f"Including experiments: {exp_names} (ids={exp_ids})")
    print(
        "\nNotes:\n"
        "- avg_score_valid_only: mean score over runs that were success==1 AND valid==1\n"
        "- avg_score_with_failures: invalid/failed runs contribute score=0 (no NaN inflation)\n"
        "- score_missing_for_valid_rate: fraction of runs where success==1 & valid==1 but score was missing\n"
    )


if __name__ == "__main__":
    main()
