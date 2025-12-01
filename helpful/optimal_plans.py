#!/usr/bin/env python3
"""
optimal_plans.py — Use a *real* planner (FF) to get baseline plans.

Usage examples:

  # Single problem, FF default search (fast, not guaranteed optimal)
  python optimal_plans.py --domain blocks --index 1 --planner ff

  # Single problem, BFS search (slower, but optimal in number of steps)
  python optimal_plans.py --domain blocks --index 1 --planner bfs

  # All problems in a domain with BFS
  python optimal_plans.py --domain blocks --planner bfs

  # Restrict to a range of indices
  python optimal_plans.py --domain blocks --planner bfs --start 1 --end 10

Environment / paths:

- By default we look for an `ff` binary on PATH.
- You can override with:
    --ff-path /path/to/ff
  or by setting env var FF_PLANNER_PATH or FF_PATH.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from green_agent.plan_parser import extract_plan
from helpful.validate_all import find_problems  # reuse your problem discovery

load_dotenv()

_STEP0_RE = re.compile(r"^\s*step\s+\d+\s*:\s*(.+)$", re.IGNORECASE)
_STEPN_RE = re.compile(r"^\s*\d+\s*:\s*(.+)$")


def guess_ff_binary(explicit: str | None) -> str | None:
    """
    Try to locate the FF planner binary.

    Resolution order:
    1) explicit --ff-path argument
    2) env var FF_PLANNER_PATH
    3) env var FF_PATH
    4) 'ff' on PATH
    """
    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_file():
            return str(p)

    envp = os.getenv("FF_PLANNER_PATH") or os.getenv("FF_PATH")
    if envp and Path(envp).exists():
        return envp

    for name in ("ff", "FF"):
        path = shutil.which(name)
        if path:
            return path

    return None


def _extract_plan_from_stdout(stdout: str) -> str:
    """
    1) Try the generic extractor (for LLM / VAL-style '(<op> ...)' lines).
    2) If that yields nothing, parse FF's 'step N: OP ARG ...' output and
       convert it into VAL-style '(op arg ...)' actions.
    """
    # 1) Try normal plan-extractor (handles fenced code blocks, etc.)
    extracted = extract_plan(stdout or "")
    if extracted.steps:
        return extracted.to_val_plan_text()

    # 2) Fallback for FF-style plans
    actions: list[str] = []
    for ln in (stdout or "").splitlines():
        m = _STEP0_RE.match(ln) or _STEPN_RE.match(ln)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue

        tokens = body.split()
        op = tokens[0].lower()
        args = " ".join(t.lower() for t in tokens[1:])
        if args:
            actions.append(f"({op} {args})")
        else:
            actions.append(f"({op})")

    if not actions:
        return ""

    return "\n".join(actions) + "\n"


def run_ff_planner(
    *,
    ff_bin: str,
    domain_pddl: str,
    problem_pddl: str,
    mode: str,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> tuple[str, str, str, int]:
    """
    Invoke FF on (domain, problem) with a given mode:

      mode == "ff"  -> default FF search
      mode == "bfs" -> BFS search (slower, optimal in steps)

    Returns: (plan_text, stdout, stderr, returncode)
    """
    if mode not in ("ff", "bfs"):
        raise ValueError(f"Unknown planner mode: {mode!r} (expected 'ff' or 'bfs')")

    args: list[str] = [ff_bin, "-o", domain_pddl, "-f", problem_pddl]

    # For many FF builds, '-g' enforces BFS search (shortest plan length).
    # If your local FF uses different flags, adjust here.
    if mode == "bfs":
        args.append("-g")

    if extra_args:
        args.extend(extra_args)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return "", e.stdout or "", e.stderr or f"Timeout after {timeout}s", -1

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    plan_text = _extract_plan_from_stdout(stdout)
    return plan_text, stdout, stderr, proc.returncode


def _iter_problems(
    domain_dir: Path,
    index: int | None,
    start: int | None,
    end: int | None,
) -> list[tuple[int, Path]]:
    """
    Yield (index, problem_path) pairs.

    If index is provided -> just that problem.
    Otherwise, we use helpful.validate_all.find_problems() and
    optionally filter by [start, end].
    """
    probs: list[tuple[int, Path]] = []

    if index is not None:
        p = domain_dir / "problems_pddl" / f"problem{index}.pddl"
        probs.append((index, p))
        return probs

    all_probs = find_problems(domain_dir)

    for p in all_probs:
        # Extract numeric suffix (same heuristic as validate_all.sort_key)
        stem = p.stem
        num = None
        for token in stem.split("_"):
            if token.isdigit():
                num = int(token)
                break
        if num is None:
            # fallback: try to scrape first integer anywhere
            import re

            m = re.search(r"(\d+)", stem)
            if m:
                num = int(m.group(1))
        if num is None:
            # skip if we really can't identify an index
            continue

        if start is not None and num < start:
            continue
        if end is not None and num > end:
            continue

        probs.append((num, p))

    return probs


def main():
    ap = argparse.ArgumentParser(
        description="Use FF (fast or BFS) to compute plans for a domain's problems."
    )
    ap.add_argument(
        "--domain",
        required=True,
        help="Domain folder under ./examples (e.g., 'blocks')",
    )
    ap.add_argument(
        "--index",
        type=int,
        default=None,
        help="Single problem index (problem{index}.pddl). "
        "If omitted, all problems are processed"
        " (optionally filtered by --start/--end).",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help="First index to include when scanning all problems"
        " (ignored if --index is set).",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last index to include when scanning all problems"
        " (ignored if --index is set).",
    )
    ap.add_argument(
        "--planner",
        choices=["ff", "bfs"],
        default="bfs",
        help="ff = fast heuristic search (not guaranteed optimal); "
        "bfs = BFS search (slower, but shortest plan in #actions).",
    )
    ap.add_argument(
        "--ff-path",
        default=None,
        help="Explicit path to FF planner binary"
        "(otherwise uses FF_PLANNER_PATH / FF_PATH / PATH).",
    )
    ap.add_argument(
        "--out-dir",
        default="out/optimal_plans",
        help="Root directory to write .plan files (default: out/optimal_plans).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds per problem for the planner.",
    )
    ap.add_argument(
        "--extra-args",
        nargs="*",
        default=None,
        help="Extra arguments passed directly to the FF binary.",
    )
    ap.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save raw stdout/stderr from the planner next to the .plan file.",
    )
    args = ap.parse_args()

    # Resolve domain.pddl
    domain_dir = Path("examples") / args.domain
    domain_pddl = domain_dir / "domain.pddl"
    if not domain_pddl.exists():
        print(f"[ERROR] domain.pddl not found at: {domain_pddl}", file=sys.stderr)
        sys.exit(2)

    ff_bin = guess_ff_binary(args.ff_path)
    if not ff_bin:
        print(
            "[ERROR] FF planner binary not found.\n"
            "  - Install the FF planner and ensure 'ff' is on PATH, or\n"
            "  - Set env FF_PLANNER_PATH / FF_PATH, or\n"
            "  - Pass --ff-path /path/to/ff",
            file=sys.stderr,
        )
        sys.exit(2)

    problems = _iter_problems(domain_dir, args.index, args.start, args.end)
    if not problems:
        print(
            f"[ERROR] No problems found for domain '{args.domain}' with given filters.",
            file=sys.stderr,
        )
        sys.exit(2)

    out_root = Path(args.out_dir) / args.planner / args.domain
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"FF binary: {ff_bin}")
    print(f"Domain:    {domain_pddl}")
    print(f"Planner:   {args.planner}  (ff=fast, bfs=slower but step-optimal)")
    print(f"Problems:  {len(problems)}")
    print(f"Output:    {out_root}")
    print()

    successes = 0
    for idx, prob_path in problems:
        if not prob_path.exists():
            print(
                f"[SKIP] problem{idx}: file not found at {prob_path}", file=sys.stderr
            )
            continue

        print(f"[RUN] problem{idx} ({prob_path.name}) ...", end="", flush=True)
        plan_text, stdout, stderr, rc = run_ff_planner(
            ff_bin=ff_bin,
            domain_pddl=str(domain_pddl),
            problem_pddl=str(prob_path),
            mode=args.planner,
            extra_args=args.extra_args,
            timeout=args.timeout,
        )

        if rc != 0 and not plan_text.strip():
            print(" FAILED (no plan, non-zero exit)")
            # Optional: print brief error
            if stderr.strip():
                print(f"  stderr: {stderr.strip().splitlines()[-1]}", file=sys.stderr)
            continue

        if not plan_text.strip():
            print(" FAILED (no actions parsed from planner output)")
            continue

        # Save .plan
        plan_path = out_root / f"problem{idx}.plan"
        plan_path.write_text(plan_text, encoding="utf-8")
        num_actions = len(
            [ln for ln in plan_text.splitlines() if ln.strip().startswith("(")]
        )

        if args.save_raw:
            raw_out = out_root / f"problem{idx}.planner_stdout.txt"
            raw_err = out_root / f"problem{idx}.planner_stderr.txt"
            raw_out.write_text(stdout, encoding="utf-8")
            raw_err.write_text(stderr, encoding="utf-8")

        successes += 1
        print(f" OK  (actions: {num_actions}, saved to {plan_path})")

    print()
    print(f"Done. Successful plans: {successes}/{len(problems)}")
    # Exit non-zero if any problem failed
    sys.exit(0 if successes == len(problems) else 1)


if __name__ == "__main__":
    main()
