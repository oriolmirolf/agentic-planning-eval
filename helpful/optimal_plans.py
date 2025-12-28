#!/usr/bin/env python3
"""
optimal_plans.py — Use a *real* planner (FF / Metric-FF) to get baseline plans.

Key fixes vs older version:
- Parse plan from stdout+stderr (some FF builds print to stderr).
- On failure, print tail of stdout/stderr (often FF errors go to stdout).
- Optionally save raw stdout/stderr even on failures (--save-raw).
- Support different FF CLI styles (try -o/-f and positional).

Usage examples:

  # All problems, FF default search (fast, not guaranteed optimal)
  python helpful/optimal_plans.py --domain blocks --planner ff

  # All problems, BFS mode (if your FF supports it; many use -g)
  python helpful/optimal_plans.py --domain blocks --planner bfs

  # Single problem
  python helpful/optimal_plans.py --domain blocks --index 1 --planner ff

Environment / paths:
- By default we look for an `ff` binary on PATH.
- Override with --ff-path, or env FF_PLANNER_PATH / FF_PATH.
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

from helpful.validate_all import find_problems  # reuse your problem discovery

load_dotenv()

_STEP0_RE = re.compile(r"^\s*step\s+\d+\s*:\s*(.+)$", re.IGNORECASE)
_STEPN_RE = re.compile(r"^\s*\d+\s*:\s*(.+)$")
# Some builds may output something like: "0: (PICKUP A)"
_STEPN_PARENS_RE = re.compile(r"^\s*\d+\s*:\s*\((.+)\)\s*$")

# Useful for detecting "wrong flags / help printed"
_USAGE_HINT_RE = re.compile(r"\b(usage:|options:|help)\b", re.IGNORECASE)


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


def _extract_plan_from_ff_output(stdout: str, stderr: str = "") -> str:
    """
    Parse FF output safely:
    1) Prefer explicit 'step ...:' / 'N:' plan lines (true FF plan).
    2) If none found, try parsing "(...)" action lines.
    """
    text = (stdout or "") + "\n" + (stderr or "")

    actions: list[str] = []

    # 1) FF-style step parsing
    for ln in text.splitlines():
        m = _STEP0_RE.match(ln) or _STEPN_RE.match(ln)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue

        # If body is already "(...)" keep it (normalize to lowercase)
        if body.startswith("(") and body.endswith(")"):
            inside = body[1:-1].strip().lower()
            if inside:
                actions.append(f"({inside})")
            continue

        # Otherwise, assume "OP ARG1 ARG2 ..."
        tokens = body.split()
        if not tokens:
            continue
        op = tokens[0].lower()
        args = " ".join(t.lower() for t in tokens[1:])
        actions.append(f"({op} {args})" if args else f"({op})")

    if actions:
        return "\n".join(actions) + "\n"

    # 2) Sometimes it prints "0: (PICKUP A)" but without matching above
    for ln in text.splitlines():
        m2 = _STEPN_PARENS_RE.match(ln)
        if not m2:
            continue
        inside = m2.group(1).strip().lower()
        if inside:
            actions.append(f"({inside})")
    if actions:
        return "\n".join(actions) + "\n"

    # 3) Generic fallback: scrape parenthesized actions anywhere
    paren_actions = re.findall(r"\(([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)*)\)", text)
    if paren_actions:
        return "\n".join(f"({a.strip().lower()})" for a in paren_actions) + "\n"

    return ""


def _tail(text: str, n: int = 30) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def run_ff_planner(
    *,
    ff_bin: str,
    domain_pddl: str,
    problem_pddl: str,
    mode: str,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
    ff_call: str = "auto",  # auto|of|positional
) -> tuple[str, str, str, int, list[str]]:
    """
    Invoke FF/Metric-FF on (domain, problem) with a given mode:

      mode == "ff"  -> default FF search
      mode == "bfs" -> BFS-ish mode (commonly -g)

    ff_call:
      - "of":         ff -o domain -f problem
      - "positional": ff domain problem
      - "auto":       try -o/-f first; if it looks like usage/wrong flags,
                      retry positional

    Returns: (plan_text, stdout, stderr, returncode, cmd_used)
    """
    if mode not in ("ff", "bfs"):
        raise ValueError(f"Unknown planner mode: {mode!r} (expected 'ff' or 'bfs')")

    def build_cmd(style: str) -> list[str]:
        if style == "positional":
            cmd = [ff_bin, domain_pddl, problem_pddl]
        else:
            cmd = [ff_bin, "-o", domain_pddl, "-f", problem_pddl]

        # For many FF builds, '-g' enforces BFS (shortest plan length).
        if mode == "bfs":
            cmd.append("-g")

        if extra_args:
            cmd.extend(extra_args)
        return cmd

    styles = [ff_call] if ff_call != "auto" else ["of", "positional"]

    last = ("", "", -999, [])  # stdout, stderr, rc, cmd
    for style in styles:
        cmd = build_cmd(style)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return "", e.stdout or "", e.stderr or f"Timeout after {timeout}s", -1, cmd

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode

        plan_text = _extract_plan_from_ff_output(stdout, stderr)

        # If we got a plan, accept immediately even if rc is odd.
        if plan_text.strip():
            return plan_text, stdout, stderr, rc, cmd

        # If it looks like "usage/help" likely wrong flag style → try next
        # style.
        if ff_call == "auto" and _USAGE_HINT_RE.search(stdout) and not stderr.strip():
            last = (stdout, stderr, rc, cmd)
            continue

        # Otherwise, keep last and stop.
        return plan_text, stdout, stderr, rc, cmd

    # if both failed under auto, return last attempt
    stdout, stderr, rc, cmd = last
    return "", stdout, stderr, rc, cmd


def _iter_problems(
    domain_dir: Path,
    index: int | None,
    start: int | None,
    end: int | None,
) -> list[tuple[int, Path]]:
    probs: list[tuple[int, Path]] = []

    if index is not None:
        p = domain_dir / "problems_pddl" / f"problem{index}.pddl"
        return [(index, p)]

    all_probs = find_problems(domain_dir)

    for p in all_probs:
        stem = p.stem
        num = None
        for token in stem.split("_"):
            if token.isdigit():
                num = int(token)
                break
        if num is None:
            m = re.search(r"(\d+)", stem)
            if m:
                num = int(m.group(1))
        if num is None:
            continue

        if start is not None and num < start:
            continue
        if end is not None and num > end:
            continue

        probs.append((num, p))

    return probs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Use FF/Metric-FF (fast or BFS-ish) to compute plans for "
            "a domain's problems."
        )
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
        help="Single problem index (problem{index}.pddl)",
    )
    ap.add_argument(
        "--start", type=int, default=None, help="First index when scanning all problems"
    )
    ap.add_argument(
        "--end", type=int, default=None, help="Last index when scanning all problems"
    )
    ap.add_argument(
        "--planner",
        choices=["ff", "bfs"],
        default="bfs",
        help="ff = fast heuristic search; bfs = BFS-ish (often -g) "
        "for shortest plan length.",
    )
    ap.add_argument(
        "--ff-path",
        default=None,
        help="Explicit path to FF/Metric-FF binary (else "
        "FF_PLANNER_PATH/FF_PATH/PATH).",
    )
    ap.add_argument(
        "--ff-call",
        choices=["auto", "of", "positional"],
        default="auto",
        help="How to call ff: auto tries -o/-f then positional; "
        "of forces -o/-f; positional forces 'ff domain problem'.",
    )
    ap.add_argument(
        "--out-dir", default="out/optimal_plans", help="Root dir to write .plan files"
    )
    ap.add_argument(
        "--timeout", type=float, default=None, help="Timeout per problem in seconds"
    )
    ap.add_argument(
        "--extra-args", nargs="*", default=None, help="Extra args passed to FF binary"
    )
    ap.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw stdout/stderr for every run (success OR failure).",
    )
    ap.add_argument(
        "--debug-fail",
        action="store_true",
        help="On failure, print the last ~30 lines of stdout and stderr.",
    )
    args = ap.parse_args()

    domain_dir = Path("examples") / args.domain
    domain_pddl = domain_dir / "domain.pddl"
    if not domain_pddl.exists():
        print(f"[ERROR] domain.pddl not found at: {domain_pddl}", file=sys.stderr)
        sys.exit(2)

    ff_bin = guess_ff_binary(args.ff_path)
    if not ff_bin:
        print(
            "[ERROR] FF/Metric-FF binary not found.\n"
            "  - Ensure 'ff' is on PATH, or\n"
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
        plan_text, stdout, stderr, rc, cmd_used = run_ff_planner(
            ff_bin=ff_bin,
            domain_pddl=str(domain_pddl),
            problem_pddl=str(prob_path),
            mode=args.planner,
            extra_args=args.extra_args,
            timeout=args.timeout,
            ff_call=args.ff_call,
        )

        # Save raw outputs always if requested
        if args.save_raw:
            raw_out = out_root / f"problem{idx}.planner_stdout.txt"
            raw_err = out_root / f"problem{idx}.planner_stderr.txt"
            raw_cmd = out_root / f"problem{idx}.planner_cmd.txt"
            raw_out.write_text(stdout, encoding="utf-8")
            raw_err.write_text(stderr, encoding="utf-8")
            raw_cmd.write_text(" ".join(cmd_used) + "\n", encoding="utf-8")

        if rc != 0 and not plan_text.strip():
            print(" FAILED (no plan, non-zero exit)")
            if args.debug_fail:
                if stdout.strip():
                    print("\n--- stdout (tail) ---")
                    print(_tail(stdout))
                if stderr.strip():
                    print("\n--- stderr (tail) ---")
                    print(_tail(stderr))
                print("\n--- cmd ---")
                print(" ".join(cmd_used))
                print()
            continue

        if not plan_text.strip():
            print(" FAILED (no actions parsed from planner output)")
            if args.debug_fail:
                if stdout.strip():
                    print("\n--- stdout (tail) ---")
                    print(_tail(stdout))
                if stderr.strip():
                    print("\n--- stderr (tail) ---")
                    print(_tail(stderr))
                print("\n--- cmd ---")
                print(" ".join(cmd_used))
                print()
            continue

        plan_path = out_root / f"problem{idx}.plan"
        plan_path.write_text(plan_text, encoding="utf-8")
        num_actions = len(
            [ln for ln in plan_text.splitlines() if ln.strip().startswith("(")]
        )

        successes += 1
        print(f" OK  (actions: {num_actions}, saved to {plan_path})")

    print()
    print(f"Done. Successful plans: {successes}/{len(problems)}")
    sys.exit(0 if successes == len(problems) else 1)


if __name__ == "__main__":
    main()
