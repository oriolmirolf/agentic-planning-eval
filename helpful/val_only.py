from __future__ import annotations
import argparse, json, time, sys, re
from pathlib import Path
from typing import Tuple, Optional

# Uses your existing retry+warning logic in val_wrapper.run_val
from green_agent.val_wrapper import run_val

_TS_RE = re.compile(r"^\d{8}-\d{6}$")  # e.g., 20251105-101112


def _find_last_batch_root(domain: str, base_out: Path = Path("out")) -> Optional[Path]:
    """
    Find the newest out/<domain>-YYYYMMDD-HHMMSS folder for this domain.
    """
    if not base_out.exists():
        return None
    candidates: list[Path] = []
    prefix = f"{domain}-"
    for p in base_out.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            ts = p.name[len(prefix):]
            if _TS_RE.match(ts):
                candidates.append(p)
    if not candidates:
        # fallback: newest dir with that prefix
        candidates = [p for p in base_out.iterdir() if p.is_dir() and p.name.startswith(prefix)]
        if not candidates:
            return None
        return max(candidates, key=lambda pp: pp.stat().st_mtime)

    def _parse_ts(pp: Path) -> tuple[int, int, int, int, int, int]:
        ts = pp.name[len(prefix):]
        try:
            y = int(ts[0:4]); m = int(ts[4:6]); d = int(ts[6:8])
            H = int(ts[9:11]); M = int(ts[11:13]); S = int(ts[13:15])
            return (y, m, d, H, M, S)
        except Exception:
            return (0, 0, 0, 0, 0, 0)

    return max(candidates, key=_parse_ts)


def _paths(domain: str, index: int, run_root: Optional[str], plan: Optional[str]) -> Tuple[Path, Path, Path, Path]:
    """
    Resolve domain/problem/plan/output paths.

    Special: if --run-root is 'last', auto-pick the latest batch for the domain under ./out/.
    """
    domain_dir = Path("examples") / domain
    domain_pddl = domain_dir / "domain.pddl"
    problem_pddl = domain_dir / "problems_pddl" / f"problem{index}.pddl"

    if plan:
        plan_path = Path(plan)
        out_dir = plan_path.parent / f"recheck-{time.strftime('%Y%m%d-%H%M%S')}"
        return domain_pddl, problem_pddl, plan_path, out_dir

    if not run_root:
        raise SystemExit("--run-root is required when --plan is not provided. You can use --run-root last")

    if run_root.strip().lower() == "last":
        auto_root = _find_last_batch_root(domain)
        if auto_root is None:
            raise SystemExit(f"Could not auto-detect last batch for domain '{domain}' under ./out/.")
        run_root_path = auto_root
    else:
        run_root_path = Path(run_root)

    prob_dir = run_root_path / f"p{index:02d}"
    plan_path = prob_dir / "purple.plan"
    out_dir = prob_dir / f"recheck-{time.strftime('%Y%m%d-%H%M%S')}"
    return domain_pddl, problem_pddl, plan_path, out_dir


def main():
    ap = argparse.ArgumentParser(description="Re-run VAL on an existing plan (no LLM involved).")
    ap.add_argument("--domain", required=True, help="Domain folder under ./examples (e.g., blocks)")
    ap.add_argument("--index", type=int, required=True, help="Problem index (e.g., 3)")
    ap.add_argument("--run-root", help="Batch folder from evaluate-domain (e.g., out/blocks-YYYYMMDD-HHMMSS). Use 'last' to auto-pick the latest for this domain.")
    ap.add_argument("--plan", help="Explicit path to .plan (overrides --run-root default)")
    ap.add_argument("--val-path", help="Path to Validate binary (optional)")
    ap.add_argument("--tolerance", type=float, default=0.001, help="VAL -t epsilon (default: 0.001)")
    # DEFAULT: no plan repair advice. You can opt-in with --advice.
    ap.add_argument("--advice", action="store_true", help="Enable VAL plan repair advice (-e). Avoid for stability.")
    ap.add_argument("--no-verbose", action="store_true", help="Do not pass -v (quiet). (arg name is 'no_verbose')")
    ap.add_argument("--json", action="store_true", help="Print a final JSON record to stdout")
    args = ap.parse_args()

    domain_pddl, problem_pddl, plan_path, out_dir = _paths(args.domain, args.index, args.run_root, args.plan)

    # preflight
    if not domain_pddl.exists():
        print(f"[ERROR] domain.pddl not found: {domain_pddl}", file=sys.stderr)
        sys.exit(2)
    if not problem_pddl.exists():
        print(f"[ERROR] problem.pddl not found: {problem_pddl}", file=sys.stderr)
        sys.exit(2)
    if not plan_path.exists():
        print(f"[ERROR] plan file not found: {plan_path}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build flags: by default pass "-v" only; optionally add "-e" with --advice.
    flags = []
    if not args.no_verbose:   # <-- fixed: argparse stores '--no-verbose' as 'no_verbose'
        flags.append("-v")
    if args.advice:
        flags.append("-e")
    flags.extend(["-t", str(args.tolerance)])

    with plan_path.open("r", encoding="utf-8") as f:
        plan_text = f.read()

    res = run_val(
        str(domain_pddl),
        str(problem_pddl),
        plan_text,
        val_path=args.val_path,
        flags=tuple(flags),
    )

    # Write artifacts
    stdout_path = out_dir / "val_stdout.txt"
    stderr_path = out_dir / "val_stderr.txt"
    trace_path  = out_dir / "val_trace.json"
    stdout_path.write_text(res.stdout or "", encoding="utf-8")
    stderr_path.write_text(res.stderr or "", encoding="utf-8")
    trace_path.write_text(
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
                for st in res.steps
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Console summary
    print("\n=== VAL Recheck — Existing Plan ===")
    print(f"Domain:         {domain_pddl}")
    print(f"Problem:        {problem_pddl}")
    print(f"Plan:           {plan_path}")
    print(f"Output dir:     {out_dir}")
    print(f"OK:             {res.ok}")
    print(f"Failure reason: {res.failure_reason or '—'}")
    print(f"Value (cost):   {res.value}")
    print(f"Attempts:       {getattr(res, 'attempts', 1)}")
    warn = getattr(res, "warning", None)
    if warn:
        print(f"Warning:        {warn}")
    print(f"stdout:         {stdout_path}")
    print(f"stderr:         {stderr_path}")
    print(f"trace:          {trace_path}")

    if args.json:
        record = {
            "domain": str(domain_pddl),
            "problem": str(problem_pddl),
            "plan": str(plan_path),
            "out_dir": str(out_dir),
            "ok": res.ok,
            "failure_reason": res.failure_reason,
            "value": res.value,
            "attempts": getattr(res, "attempts", 1),
            "warning": getattr(res, "warning", None),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "trace_path": str(trace_path),
        }
        print(json.dumps(record, ensure_ascii=False))

    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
