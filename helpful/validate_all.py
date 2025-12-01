#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from green_agent.val_wrapper import guess_val_binary

load_dotenv()


def run_validate(
    val_bin: str, domain: str, problem: str, plan_text: str, flags: tuple[str, ...]
) -> tuple[int, str, str]:
    """Invoke VAL and return (returncode, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".plan", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(plan_text)
        tf.flush()
        plan_path = tf.name
    try:
        cmd = [val_bin, *flags, domain, problem, plan_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        try:
            os.unlink(plan_path)
        except Exception:
            pass


# Heuristics for parsing output (VAL variants differ a bit).
_PARSE_ERROR_PATTERNS = [
    r"\bparse error\b",
    r"\bparsing error\b",
    r"\bsyntax error\b",
    r"\bunexpected token\b",
    r"\berrors? detected\b",
    r"\bLexical error\b",
    r"\bFailed to open\b",
    r"\bFile not found\b",
]
_PLAN_VALID_PATTERNS = [r"\bPlan valid\b", r"\bSuccessful plans\b"]
_PLAN_INVALID_PATTERNS = [
    r"\bPlan failed\b",
    r"\bInvalid plan\b",
    r"\bGoal not satisfied\b",
]


def parsed_ok(stdout: str, stderr: str) -> bool:
    blob = f"{stdout}\n{stderr}".lower()
    return not any(re.search(p, blob) for p in _PARSE_ERROR_PATTERNS)


def plan_valid(stdout: str, stderr: str) -> bool | None:
    """Return True/False if we can tell; None if unknown."""
    blob = f"{stdout}\n{stderr}"
    if any(re.search(p, blob) for p in _PLAN_VALID_PATTERNS):
        return True
    if any(re.search(p, blob) for p in _PLAN_INVALID_PATTERNS):
        return False
    return None


def find_problems(domain_dir: Path) -> list[Path]:
    """Return all *.pddl problems sorted by numeric suffix if present."""
    probs_dir = domain_dir / "problems_pddl"
    if not probs_dir.exists():
        # Fallback: any problem*.pddl in the domain dir
        candidates = sorted(domain_dir.glob("problem*.pddl"))
    else:
        candidates = sorted(probs_dir.glob("*.pddl"))

    def sort_key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 1_000_000, p.name)

    return sorted(candidates, key=sort_key)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Validate (parse-check) a PDDL domain and all its problems using VAL."
        )
    )
    ap.add_argument(
        "--domain-dir",
        required=True,
        help="Folder containing domain.pddl and problems_pddl/",
    )
    ap.add_argument(
        "--val-path",
        default=None,
        help="Path to Validate binary (otherwise uses VAL_PATH or PATH)",
    )
    ap.add_argument(
        "--flags",
        nargs="*",
        default=["-v", "-e"],
        help="Extra flags for VAL (default: -v -e)",
    )
    ap.add_argument(
        "--json-out", default=None, help="Optional path to write a JSON report"
    )
    args = ap.parse_args()

    domain_dir = Path(args.domain_dir)
    domain_pddl = domain_dir / "domain.pddl"
    if not domain_pddl.exists():
        print(f"[ERROR] domain.pddl not found at: {domain_pddl}", file=sys.stderr)
        sys.exit(2)

    problems = find_problems(domain_dir)
    if not problems:
        print(
            f"[WARN] No problems found under {domain_dir}/problems_pddl/*.pddl",
            file=sys.stderr,
        )

    val_bin = guess_val_binary(args.val_path)
    if not val_bin:
        print(
            "[ERROR] Validate binary not found. "
            "Install VAL or set --val-path or VAL_PATH.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"VAL: {val_bin}")
    print(f"Domain: {domain_pddl}")
    print(f"Problems: {len(problems)} found")

    dummy_plan = ";; empty plan for parse-check only\n"

    rows: list[dict[str, Any]] = []
    all_parsed = True

    for prob in problems:
        rc, out, err = run_validate(
            val_bin, str(domain_pddl), str(prob), dummy_plan, tuple(args.flags)
        )
        ok = parsed_ok(out, err)
        pv = plan_valid(out, err)
        rows.append(
            {
                "problem": str(prob),
                "parsed": ok,
                "plan_valid": pv,  # will usually be False (no plan)
                "val_returncode": rc,
            }
        )
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {prob}")
        if not ok:
            all_parsed = False

    # Summary
    parsed_cnt = sum(1 for r in rows if r["parsed"])
    print(f"\nParsed OK: {parsed_cnt}/{len(rows)}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "domain": str(domain_pddl),
                    "val": val_bin,
                    "results": rows,
                    "summary": {"parsed_ok": parsed_cnt, "total": len(rows)},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Wrote JSON report to {args.json_out}")

    # Exit 0 only if every (domain, problem) parsed
    sys.exit(0 if all_parsed else 1)


if __name__ == "__main__":
    main()
