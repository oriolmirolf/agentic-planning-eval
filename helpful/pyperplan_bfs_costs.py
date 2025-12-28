#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# pyperplan-style lines
RE_PLAN_LEN = re.compile(r"Plan length:\s*(\d+)", re.IGNORECASE)
RE_NODES = re.compile(r"(\d+)\s+Nodes expanded", re.IGNORECASE)
RE_TIME = re.compile(r"Search time:\s*([0-9.]+)", re.IGNORECASE)

# validation (pyperplan + VAL)
RE_PLAN_OK = re.compile(r"\bPlan (correct|valid)\b", re.IGNORECASE)
RE_PLAN_BAD = re.compile(r"\bPlan (invalid|failed)\b", re.IGNORECASE)
RE_NO_VALIDATE = re.compile(r"validate could not be found on the PATH", re.IGNORECASE)

# FF / Metric-FF plan lines commonly look like:
#   step    0: PICKUP A
#   0: PICKUP A
_RE_FF_STEP0 = re.compile(r"^\s*step\s+\d+\s*:\s*(.+)$", re.IGNORECASE)
_RE_FF_STEPN = re.compile(r"^\s*\d+\s*:\s*(.+)$")


@dataclass
class Row:
    idx: int
    plan_len: int | None
    plan_ok: bool | None
    validator_available: bool | None
    nodes: int | None
    search_time_s: float | None
    rc: int
    note: str


def _num_from_stem(p: Path) -> int:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else 0


def _pyperplan_supported_searches(pyperplan_cmd: list[str]) -> set[str]:
    """
    Try to parse the supported -s/--search choices from 'pyperplan --help'.
    Returns empty set if parsing fails (we'll be permissive then).
    """
    try:
        proc = subprocess.run(
            [*pyperplan_cmd, "--help"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()

    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # In your output, it's like: [-s {astar,wastar,gbf,bfs,ehs,ids,sat}]
    m = re.search(r"-s\s+\{([^}]+)\}", help_text)
    if not m:
        return set()
    return {s.strip() for s in m.group(1).split(",") if s.strip()}


def run_pyperplan(
    *,
    domain_pddl: Path,
    problem_pddl: Path,
    pyperplan_cmd: list[str],
    search: str,
    heuristic: str | None = None,
) -> tuple[int, str]:
    # Example cmd: uv run pyperplan -s bfs -H hff domain.pddl problem.pddl
    cmd: list[str] = [*pyperplan_cmd, "-s", search]
    if heuristic:
        cmd += ["-H", heuristic]
    cmd += [str(domain_pddl), str(problem_pddl)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out


def _extract_actions_from_ff_output(stdout: str, stderr: str = "") -> list[str]:
    """
    Extract actions from FF/Metric-FF output.
    Returns list like ["(pickup a)", "(stack a b)", ...]
    """
    text = (stdout or "") + "\n" + (stderr or "")
    actions: list[str] = []

    for ln in text.splitlines():
        m = _RE_FF_STEP0.match(ln) or _RE_FF_STEPN.match(ln)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue

        tokens = body.split()
        op = tokens[0].lower()
        args = " ".join(t.lower() for t in tokens[1:])
        actions.append(f"({op} {args})" if args else f"({op})")

    return actions


def run_ff_planner(
    *,
    domain_pddl: Path,
    problem_pddl: Path,
    ff_cmd: list[str],
    timeout_s: float | None = None,
) -> tuple[int, str, list[str]]:
    """
    Run an external FF-family planner (ideally Metric-FF for numeric fluents).
    Returns (rc, combined_output, actions).
    """
    # Typical FF usage: ff -o domain.pddl -f problem.pddl
    cmd = [*ff_cmd, "-o", str(domain_pddl), "-f", str(problem_pddl)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        actions = _extract_actions_from_ff_output(proc.stdout or "", proc.stderr or "")
        return proc.returncode, out, actions
    except subprocess.TimeoutExpired as e:
        out = (
            (e.stdout or "") + "\n" + (e.stderr or "") + f"\nTimeout after {timeout_s}s"
        )
        return -1, out, []


def _find_validate() -> str | None:
    return shutil.which("validate")


def _validate_plan(
    *,
    validate_bin: str,
    domain_pddl: Path,
    problem_pddl: Path,
    plan_text: str,
) -> tuple[bool | None, str]:
    """
    Validate a plan using VAL's 'validate' binary.
    Returns (plan_ok, validate_output).
    plan_ok is None if we couldn't determine.
    """
    if not plan_text.strip():
        return None, "no plan text to validate"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".plan", delete=False) as f:
        f.write(plan_text if plan_text.endswith("\n") else plan_text + "\n")
        plan_path = f.name

    # Try common invocation: validate -v domain problem plan
    cmd1 = [validate_bin, "-v", str(domain_pddl), str(problem_pddl), plan_path]
    proc = subprocess.run(cmd1, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Some validate builds may not like -v; retry without it if needed.
    if proc.returncode != 0 and (
        "unknown option" in out.lower() or "unrecognized option" in out.lower()
    ):
        cmd2 = [validate_bin, str(domain_pddl), str(problem_pddl), plan_path]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        out = (proc2.stdout or "") + "\n" + (proc2.stderr or "")

    if RE_PLAN_OK.search(out):
        return True, out
    if RE_PLAN_BAD.search(out):
        return False, out

    # If VAL output uses different wording, fall back to return code guess.
    if proc.returncode == 0:
        return None, out
    return None, out


def parse_output_pyperplan(idx: int, rc: int, out: str) -> Row:
    mlen = RE_PLAN_LEN.search(out)
    plan_len = int(mlen.group(1)) if mlen else None

    validator_available = None
    if RE_NO_VALIDATE.search(out):
        validator_available = False
    elif RE_PLAN_OK.search(out) or RE_PLAN_BAD.search(out):
        validator_available = True

    plan_ok = None
    if RE_PLAN_OK.search(out):
        plan_ok = True
    elif RE_PLAN_BAD.search(out) or ("not correct" in out.lower()):
        plan_ok = False

    mn = RE_NODES.search(out)
    nodes = int(mn.group(1)) if mn else None

    mt = RE_TIME.search(out)
    search_time_s = float(mt.group(1)) if mt else None

    note = ""
    if plan_len is None:
        note = "no plan length parsed"
    elif validator_available is False:
        note = "validator not found (needs `validate` on PATH)"
    elif plan_ok is False:
        note = "plan invalid"
    elif plan_ok is True:
        note = "ok"
    else:
        note = "no validation message"

    return Row(
        idx=idx,
        plan_len=plan_len,
        plan_ok=plan_ok,
        validator_available=validator_available,
        nodes=nodes,
        search_time_s=search_time_s,
        rc=rc,
        note=note,
    )


def parse_output_ff(
    idx: int,
    rc: int,
    out: str,
    actions: list[str],
    validate_bin: str | None,
    domain_pddl: Path,
    problem_pddl: Path,
) -> Row:
    plan_len = len(actions) if actions else None

    validator_available = True if validate_bin else False
    plan_ok: bool | None = None
    note = ""

    if plan_len is None:
        note = "no plan parsed from FF output"
    else:
        if validate_bin:
            plan_text = "\n".join(actions) + "\n"
            plan_ok, _val_out = _validate_plan(
                validate_bin=validate_bin,
                domain_pddl=domain_pddl,
                problem_pddl=problem_pddl,
                plan_text=plan_text,
            )
            if plan_ok is True:
                note = "ok"
            elif plan_ok is False:
                note = "plan invalid"
            else:
                note = "validation inconclusive"
        else:
            note = "validator not found (needs `validate` on PATH)"

    # Nodes/time are not consistently available in FF output; leave blank.
    return Row(
        idx=idx,
        plan_len=plan_len,
        plan_ok=plan_ok,
        validator_available=validator_available,
        nodes=None,
        search_time_s=None,
        rc=rc,
        note=note,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run pyperplan on all problems and parse plan length + validation.\n"
            "Special case: --search ff runs an external FF/Metric-FF binary "
            "(use this if you need numeric fluents/metrics)."
        )
    )
    ap.add_argument(
        "--domain",
        required=True,
        help="Domain folder under ./examples (e.g. blocks)",
    )
    ap.add_argument(
        "--search",
        default="bfs",
        help=(
            "Search algorithm. "
            "For pyperplan: astar,wastar,gbf,bfs,ehs,ids,sat (depends on your build). "
            "Special: ff = run external FF/Metric-FF (for numeric fluents)."
        ),
    )
    ap.add_argument(
        "--heuristic",
        default=None,
        help="pyperplan heuristic (-H). Ignored when --search ff.",
    )
    ap.add_argument(
        "--pyperplan-cmd",
        nargs="+",
        default=["uv", "run", "pyperplan"],
        help="Command prefix to run pyperplan (default: uv run pyperplan)",
    )
    ap.add_argument(
        "--ff-cmd",
        nargs="+",
        default=["ff"],
        help="Command to run FF/Metric-FF when --search ff (default: ff).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout per problem (seconds)",
    )
    ap.add_argument("--max", type=int, default=None, help="Limit number of problems")
    args = ap.parse_args()

    domain_dir = Path("examples") / args.domain
    domain_pddl = domain_dir / "domain.pddl"
    problems_dir = domain_dir / "problems_pddl"

    if not domain_pddl.exists():
        raise SystemExit(f"[ERROR] domain.pddl not found: {domain_pddl}")
    if not problems_dir.exists():
        raise SystemExit(f"[ERROR] problems_pddl not found: {problems_dir}")

    # If user requested a pyperplan search, sanity check it against --help.
    if args.search != "ff":
        supported = _pyperplan_supported_searches(args.pyperplan_cmd)
        if supported and args.search not in supported:
            raise SystemExit(
                f"[ERROR] --search {args.search!r} not supported by this pyperplan.\n"
                f"        Supported: {', '.join(sorted(supported))}"
            )

    # If user requested ff, make sure ff command exists (first token).
    if args.search == "ff":
        ff0 = args.ff_cmd[0]
        if not (Path(ff0).exists() or shutil.which(ff0)):
            raise SystemExit(
                "[ERROR] --search ff requested, but FF/Metric-FF binary not found.\n"
                "        Either put it on PATH (e.g. `ff`), or"
                " pass --ff-cmd /path/to/ff\n NOTE: For numeric fluents/metrics you"
                " typically need Metric-FF."
            )

    probs = sorted(problems_dir.glob("problem*.pddl"), key=_num_from_stem)
    if args.max:
        probs = probs[: args.max]

    validate_bin = _find_validate()

    table = Table(title=f"planner ({args.search}) — {args.domain}", show_lines=False)
    table.add_column("Prob", justify="right")
    table.add_column("Len", justify="right")
    table.add_column("Validated", justify="center")
    table.add_column("Plan OK", justify="center")
    table.add_column("Nodes", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("RC", justify="right")
    table.add_column("Note", overflow="fold")

    rows: list[Row] = []
    for prob in probs:
        idx = _num_from_stem(prob)

        if args.search == "ff":
            rc, out, actions = run_ff_planner(
                domain_pddl=domain_pddl,
                problem_pddl=prob,
                ff_cmd=args.ff_cmd,
                timeout_s=args.timeout,
            )
            row = parse_output_ff(
                idx=idx,
                rc=rc,
                out=out,
                actions=actions,
                validate_bin=validate_bin,
                domain_pddl=domain_pddl,
                problem_pddl=prob,
            )
        else:
            rc, out = run_pyperplan(
                domain_pddl=domain_pddl,
                problem_pddl=prob,
                pyperplan_cmd=args.pyperplan_cmd,
                search=args.search,
                heuristic=args.heuristic,
            )
            row = parse_output_pyperplan(idx, rc, out)

        rows.append(row)

        validated_txt = "—"
        if row.validator_available is True:
            validated_txt = "[green]YES[/green]"
        elif row.validator_available is False:
            validated_txt = "[red]NO[/red]"

        ok_txt = "—"
        if row.plan_ok is True:
            ok_txt = "[green]YES[/green]"
        elif row.plan_ok is False:
            ok_txt = "[red]NO[/red]"

        table.add_row(
            str(idx),
            str(row.plan_len) if row.plan_len is not None else "—",
            validated_txt,
            ok_txt,
            str(row.nodes) if row.nodes is not None else "—",
            f"{row.search_time_s:.2f}" if isinstance(row.search_time_s, float) else "—",
            str(row.rc),
            row.note,
        )

    console.print(table)

    # Summary
    n = len(rows)
    ok = sum(1 for r in rows if r.plan_ok is True)
    invalid = sum(1 for r in rows if r.plan_ok is False)
    noval = sum(1 for r in rows if r.validator_available is False)
    console.print(
        f"[bold]Summary:[/bold] problems={n} plan_ok={ok} "
        f"invalid={invalid} missing_validator={noval}"
    )

    # Important note
    if args.search == "ff":
        console.print(
            "\n[dim]Note: '--search ff' runs an external FF/Metric-FF binary.\n"
            "If you need numeric fluents/(:functions)/(:metric ...), you must use "
            "a build that supports them (typically Metric-FF). pyperplan itself does "
            "not support numeric fluents.[/dim]"
        )
    else:
        console.print(
            "\n[dim]Note: pyperplan BFS is optimal in number of actions. "
            "If your domain uses non-unit action costs / numeric metrics, "
            "this is not necessarily optimal cost.[/dim]"
        )


if __name__ == "__main__":
    main()
