# green_agent/val_wrapper.py
from __future__ import annotations
import os, shutil, subprocess, tempfile, re, json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

def _guess_val_binary(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    for name in ("Validate", "validate", "Validate.exe", "validate.exe"):
        path = shutil.which(name) or (explicit if explicit else None)
        if path and os.path.exists(path):
            return path
    envp = os.getenv("VAL_PATH")
    if envp and os.path.exists(envp):
        return envp
    return shutil.which("Validate") or shutil.which("validate")

# --- Patterns seen across VAL variants ---
_UNSAT_RE = re.compile(r"Unsatisfied precondition", re.IGNORECASE)
_UNSAT_LINE_RE = re.compile(r"Plan failed because of unsatisfied precondition", re.IGNORECASE)
_FAILED_AT_STEP_RE = re.compile(r"\baction\s+(\d+)\b", re.IGNORECASE)
_TIME_RE = re.compile(r"Checking next happening\s*\(time\s*(\d+)\)", re.IGNORECASE)

_VALUE_RE = re.compile(r"Value\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"Successful plans|Plan valid", re.IGNORECASE)
_FAILED_RE = re.compile(r"Failed plans|Plan failed|Invalid plan|Goal not satisfied", re.IGNORECASE)
_GOAL_NOT_SAT_RE = re.compile(r"Goal not satisfied", re.IGNORECASE)

_ACTION_LINE_RE = re.compile(r"^\s*\(([^()]+)\)\s*$")  # "(move-small ...)"
_PLAN_INDEX_RE = re.compile(r"^\s*(\d+)\s*:\s*$")      # "1:" then action on next line

_ADDING_RE = re.compile(r"^\s*Adding\s*\((.+)\)\s*$", re.IGNORECASE)
_DELETING_RE = re.compile(r"^\s*Deleting\s*\((.+)\)\s*$", re.IGNORECASE)

@dataclass
class Unsat:
    at_action_index: Optional[int]
    detail: str

@dataclass
class TraceStep:
    time: int
    action: Optional[str] = None
    adds: List[str] = field(default_factory=list)
    deletes: List[str] = field(default_factory=list)
    failed: bool = False
    failure_detail: Optional[str] = None

@dataclass
class ValResult:
    ok: bool
    stdout: str
    stderr: str
    value: Optional[float] = None
    unsatisfied: List[Unsat] = field(default_factory=list)
    failure_reason: Optional[str] = None  # "goal_not_satisfied" | "precondition_unsatisfied" | "unknown_failure"
    plan_size: Optional[int] = None
    plan_actions: Dict[int, str] = field(default_factory=dict)  # index -> "(action ...)"
    steps: List[TraceStep] = field(default_factory=list)         # per happening
    last_executed_step: Optional[int] = None

def _parse_plan_listing(lines: List[str]) -> Dict[int, str]:
    """Parse the 'Plan to validate' listing into {index: action}."""
    plan: Dict[int, str] = {}
    i = 0
    while i < len(lines):
        m = _PLAN_INDEX_RE.match(lines[i])
        if m:
            idx = int(m.group(1))
            # find next non-empty action line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                mm = _ACTION_LINE_RE.match(lines[j])
                if mm:
                    plan[idx] = "(" + mm.group(1).strip() + ")"
                    i = j
        i += 1
    return plan


def run_val(
    domain: str,
    problem: str,
    plan_text: str,
    *,
    val_path: Optional[str] = None,
    flags: Tuple[str, ...] = ("-v","-e")
) -> ValResult:
    bin_path = _guess_val_binary(val_path)
    if not bin_path:
        raise RuntimeError("VAL binary not found. Install or set VAL_PATH.")

    with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as tf:
        tf.write(plan_text); tf.flush(); plan_path = tf.name

    try:
        cmd = [bin_path, *flags, domain, problem, plan_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.stdout, proc.stderr
        print(out)

        ok = bool(_SUCCESS_RE.search(out)) and not _FAILED_RE.search(out)

        # numeric value (if any)
        value = None
        mval = _VALUE_RE.search(out)
        if mval:
            try: value = float(mval.group(1))
            except ValueError: pass

        lines = out.splitlines()
        plan_actions = _parse_plan_listing(lines)
        plan_size = max(plan_actions.keys(), default=0) or None

        # trace parsing
        steps: List[TraceStep] = []
        current: Optional[TraceStep] = None
        current_step_idx: Optional[int] = None

        unsatisfied: list[Unsat] = []
        last_executed_step: Optional[int] = None

        i = 0
        while i < len(lines):
            line = lines[i]

            mtime = _TIME_RE.search(line)
            if mtime:
                # close previous step
                if current:
                    steps.append(current)
                current_step_idx = int(mtime.group(1))
                current = TraceStep(
                    time=current_step_idx,
                    action=plan_actions.get(current_step_idx)
                )
                # If there was no failure note yet, we consider this step "executed" at least up to check.
                last_executed_step = current_step_idx

            elif _UNSAT_RE.search(line) or _UNSAT_LINE_RE.search(line):
                # Capture failure at this step
                idx = None
                mact = _FAILED_AT_STEP_RE.search(line)
                if mact:
                    try: idx = int(mact.group(1))
                    except ValueError: idx = None
                # If not present, fall back to current step index
                if idx is None:
                    idx = current_step_idx

                # Next non-empty line may have the action
                detail_action = ""
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    mm = _ACTION_LINE_RE.match(lines[j])
                    if mm:
                        detail_action = "(" + mm.group(1).strip() + ")"
                    else:
                        detail_action = lines[j].strip()

                unsatisfied.append(Unsat(at_action_index=idx, detail=detail_action))
                if current:
                    current.failed = True
                    current.failure_detail = "unsatisfied_precondition"
                    if not current.action and detail_action:
                        current.action = detail_action

            else:
                madd = _ADDING_RE.match(line)
                if madd and current:
                    current.adds.append("(" + madd.group(1).strip() + ")")
                mdel = _DELETING_RE.match(line)
                if mdel and current:
                    current.deletes.append("(" + mdel.group(1).strip() + ")")

            i += 1

        if current:
            steps.append(current)

        # failure reason
        failure_reason: Optional[str] = None
        if not ok:
            if unsatisfied:
                failure_reason = "precondition_unsatisfied"
            elif _GOAL_NOT_SAT_RE.search(out):
                failure_reason = "goal_not_satisfied"
            else:
                failure_reason = "unknown_failure"

        return ValResult(
            ok=ok,
            stdout=out,
            stderr=err,
            value=value,
            unsatisfied=unsatisfied,
            failure_reason=failure_reason,
            plan_size=plan_size,
            plan_actions=plan_actions,
            steps=steps,
            last_executed_step=last_executed_step,
        )

    finally:
        try: os.unlink(plan_path)
        except Exception: pass
