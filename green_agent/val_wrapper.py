# green_agent/val_wrapper.py
from __future__ import annotations
import os, shutil, subprocess, tempfile, re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
_FAILED_AT_STEP_RE = re.compile(r"\baction\s+(\d+)\b", re.IGNORECASE)  # legacy format
_TIME_RE = re.compile(r"Checking next happening\s*\(time\s*(\d+)\)", re.IGNORECASE)

_VALUE_RE = re.compile(r"Value\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"Successful plans|Plan valid", re.IGNORECASE)
_FAILED_RE = re.compile(r"Failed plans|Plan failed|Invalid plan|Goal not satisfied", re.IGNORECASE)

# specific failure reason
_GOAL_NOT_SAT_RE = re.compile(r"Goal not satisfied", re.IGNORECASE)

_ACTION_LINE_RE = re.compile(r"^\s*\(([^()]+)\)\s*$")  # e.g. "(move-small red loc2_3 loc3_3 loc4_3)"

@dataclass
class Unsat:
    at_action_index: Optional[int]
    detail: str

@dataclass
class ValResult:
    ok: bool
    stdout: str
    stderr: str
    value: Optional[float] = None
    unsatisfied: List[Unsat] = field(default_factory=list)
    failure_reason: Optional[str] = None  # "goal_not_satisfied" | "precondition_unsatisfied" | "unknown_failure"


def run_val(domain: str, problem: str, plan_text: str, *, val_path: Optional[str] = None, flags: Tuple[str, ...] = ("-v","-e")) -> ValResult:
    bin_path = _guess_val_binary(val_path)
    if not bin_path:
        raise RuntimeError("VAL binary not found. Install or set VAL_PATH.")
    with tempfile.NamedTemporaryFile("w", suffix=".plan", delete=False) as tf:
        tf.write(plan_text); tf.flush(); plan_path = tf.name
    try:
        cmd = [bin_path, *flags, domain, problem, plan_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.stdout, proc.stderr
        ok = bool(_SUCCESS_RE.search(out)) and not _FAILED_RE.search(out)

        # Extract numeric plan value (if any)
        value = None
        mval = _VALUE_RE.search(out)
        if mval:
            try:
                value = float(mval.group(1))
            except ValueError:
                pass

        # --- Robustly extract unsatisfied preconditions + step index ---
        lines = out.splitlines()
        current_step: Optional[int] = None
        unsatisfied: list[Unsat] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Always track "Checking next happening (time X)" as the step index for instantaneous plans
            mtime = _TIME_RE.search(line)
            if mtime:
                try:
                    current_step = int(mtime.group(1))
                except ValueError:
                    current_step = None

            # Legacy pattern sometimes prints "... action N" on the same line
            if _UNSAT_RE.search(line):
                idx: Optional[int] = None
                # try legacy "action N"
                mact = _FAILED_AT_STEP_RE.search(line)
                if mact:
                    try:
                        idx = int(mact.group(1))
                    except ValueError:
                        idx = None
                # if not found, use the current_step tracked from "time X"
                if idx is None:
                    idx = current_step

                # On newer builds, the action itself is on the next non-empty line
                detail = ""
                if _UNSAT_LINE_RE.search(line):
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        mactline = _ACTION_LINE_RE.match(lines[j])
                        if mactline:
                            detail = "(" + mactline.group(1).strip() + ")"
                        else:
                            # fallback: capture the whole next line
                            detail = lines[j].strip()
                else:
                    # fallback: keep remainder of the line
                    detail = line.strip()

                unsatisfied.append(Unsat(at_action_index=idx, detail=detail))

            i += 1
            
        
        # failure reason heuristics
        failure_reason: Optional[str] = None
        if not ok:
            if unsatisfied:
                failure_reason = "precondition_unsatisfied"
            elif _GOAL_NOT_SAT_RE.search(out):
                failure_reason = "goal_not_satisfied"
            else:
                failure_reason = "unknown_failure"

        return ValResult(ok=ok, stdout=out, stderr=err, value=value, unsatisfied=unsatisfied, failure_reason=failure_reason)

    finally:
        try: os.unlink(plan_path)
        except Exception: pass
