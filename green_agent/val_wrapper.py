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

_UNSAT_RE = re.compile(r"Unsatisfied precondition.*?:\s*(.*)", re.IGNORECASE)
_FAILED_AT_STEP_RE = re.compile(r"action\s*(\d+)", re.IGNORECASE)
_VALUE_RE = re.compile(r"Value\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"Successful plans|Plan valid", re.IGNORECASE)
_FAILED_RE = re.compile(r"Failed plans|Plan failed|Invalid plan|Goal not satisfied", re.IGNORECASE)

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
        value = None
        mval = _VALUE_RE.search(out)
        if mval:
            try: value = float(mval.group(1))
            except ValueError: pass
        unsatisfied: list[Unsat] = []
        for line in out.splitlines():
            mu = _UNSAT_RE.search(line)
            if mu:
                detail = mu.group(1).strip()
                mstep = _FAILED_AT_STEP_RE.search(line)
                idx = int(mstep.group(1)) if mstep else None
                unsatisfied.append(Unsat(at_action_index=idx, detail=detail))
        return ValResult(ok=ok, stdout=out, stderr=err, value=value, unsatisfied=unsatisfied)
    finally:
        try: os.unlink(plan_path)
        except Exception: pass
