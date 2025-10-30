# green_agent/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import re
from .val_wrapper import run_val, TraceStep

# Header lines in "Plan Repair Advice":
# "(move-small ...) has an unsatisfied precondition at time 2"
_ADVICE_HDR_RE = re.compile(
    r"\(([^)]+)\)\s+has an unsatisfied precondition at time\s+(\d+)",
    re.IGNORECASE
)

# Advice lines:
# "Set (isoccupied loc3_6) to false"  OR  "Set (adjacent …) to true"
#  -> capture atom + desired truth value
_ADVICE_SET_RE = re.compile(
    r"Set\s*\(([^)]+)\)\s*to\s*(true|false)",
    re.IGNORECASE
)

@dataclass
class PlanMetrics:
    valid: bool
    length: int
    cost_value: Optional[float]
    first_failure_at: Optional[int]
    unsat_count: int
    redundant_indices: Optional[List[int]]
    failure_reason: Optional[str]

    first_failed_action: Optional[str]
    first_failure_reason: Optional[str]     # short atom-only summary (e.g., "isoccupied loc3_6")
    first_failure_detail: Optional[str]     # human-readable full sentence

    # Advice-derived signal
    advice_count: int
    advice_top_predicates: List[Tuple[str, int]]

    # raw logs
    val_stdout: str
    val_stderr: str
    # optional trace
    steps: List[TraceStep]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "length": self.length,
            "cost_value": self.cost_value,
            "first_failure_at": self.first_failure_at,
            "unsat_count": self.unsat_count,
            "redundant_indices": self.redundant_indices,
            "failure_reason": self.failure_reason,
            "first_failed_action": self.first_failed_action,
            "first_failure_reason": self.first_failure_reason,
            "first_failure_detail": self.first_failure_detail,
            "advice_count": self.advice_count,
            "advice_top_predicates": self.advice_top_predicates,
        }

def _parse_advice_by_time(stdout: str) -> Dict[int, List[Tuple[str, bool]]]:
    """
    Parse VAL's 'Plan Repair Advice' into: time -> list of (atom, desired_value)
    desired_value is True if advice says 'to true', else False.
    """
    lines = (stdout or "").splitlines()
    advice: Dict[int, List[Tuple[str, bool]]] = {}

    i = 0
    n = len(lines)
    while i < n:
        hdr = _ADVICE_HDR_RE.search(lines[i])
        if not hdr:
            i += 1
            continue

        try:
            t = int(hdr.group(2))
        except ValueError:
            t = None

        j = i + 1
        pairs: List[Tuple[str, bool]] = []
        while j < n and not _ADVICE_HDR_RE.search(lines[j]):
            mset = _ADVICE_SET_RE.search(lines[j])
            if mset:
                atom = mset.group(1).strip()
                desired_raw = mset.group(2).strip().lower()
                desired = (desired_raw == "true")
                pairs.append((atom, desired))
            j += 1

        if t is not None and pairs:
            advice.setdefault(t, []).extend(pairs)

        i = j

    return advice

def compute_metrics(
    *,
    domain: str,
    problem: str,
    plan_text: str,
    val_path: Optional[str] = None,
    flags: Tuple[str, ...] = ("-v","-e"),
    check_redundancy: bool = False,
) -> PlanMetrics:
    base = run_val(domain, problem, plan_text, val_path=val_path, flags=flags)

    length = len([ln for ln in plan_text.splitlines() if ln.strip().startswith("(")])

    # first failure
    first_fail = None
    first_failed_action = None
    for st in base.steps:
        if st.failed:
            first_fail = st.time
            first_failed_action = st.action
            break

    # quick redundancy probe (remove-one validation)+
    redundant = None
    if check_redundancy:
        redundant: list[int] = []
        if length > 0:
            lines = [ln for ln in plan_text.splitlines() if ln.strip()]
            for i in range(len(lines)):
                if not lines[i].strip().startswith("("):
                    continue
                variant = "\n".join(lines[j] for j in range(len(lines)) if j != i) + "\n"
                res = run_val(domain, problem, variant, val_path=val_path, flags=flags)
                if res.ok:
                    redundant.append(i + 1)  # 1-based

    # Advice parsing
    advice_by_time = _parse_advice_by_time(base.stdout or "")

    # Flatten for counts/top predicates (use just the atom names)
    all_advice_atoms = [atom for pairs in advice_by_time.values() for (atom, _desired) in pairs]
    advice_count = len(all_advice_atoms)
    pred_counts = Counter(
        (a.strip()[1:-1] if a.strip().startswith("(") and a.strip().endswith(")") else a).split()[0].lower()
        for a in all_advice_atoms
        if isinstance(a, str) and a.strip()
    )
    advice_top_predicates = pred_counts.most_common(8)

    # First failure reason (short + detailed)
    first_failure_reason = None
    first_failure_detail = None
    if first_fail is not None and advice_by_time.get(first_fail):
        # short reason: first atom only
        first_atom, first_desired = advice_by_time[first_fail][0]
        first_failure_reason = first_atom

        # Build human-readable detail. We *infer* current value as the opposite of desired.
        # (If VAL says "Set X to false", we assume X was true.)
        def _fmt_pair(pair: Tuple[str, bool]) -> str:
            atom, desired = pair
            desired_str = "true" if desired else "false"
            assumed_was = "false" if desired else "true"
            return f"{atom} = {desired_str} (but was {assumed_was})"

        pairs = advice_by_time[first_fail]
        if len(pairs) == 1:
            # Single predicate missing
            detail_bits = _fmt_pair(pairs[0])
        else:
            # Summarize first two, then "+N more"
            shown = ", ".join(_fmt_pair(p) for p in pairs[:2])
            more = len(pairs) - 2
            detail_bits = f"{shown}" + (f", +{more} more" if more > 0 else "")

        action_txt = first_failed_action or "unknown action"
        first_failure_detail = (
            f"Unsatisfied preconditions at step {first_fail} for {action_txt}: {detail_bits}."
        )

    return PlanMetrics(
        valid=base.ok,
        length=length,
        cost_value=base.value,
        first_failure_at=first_fail,
        unsat_count=len(base.unsatisfied),
        redundant_indices=redundant,
        failure_reason=base.failure_reason,
        first_failed_action=first_failed_action,
        first_failure_reason=first_failure_reason,
        first_failure_detail=first_failure_detail,
        advice_count=advice_count,
        advice_top_predicates=advice_top_predicates,
        val_stdout=base.stdout,
        val_stderr=base.stderr,
        steps=base.steps,
    )
