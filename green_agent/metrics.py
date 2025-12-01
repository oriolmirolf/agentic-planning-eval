# green_agent/metrics.py
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .val_wrapper import TraceStep, run_val

# Header lines in "Plan Repair Advice":
# "(move-small ...) has an unsatisfied precondition at time 2"
_ADVICE_HDR_RE = re.compile(
    r"\(([^)]+)\)\s+has an unsatisfied precondition at time\s+(\d+)", re.IGNORECASE
)

# Advice lines:
# "Set (isoccupied loc3_6) to false"  OR  "Set (adjacent …) to true"
#  -> capture atom + desired truth value
_ADVICE_SET_RE = re.compile(r"Set\s*\(([^)]+)\)\s*to\s*(true|false)", re.IGNORECASE)


@dataclass(slots=True)
class PlanMetrics:
    valid: bool
    length: int
    cost_value: float | None
    first_failure_at: int | None
    unsat_count: int
    redundant_indices: list[int] | None
    failure_reason: str | None

    first_failed_action: str | None
    first_failure_reason: (
        str | None
    )  # short atom-only summary (e.g., "isoccupied loc3_6")
    first_failure_detail: str | None  # human-readable full sentence

    # Advice-derived signal
    advice_count: int
    advice_top_predicates: list[tuple[str, int]]

    # raw logs
    val_stdout: str
    val_stderr: str
    # optional trace
    steps: list[TraceStep]

    # NEW: VAL retry diagnostics
    val_attempts: int
    val_warning: str | None

    def as_dict(self) -> dict[str, Any]:
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
            "val_attempts": self.val_attempts,
            "val_warning": self.val_warning,
        }


def _parse_advice_by_time(stdout: str) -> dict[int, list[tuple[str, bool]]]:
    """
    Parse VAL's 'Plan Repair Advice' into: time -> list of (atom, desired_value)
    desired_value is True if advice says 'to true', else False.
    """
    lines = (stdout or "").splitlines()
    advice: dict[int, list[tuple[str, bool]]] = {}

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
        pairs: list[tuple[str, bool]] = []
        while j < n and not _ADVICE_HDR_RE.search(lines[j]):
            mset = _ADVICE_SET_RE.search(lines[j])
            if mset:
                atom = mset.group(1).strip()
                desired_raw = mset.group(2).strip().lower()
                desired = desired_raw == "true"
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
    val_path: str | None = None,
    flags: tuple[str, ...] = ("-v"),
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

    # quick redundancy check (remove-one validation)
    redundant = None
    if check_redundancy and length > 0:
        redundant = []
        lines = [ln for ln in plan_text.splitlines() if ln.strip()]
        # consider only action lines; map from action index -> original line index
        action_line_idxs = [
            idx for idx, ln in enumerate(lines) if ln.strip().startswith("(")
        ]
        for k, remove_idx in enumerate(action_line_idxs):
            # build variant skipping exactly this action line
            variant = (
                "\n".join(lines[j] for j in range(len(lines)) if j != remove_idx) + "\n"
            )
            res = run_val(domain, problem, variant, val_path=val_path, flags=flags)
            if res.ok:
                redundant.append(k + 1)  # 1-based among ACTIONS

    # Advice parsing
    advice_by_time = _parse_advice_by_time(base.stdout or "")

    # Flatten for counts/top predicates (use just the atom names)
    all_advice_atoms = [
        atom for pairs in advice_by_time.values() for (atom, _desired) in pairs
    ]
    advice_count = len(all_advice_atoms)
    pred_counts = Counter(
        (
            a.strip()[1:-1]
            if a.strip().startswith("(") and a.strip().endswith(")")
            else a
        )
        .split()[0]
        .lower()
        for a in all_advice_atoms
        if isinstance(a, str) and a.strip()
    )
    advice_top_predicates = pred_counts.most_common(8)

    # First failure reason (short + detailed)
    first_failure_reason = None
    first_failure_detail = None
    if first_fail is not None and advice_by_time.get(first_fail):
        # short reason: first atom only
        first_atom, _ = advice_by_time[first_fail][0]
        first_failure_reason = first_atom

        # Build human-readable detail. We infer current value as the opp. of desired.
        # (If VAL says "Set X to false", we assume X was true.)
        def _fmt_pair(pair: tuple[str, bool]) -> str:
            atom, desired = pair
            desired_str = "true" if desired else "false"
            assumed_was = "false" if desired else "true"
            return f"{atom} = {desired_str} (but was {assumed_was})"

        first_failure_detail = (
            f"Unsatisfied preconditions at step {first_fail} "
            "for {action_txt}: {detail_bits}."
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
        val_attempts=base.attempts,
        val_warning=base.warning,
    )
