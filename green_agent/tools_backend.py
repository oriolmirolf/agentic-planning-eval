# /agentic-planning-eval/green_agent/tools_backend.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import compute_metrics
from .val_wrapper import run_val

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaskOverview:
    description: str
    initial_state: str
    goal: str


@dataclass(slots=True)
class ObjectSpec:
    name: str
    kind: str
    summary: str
    attributes: dict[str, Any]


@dataclass(slots=True)
class ParameterSpec:
    name: str
    kind: str


@dataclass(slots=True)
class ActionSchema:
    name: str
    signature: str
    parameters: list[ParameterSpec]
    preconditions: list[str]
    effects: list[str]
    cost: float


@dataclass(slots=True)
class ProblemSpec:
    domain: str
    index: int
    overview: TaskOverview
    objects: list[ObjectSpec]
    actions: list[ActionSchema]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prompts_path(domain: str) -> Path:
    return Path("examples") / domain / "prompts.json"


def _load_prompts(domain: str) -> dict[str, Any]:
    path = _prompts_path(domain)
    if not path.exists():
        raise FileNotFoundError(
            f"prompts.json not found for domain '{domain}' at {path}"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"prompts.json for domain '{domain}' must be a JSON object.")
    return data


def _find_problem_entry(data: dict[str, Any], index: int) -> dict[str, Any]:
    """
    Look up a problem in prompts.json by index.
    Accepts IDs 'pNN' or plain 'NN'.
    """
    problems = data.get("problems") or []
    pid = f"p{int(index):02d}"
    for entry in problems:
        raw_id = str(entry.get("id", "")).strip()
        if raw_id == pid or raw_id == str(index):
            return entry
    raise KeyError(f"No problem with id '{pid}' (index {index}) in prompts.json.")


def _make_overview(
    domain: str, data: dict[str, Any], entry: dict[str, Any]
) -> TaskOverview:
    domain_prompt = (data.get("domain_prompt") or "").strip()
    overview = entry.get("overview") or {}

    desc = (
        overview.get("description")
        or "\n\n".join(
            p for p in (domain_prompt, (entry.get("prompt") or "").strip()) if p
        ).strip()
    )

    init = (overview.get("initial_state") or entry.get("initial_state") or "").strip()
    goal = (overview.get("goal") or entry.get("goal") or "").strip()

    return TaskOverview(description=desc, initial_state=init, goal=goal)


def _make_actions(data: dict[str, Any]) -> list[ActionSchema]:
    raw_actions = data.get("actions") or []
    actions: list[ActionSchema] = []
    for a in raw_actions:
        name = str(a.get("name") or "").strip()
        if not name:
            continue
        signature = str(a.get("signature") or name)
        params_raw = a.get("parameters") or []
        params = [
            ParameterSpec(
                name=str(p.get("name") or "").strip(),
                kind=str(p.get("kind") or "object").strip(),
            )
            for p in params_raw
            if p.get("name") is not None
        ]
        pre = [str(x).strip() for x in (a.get("preconditions") or []) if str(x).strip()]
        eff = [str(x).strip() for x in (a.get("effects") or []) if str(x).strip()]
        cost = float(a.get("cost", 1))

        actions.append(
            ActionSchema(
                name=name,
                signature=signature,
                parameters=params,
                preconditions=pre,
                effects=eff,
                cost=cost,
            )
        )
    return actions


def _make_objects(entry: dict[str, Any]) -> list[ObjectSpec]:
    objs_raw = entry.get("objects") or []
    objs: list[ObjectSpec] = []
    for o in objs_raw:
        name = str(o.get("name") or "").strip()
        if not name:
            continue
        kind = str(o.get("kind") or "object").strip()
        summary = str(o.get("summary") or "").strip()
        attrs = o.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {"raw": attrs}
        objs.append(ObjectSpec(name=name, kind=kind, summary=summary, attributes=attrs))
    return objs


def _resolve_pddl_paths(domain: str, index: int) -> tuple[str, str]:
    base = Path("examples") / domain
    domain_pddl = base / "domain.pddl"
    problems_dir = base / "problems_pddl"

    if not domain_pddl.exists():
        raise FileNotFoundError(
            f"domain.pddl not found for domain '{domain}' at {domain_pddl}"
        )
    problem_pddl = problems_dir / f"problem{int(index)}.pddl"
    if not problem_pddl.exists():
        raise FileNotFoundError(
            f"problem{index}.pddl not found for domain '{domain}' at {problem_pddl}"
        )
    return str(domain_pddl), str(problem_pddl)


def load_problem_spec(domain: str, index: int) -> ProblemSpec:
    data = _load_prompts(domain)
    entry = _find_problem_entry(data, index)
    overview = _make_overview(domain, data, entry)
    actions = _make_actions(data)
    objects = _make_objects(entry)
    return ProblemSpec(
        domain=domain,
        index=index,
        overview=overview,
        objects=objects,
        actions=actions,
    )


# ---------------------------------------------------------------------------
# Public tool backends (what you wrap as LLM tools)
# ---------------------------------------------------------------------------


def get_task_overview(domain: str, index: int) -> dict[str, str]:
    spec = load_problem_spec(domain, index)
    return {
        "description": spec.overview.description,
        "initial_state": spec.overview.initial_state,
        "goal": spec.overview.goal,
    }


def list_objects(domain: str, index: int, kind: str | None = None) -> dict[str, Any]:
    spec = load_problem_spec(domain, index)
    objs = spec.objects
    if kind:
        kind = str(kind).strip()
        objs = [o for o in objs if o.kind == kind]

    return {
        "kind": kind or "all",
        "objects": [
            {
                "name": o.name,
                "kind": o.kind,
                "summary": o.summary,
                "attributes": o.attributes,
            }
            for o in objs
        ],
    }


def describe_object(domain: str, index: int, name: str) -> dict[str, Any]:
    spec = load_problem_spec(domain, index)
    name = str(name).strip()
    for o in spec.objects:
        if o.name == name:
            return {
                "name": o.name,
                "kind": o.kind,
                "summary": o.summary,
                "attributes": o.attributes,
            }
    raise KeyError(f"Unknown object '{name}' for domain '{domain}' problem {index}.")


def get_action_schemas(domain: str) -> dict[str, Any]:
    data = _load_prompts(domain)
    actions = _make_actions(data)
    return {
        "actions": [
            {
                "name": a.name,
                "signature": a.signature,
                "parameters": [{"name": p.name, "kind": p.kind} for p in a.parameters],
                "preconditions": a.preconditions,
                "effects": a.effects,
                "cost": a.cost,
            }
            for a in actions
        ]
    }


def submit_plan(
    domain: str,
    index: int,
    steps: list[dict[str, Any]],
    *,
    val_path: str | None = None,
    tolerance: float = 0.001,
    check_redundancy: bool = False,
) -> dict[str, Any]:
    domain_path, problem_path = _resolve_pddl_paths(domain, index)

    lines: list[str] = []
    for s in steps:
        action = str(s.get("action") or "").strip()
        if not action:
            continue
        args = s.get("args") or []
        arg_str = " ".join(str(a).strip() for a in args if str(a).strip())
        lines.append(f"({action} {arg_str})" if arg_str else f"({action})")

    plan_text = "\n".join(lines) + ("\n" if lines else "")

    flags = ("-v", "-t", str(tolerance))
    metrics = compute_metrics(
        domain=domain_path,
        problem=problem_path,
        plan_text=plan_text,
        val_path=val_path,
        flags=flags,
        check_redundancy=check_redundancy,
    )

    return {
        "accepted": bool(metrics.valid),
        "length": metrics.length,
        "cost_value": metrics.cost_value,
        "first_failure_at": metrics.first_failure_at,
        "first_failed_action": metrics.first_failed_action,
        "first_failure_reason": metrics.first_failure_reason,
        "first_failure_detail": metrics.first_failure_detail,
        "unsat_count": metrics.unsat_count,
        "redundant_indices": metrics.redundant_indices,
        "advice_count": metrics.advice_count,
        "advice_top_predicates": metrics.advice_top_predicates,
        "val_attempts": getattr(metrics, "val_attempts", 1),
        "val_warning": getattr(metrics, "val_warning", None),
    }


# =============================================================================
# NL-only tools + NL plan compilation (PDDL-agnostic to purple)
# =============================================================================

# Updated regex to match PDDL identifiers (e.g., pick-up, move-block)
_ACTION_NAME_RE = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9_\-]*)", re.IGNORECASE)


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _strip_code_fences(raw: str) -> str:
    if "```" not in (raw or ""):
        return raw or ""
    parts = (raw or "").split("```")
    best = ""
    for i, p in enumerate(parts):
        if i % 2 == 1:
            lines = p.splitlines()
            body = (
                "\n".join(lines[1:])
                if lines and re.match(r"^[a-zA-Z0-9_-]+$", lines[0].strip())
                else p
            )
            if len(body) > len(best):
                best = body
    return best if best else (raw or "")


def _get_action_map(domain: str) -> dict[str, ActionSchema]:
    """Returns a dict {action_name: schema}."""
    data = _load_prompts(domain)
    actions = _make_actions(data)
    # Case-insensitive map for easier matching
    return {a.name.lower(): a for a in actions}


def get_task_overview_nl(domain: str, index: int) -> str:
    ov = get_task_overview(domain, index)
    lines: list[str] = []
    if ov.get("description"):
        lines.append("Task description:")
        lines.append(ov["description"].strip())
    if ov.get("initial_state"):
        lines.append("")
        lines.append("Initial situation:")
        lines.append(ov["initial_state"].strip())
    if ov.get("goal"):
        lines.append("")
        lines.append("Goal:")
        lines.append(ov["goal"].strip())
    return "\n".join(lines).strip() or "No overview available."


def list_objects_nl(domain: str, index: int, kind: str | None = None) -> str:
    res = list_objects(domain, index, kind=kind)
    objs = res.get("objects") or []
    if not objs:
        return "No objects."
    lines: list[str] = []
    for o in objs:
        name = str(o.get("name") or "").strip()
        k = str(o.get("kind") or "").strip()
        summary = str(o.get("summary") or "").strip()
        lines.append(f"- {name} (type: {k})" + (f" — {summary}" if summary else ""))
    return "\n".join(lines)


def describe_object_nl(domain: str, index: int, name: str) -> str:
    o = describe_object(domain, index, name)
    lines = [
        f"Name: {o.get('name', '')}",
        f"Type: {o.get('kind', '')}",
    ]
    if o.get("summary"):
        lines.append(f"Summary: {o['summary']}")
    attrs = o.get("attributes") or {}
    if isinstance(attrs, dict) and attrs:
        lines.append("Attributes:")
        for k, v in attrs.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines).strip()


def list_action_types_nl(domain: str) -> str:
    action_map = _get_action_map(domain)
    if not action_map:
        return "No action types defined for this domain in prompts.json."

    lines: list[str] = []
    # Sort by name for stability
    for name in sorted(action_map.keys()):
        a = action_map[name]
        # Use actual name instead of A1/A2
        lines.append(f"Action: {a.name}")
        if a.parameters:
            lines.append(
                "Parameters (in order): " + ", ".join(p.name for p in a.parameters)
            )
        else:
            lines.append("Parameters: none")
        if a.preconditions:
            lines.append("Allowed when:")
            for p in a.preconditions:
                lines.append(f"- {p}")
        if a.effects:
            lines.append("Effects:")
            for e in a.effects:
                lines.append(f"- {e}")
        lines.append("")
    return "\n".join(lines).strip()


def describe_action(domain: str, action_name: str) -> str:
    action_map = _get_action_map(domain)
    aname = (action_name or "").strip().lower()
    if aname not in action_map:
        return (
            f"Unknown action '{action_name}'. Use list_action_types_nl(domain) to see "
            "valid names."
        )
    a = action_map[aname]
    lines = [f"Action: {a.name}"]
    if a.parameters:
        lines.append(
            "Parameters (in order): " + ", ".join(p.name for p in a.parameters)
        )
    else:
        lines.append("Parameters: none")
    if a.preconditions:
        lines.append("Allowed when:")
        for p in a.preconditions:
            lines.append(f"- {p}")
    if a.effects:
        lines.append("Effects:")
        for e in a.effects:
            lines.append(f"- {e}")
    return "\n".join(lines)


def _parse_nl_step(line: str) -> tuple[str, dict[str, str] | list[str]]:
    ln = (line or "").strip()
    if not ln:
        raise ValueError("empty step")

    # Strip leading numbers "1. pick-up..." or "1) pick-up..."
    ln = re.sub(r"^\s*\d+\s*[\)\.:\-]\s*", "", ln).strip()

    # Match action name at start of line
    m = _ACTION_NAME_RE.search(ln)
    if not m:
        raise ValueError(f"Could not parse action name from: '{ln}'")

    act_name = m.group(1).lower()
    after = ln[m.end() :].strip()

    # Remove parens if user wrapped it like "pick-up(A, B)"
    if after.startswith("(") and after.endswith(")"):
        after = after[1:-1].strip()

    payload = ""
    # Handle "pick-up with: X=A" or "pick-up: A"
    if "with:" in after.lower():
        payload = re.split(r"with:", after, flags=re.IGNORECASE, maxsplit=1)[1].strip()
    elif after.startswith(":"):
        payload = after[1:].strip()
    else:
        payload = after.strip()

    if not payload:
        return act_name, []

    # Check for kwargs style "X=A, Y=B"
    if "=" in payload or re.search(r"\w+\s*:\s*\S+", payload):
        parts = [p.strip() for p in payload.split(",") if p.strip()]
        out: dict[str, str] = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
            elif ":" in p:
                k, v = p.split(":", 1)
            else:
                continue
            out[k.strip()] = v.strip()
        return act_name, out

    # Positional args "A B" or "A, B"
    clean_args = payload.replace(",", " ")
    return act_name, [t for t in clean_args.split() if t.strip()]


def compile_nl_plan(domain: str, raw: str) -> tuple[str, list[str]]:
    action_map = _get_action_map(domain)
    text = _strip_code_fences(raw)
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    errors: list[str] = []
    pddl_lines: list[str] = []

    for ln in lines:
        try:
            act_name, args = _parse_nl_step(ln)
            if act_name not in action_map:
                raise ValueError(f"Unknown action '{act_name}'")

            schema = action_map[act_name]
            param_names = [p.name for p in schema.parameters]

            ordered: list[str] = []
            if isinstance(args, dict):
                # Kwargs mode
                keymap = {_norm_key(k): v for k, v in args.items()}
                for pn in param_names:
                    k = _norm_key(pn)
                    if k not in keymap:
                        raise ValueError(
                            f"missing parameter '{pn}' for action '{schema.name}'"
                        )
                    ordered.append(str(keymap[k]).strip())
            else:
                # Positional mode
                if len(args) != len(param_names):
                    raise ValueError(
                        f"Action '{schema.name}' expects {len(param_names)} args "
                        f"({', '.join(param_names)}), got {len(args)}."
                    )
                ordered = [str(x).strip() for x in args]

            arg_str = " ".join(ordered)
            pddl_lines.append(
                f"({schema.name} {arg_str})" if arg_str else f"({schema.name})"
            )
        except Exception as e:
            errors.append(f"{ln} -> {e}")

    plan_text = "\n".join(pddl_lines) + ("\n" if pddl_lines else "")
    return plan_text, errors


def submit_plan_nl(
    domain: str,
    index: int,
    plan_steps_text: str,
    *,
    val_path: str | None = None,
    tolerance: float = 0.001,
    check_redundancy: bool = False,
) -> str:
    domain_path, problem_path = _resolve_pddl_paths(domain, index)
    plan_text, errors = compile_nl_plan(domain, plan_steps_text)
    if errors:
        return "Plan parse failed:\n- " + "\n- ".join(errors)
    if not plan_text.strip():
        return "Plan is empty."

    flags = ("-v", "-t", str(tolerance))
    metrics = compute_metrics(
        domain=domain_path,
        problem=problem_path,
        plan_text=plan_text,
        val_path=val_path,
        flags=flags,
        check_redundancy=check_redundancy,
    )

    lines = [
        f"Accepted: {'YES' if metrics.valid else 'NO'}",
        f"Plan length: {metrics.length}",
    ]
    if metrics.cost_value is not None:
        lines.append(f"Plan cost/value: {metrics.cost_value}")
    if not metrics.valid:
        lines.append(f"Failure category: {metrics.failure_reason or 'unknown_failure'}")
        if metrics.first_failure_at is not None:
            lines.append(f"First failing step: {metrics.first_failure_at}")
        if metrics.first_failed_action:
            lines.append(f"First failed action: {metrics.first_failed_action}")
        if metrics.first_failure_detail:
            lines.append(f"Details: {metrics.first_failure_detail}")
        lines.append(f"Unsatisfied conditions count: {metrics.unsat_count}")
    return "\n".join(lines)


# -------------------------------
# Stateful NL execution (episode)
# -------------------------------


@dataclass(slots=True)
class _NLEpisode:
    domain: str | None = None
    index: int | None = None
    domain_path: str | None = None
    problem_path: str | None = None
    val_path: str | None = None
    tolerance: float = 0.001
    pddl_steps: list[str] | None = None

    # Diagnostics for debugging / demos
    last_val_stdout: str | None = None
    last_val_steps: int = 0

    def __post_init__(self) -> None:
        self.pddl_steps = []


_EP = _NLEpisode()


def reset_episode_nl(
    domain: str,
    index: int,
    *,
    val_path: str | None = None,
    tolerance: float = 0.001,
) -> str:
    _EP.domain = domain
    _EP.index = int(index)
    _EP.domain_path, _EP.problem_path = _resolve_pddl_paths(domain, index)
    _EP.val_path = val_path
    _EP.tolerance = float(tolerance)
    _EP.pddl_steps = []
    _EP.last_val_stdout = None
    _EP.last_val_steps = 0
    return f"State reset for domain '{domain}', problem {index}. Step counter = 0."


def get_history_nl() -> str:
    if not _EP.pddl_steps:
        return "No actions executed yet."
    out: list[str] = []
    for i, ln in enumerate(_EP.pddl_steps, start=1):
        inner = ln.strip()
        inner = (
            inner[1:-1].strip()
            if inner.startswith("(") and inner.endswith(")")
            else inner
        )
        toks = inner.split()
        act = toks[0] if toks else "unknown"
        args = toks[1:]
        out.append(f"{i}) {act}" + (f" {' '.join(args)}" if args else ""))
    return "\n".join(out).strip()


def undo_nl(to_step: int) -> str:
    k = max(0, int(to_step))
    if not _EP.pddl_steps:
        return "Reverted to step 0."
    if k > len(_EP.pddl_steps):
        k = len(_EP.pddl_steps)
    _EP.pddl_steps = _EP.pddl_steps[:k]
    return f"Reverted to step {k}."


def _translate_fact(fact: str) -> str:
    inner = fact.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()

    parts = inner.split()
    if not parts:
        return fact

    pred = parts[0].lower()
    readable_pred = pred.replace("-", " ").replace("_", " ")
    args = parts[1:]

    if len(args) == 0:
        return f"{readable_pred} is true"
    if len(args) == 1:
        return f"{args[0]} is {readable_pred}"
    if len(args) == 2:
        return f"{args[0]} is {readable_pred} {args[1]}"

    return fact


def _translate_error(detail_raw: str) -> str:
    """
    Translates PDDL/VAL error messages into friendly Natural Language
    using generic grammatical heuristics for ANY domain.
    """
    txt = detail_raw.strip()

    # 1. Strip VAL "Set ... to true" wrapper
    m_set = re.search(r"Set\s+(\(.+\))\s+to\s+true", txt, re.IGNORECASE)
    if m_set:
        txt = m_set.group(1)

    # 2. Parse Predicate
    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1]

    parts = txt.split()
    if not parts:
        return detail_raw

    pred = parts[0].lower()
    # Normalize: "on-table" -> "on table", "connected-to" -> "connected to"
    readable_pred = pred.replace("-", " ").replace("_", " ")
    args = parts[1:]

    # 3. Apply Generic Templates based on Argument Count (Arity)

    # Case 0: Global Flags (e.g., (handempty))
    if len(args) == 0:
        return f"The condition '{readable_pred}' should be met."

    # Case 1: Properties (e.g., (clear a), (free gripper))
    if len(args) == 1:
        # Output: "a should be clear" or "gripper should be free"
        return f"{args[0]} should be {readable_pred}."

    # Case 2: Relations (e.g., (on a b), (at truck loc), (connected loc1 loc2))
    if len(args) == 2:
        # Output: "a should be on b" or "truck should be at loc"
        return f"{args[0]} should be {readable_pred} {args[1]}."

    # Case 3: Complex Relations (3+ args)
    # Fallback to functional style: "link(a, b, c) should be true"
    return f"The relationship '{readable_pred}' should hold for ({', '.join(args)})."


def act_nl(step_text: str) -> str:
    """
    Try to append exactly one action (provided as NL format).
    """
    if not (_EP.domain and _EP.domain_path and _EP.problem_path):
        return "No episode loaded. Call reset_episode_nl(domain, index) first."

    plan_text, errors = compile_nl_plan(_EP.domain, step_text)
    if errors:
        return "Executed: NO\nReason: " + errors[0]
    step_line = plan_text.strip().splitlines()
    if not step_line:
        return "Executed: NO\nReason: could not compile the step."

    candidate = (_EP.pddl_steps or []) + [step_line[0]]
    cand_text = "\n".join(candidate) + "\n"

    flags = ("-v", "-t", str(_EP.tolerance))
    res = run_val(
        _EP.domain_path,
        _EP.problem_path,
        cand_text,
        val_path=_EP.val_path,
        flags=flags,
    )

    if res.unsatisfied:
        u = res.unsatisfied[-1]
        at = u.at_action_index if u.at_action_index is not None else "?"

        raw_detail = (u.detail or "").strip()
        friendly_error = _translate_error(raw_detail)

        return f"Executed: NO\nAt step: {at}\nDetail: {friendly_error}"

    if res.failure_reason == "no_output":
        return "Executed: NO\nReason: VAL produced no usable output."

    if (not res.ok) and (res.failure_reason not in (None, "goal_not_satisfied")):
        return f"Executed: NO\nReason: VAL failure ({res.failure_reason})"

    _EP.pddl_steps = candidate
    _EP.last_val_stdout = res.stdout
    _EP.last_val_steps = len(_EP.pddl_steps)
    return f"Executed: YES\nStep counter now: {len(_EP.pddl_steps)}"


# -------------------------------
# State reconstruction (debug)
# -------------------------------


_INIT_ASSIGN_RE = re.compile(
    r"^\(\s*=\s*\(\s*([^)]+)\s*\)\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$"
)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _extract_balanced_block(text: str, start_idx: int) -> str:
    depth = 0
    i = start_idx
    n = len(text)
    while i < n and text[i] != "(":
        i += 1
    start = i
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return text[start:]


def _split_top_level_sexps(block: str) -> list[str]:
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in block:
        if c == "(":
            if depth == 0:
                buf = []
            depth += 1
        if depth > 0:
            buf.append(c)
        if c == ")":
            depth -= 1
            if depth == 0 and buf:
                out.append("".join(buf).strip())
                buf = []
    return out


def _parse_problem_init(problem_path: str) -> tuple[set[str], dict[str, float]]:
    """
    Returns:
      - facts: set of '(pred arg1 ...)' strings (LOWERCASE ONLY)
      - functions: {func_name: value}
    """
    txt = _read_text(problem_path)
    pos = txt.lower().find("(:init")
    if pos < 0:
        return set(), {}
    init_block = _extract_balanced_block(txt, pos)

    inner = init_block.strip()
    inner = inner[1:-1].strip()  # remove outer parens
    if inner.lower().startswith(":init"):
        inner = inner[5:].strip()

    sexps = _split_top_level_sexps(inner)
    facts: set[str] = set()
    funcs: dict[str, float] = {}

    for e in sexps:
        e = re.sub(r"\s+", " ", e.strip())
        if not e.startswith("(") or not e.endswith(")"):
            continue
        if e.lower().startswith("(not "):
            continue
        m = _INIT_ASSIGN_RE.match(e)
        if m:
            func_inner = m.group(1).strip()
            func_name = func_inner.split()[0].lower()  # Lowercase function names too
            try:
                funcs[func_name] = float(m.group(2))
            except Exception:
                pass
            continue

        facts.add(e.lower())

    return facts, funcs


def _apply_trace(facts: set[str], adds: list[str], deletes: list[str]) -> None:
    for d in deletes or []:
        dd = re.sub(r"\s+", " ", (d or "").strip()).lower()  # Lowercase
        if dd:
            facts.discard(dd)
    for a in adds or []:
        aa = re.sub(r"\s+", " ", (a or "").strip()).lower()  # Lowercase
        if aa:
            facts.add(aa)


def get_state_nl(max_facts: int = 200) -> str:
    """
    Read-only debug: reconstruct current facts from problem :init + VAL trace deltas.
    Re-runs VAL on the current prefix.
    """
    if not (_EP.domain and _EP.domain_path and _EP.problem_path):
        return "No episode loaded. Call reset_episode_nl(domain, index) first."

    plan_text = "\n".join(_EP.pddl_steps or []) + (
        "\n" if (_EP.pddl_steps or []) else ""
    )
    flags = ("-v", "-t", str(_EP.tolerance))
    res = run_val(
        _EP.domain_path,
        _EP.problem_path,
        plan_text,
        val_path=_EP.val_path,
        flags=flags,
    )

    facts, funcs = _parse_problem_init(_EP.problem_path)

    for st in res.steps:
        _apply_trace(facts, st.adds, st.deletes)

    if res.value is not None:
        funcs["total-cost"] = float(res.value)

    # Facts are already lowercase now, so sorting is stable
    facts_sorted = sorted(facts)
    if max_facts and len(facts_sorted) > max_facts:
        facts_sorted = facts_sorted[:max_facts] + [
            f"... ({len(facts) - max_facts} more facts)"
        ]

    lines: list[str] = []
    step_count = len(_EP.pddl_steps or [])
    lines.append(
        f"Episode: domain={_EP.domain}, problem={_EP.index}, steps={step_count}"
    )
    if funcs:
        lines.append("Functions:")
        for k in sorted(funcs.keys()):
            lines.append(f"- {k} = {funcs[k]}")

    lines.append("Facts:")
    for fct in facts_sorted:
        # --- CHANGE START: Translate the fact ---
        lines.append(f"- {_translate_fact(fct)}")
        # --- CHANGE END ---

    return "\n".join(lines)


def submit_episode_nl(*, check_redundancy: bool = False) -> str:
    """
    Validate the current prefix as a full plan (goal satisfaction enforced).
    """
    if not (_EP.domain and _EP.domain_path and _EP.problem_path):
        return "No episode loaded. Call reset_episode_nl(domain, index) first."

    plan_text = "\n".join(_EP.pddl_steps or []) + (
        "\n" if (_EP.pddl_steps or []) else ""
    )
    flags = ("-v", "-t", str(_EP.tolerance))
    metrics = compute_metrics(
        domain=_EP.domain_path,
        problem=_EP.problem_path,
        plan_text=plan_text,
        val_path=_EP.val_path,
        flags=flags,
        check_redundancy=check_redundancy,
    )

    lines = [
        f"Accepted: {'YES' if metrics.valid else 'NO'}",
        f"Plan length: {metrics.length}",
    ]
    if metrics.cost_value is not None:
        lines.append(f"Plan cost/value: {metrics.cost_value}")
    if not metrics.valid:
        lines.append(f"Failure category: {metrics.failure_reason or 'unknown_failure'}")
        if metrics.first_failure_at is not None:
            lines.append(f"First failing step: {metrics.first_failure_at}")
        if metrics.first_failed_action:
            lines.append(f"First failed action: {metrics.first_failed_action}")
        if metrics.first_failure_detail:
            lines.append(f"Details: {metrics.first_failure_detail}")
        lines.append(f"Unsatisfied conditions count: {metrics.unsat_count}")
    return "\n".join(lines)
