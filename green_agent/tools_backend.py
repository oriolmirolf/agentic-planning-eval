# /agentic-planning-eval/green_agent/tools_backend.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .metrics import compute_metrics
from .val_wrapper import run_val

# ---------------------------------------------------------------------------
# DSPy-like tool signatures (NO external dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SigField:
    name: str
    type: str
    desc: str


@dataclass(frozen=True, slots=True)
class ToolSignature:
    """
    A small, DSPy-like signature object you can use when registering tools.
    """
    name: str
    inputs: list[SigField]
    output: SigField
    doc: str


# NOTE: Per request, the submit signature MUST ONLY describe submitting the
# current episode plan (actions executed so far). It must NOT mention providing text.
TOOL_SIGNATURES: dict[str, ToolSignature] = {
    "get_task_overview": ToolSignature(
        name="get_task_overview",
        inputs=[
            SigField("domain", "str", "Domain name under examples/<domain>."),
            SigField("index", "int", "Problem index (1..N)."),
        ],
        output=SigField("overview", "str", "Human-readable task description, init, and goal."),
        doc="Return a natural-language overview for a given domain/problem.",
    ),
    "list_objects": ToolSignature(
        name="list_objects",
        inputs=[
            SigField("domain", "str", "Domain name under examples/<domain>."),
            SigField("index", "int", "Problem index (1..N)."),
            SigField("kind", "Optional[str]", "Optional type filter (e.g., 'room', 'ball')."),
        ],
        output=SigField("objects", "str", "Bullet list of objects with types and summaries."),
        doc="List objects available in the current problem (optionally filtered by kind).",
    ),
    "describe_object": ToolSignature(
        name="describe_object",
        inputs=[
            SigField("domain", "str", "Domain name under examples/<domain>."),
            SigField("index", "int", "Problem index (1..N)."),
            SigField("name", "str", "Exact object name to describe."),
        ],
        output=SigField("object", "str", "Human-readable object description and attributes."),
        doc="Describe one object by name for a given problem.",
    ),
    "list_action_types": ToolSignature(
        name="list_action_types",
        inputs=[SigField("domain", "str", "Domain name under examples/<domain>.")],
        output=SigField("actions", "str", "Action names, parameters, preconditions, effects."),
        doc="List action schemas in human-readable form.",
    ),
    "describe_action": ToolSignature(
        name="describe_action",
        inputs=[
            SigField("domain", "str", "Domain name under examples/<domain>."),
            SigField("action_name", "str", "Action name (case-insensitive)."),
        ],
        output=SigField("action", "str", "Human-readable action schema for a single action."),
        doc="Describe one action schema by name.",
    ),
    "reset_episode": ToolSignature(
        name="reset_episode",
        inputs=[
            SigField("domain", "str", "Domain name under examples/<domain>."),
            SigField("index", "int", "Problem index (1..N)."),
            SigField("val_path", "Optional[str]", "Optional custom VAL binary path."),
            SigField("tolerance", "float", "Numeric tolerance for VAL checks (default: 1e-3)."),
        ],
        output=SigField("status", "str", "Reset confirmation string."),
        doc="Initialize an interactive episode for a specific domain/problem.",
    ),
    "act": ToolSignature(
        name="act",
        inputs=[
            SigField("step_text", "str", "One action step in NL-ish format (one step)."),
        ],
        output=SigField("result", "str", "Executed YES/NO and error detail if rejected."),
        doc="Attempt to append exactly one action step to the current episode.",
    ),
    "get_history": ToolSignature(
        name="get_history",
        inputs=[],
        output=SigField("history", "str", "Numbered list of executed actions so far."),
        doc="Get the current episode action history.",
    ),
    "get_state": ToolSignature(
        name="get_state",
        inputs=[SigField("max_facts", "int", "Max facts to display (default: 200).")],
        output=SigField("state", "str", "Human-readable facts/functions reconstructed from VAL trace."),
        doc="Debug tool: reconstruct and print current world facts from :init + VAL deltas.",
    ),
    "submit": ToolSignature(
        name="submit",
        inputs=[SigField("check_redundancy", "bool", "If true, compute redundancy diagnostics.")],
        output=SigField("summary", "str", "Final validation summary for the current episode plan."),
        doc="Validate the CURRENT episode plan (actions executed so far) as a full plan.",
    ),
}

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
        raise FileNotFoundError(f"prompts.json not found for domain '{domain}' at {path}")
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


def _make_overview(domain: str, data: dict[str, Any], entry: dict[str, Any]) -> TaskOverview:
    domain_prompt = (data.get("domain_prompt") or "").strip()
    overview = entry.get("overview") or {}

    desc = (
        overview.get("description")
        or "\n\n".join(p for p in (domain_prompt, (entry.get("prompt") or "").strip()) if p).strip()
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
        raise FileNotFoundError(f"domain.pddl not found for domain '{domain}' at {domain_pddl}")
    problem_pddl = problems_dir / f"problem{int(index)}.pddl"
    if not problem_pddl.exists():
        raise FileNotFoundError(f"problem{index}.pddl not found for domain '{domain}' at {problem_pddl}")
    return str(domain_pddl), str(problem_pddl)


def load_problem_spec(domain: str, index: int) -> ProblemSpec:
    data = _load_prompts(domain)
    entry = _find_problem_entry(data, index)
    overview = _make_overview(domain, data, entry)
    actions = _make_actions(data)
    objects = _make_objects(entry)
    return ProblemSpec(domain=domain, index=index, overview=overview, objects=objects, actions=actions)


def _get_action_map(domain: str) -> dict[str, ActionSchema]:
    """Returns a dict {action_name_lower: schema}."""
    data = _load_prompts(domain)
    actions = _make_actions(data)
    return {a.name.lower(): a for a in actions}


# =============================================================================
# "JSON-ish" accessors (kept for internal / non-agent usage)
# =============================================================================

# These are intentionally *not* the primary agent tools; they are useful for UI/debug.
# The agent should use the human-readable tools below.


def get_task_overview_json(domain: str, index: int) -> dict[str, str]:
    spec = load_problem_spec(domain, index)
    return {
        "description": spec.overview.description,
        "initial_state": spec.overview.initial_state,
        "goal": spec.overview.goal,
    }


def list_objects_json(domain: str, index: int, kind: str | None = None) -> dict[str, Any]:
    spec = load_problem_spec(domain, index)
    objs = spec.objects
    if kind:
        kind = str(kind).strip()
        objs = [o for o in objs if o.kind == kind]

    return {
        "kind": kind or "all",
        "objects": [
            {"name": o.name, "kind": o.kind, "summary": o.summary, "attributes": o.attributes}
            for o in objs
        ],
    }


def describe_object_json(domain: str, index: int, name: str) -> dict[str, Any]:
    spec = load_problem_spec(domain, index)
    name = str(name).strip()
    for o in spec.objects:
        if o.name == name:
            return {"name": o.name, "kind": o.kind, "summary": o.summary, "attributes": o.attributes}
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


# =============================================================================
# Human-readable tools (agent-facing): same meaning as old *_nl
# =============================================================================

# Updated regex to match PDDL identifiers (e.g., pick-up, move-block)
_ACTION_NAME_RE = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9_\-]*)", re.IGNORECASE)


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _strip_code_fences(raw: str) -> str:
    """
    Extract the biggest fenced code block if present; otherwise return raw.
    Useful because many models wrap plans in ```...```.
    """
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


def get_task_overview(domain: str, index: int) -> str:
    """
    Agent-facing overview (formerly get_task_overview_nl).
    """
    ov = get_task_overview_json(domain, index)
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


def list_objects(domain: str, index: int, kind: str | None = None) -> str:
    """
    Agent-facing object listing (formerly list_objects_nl).
    """
    res = list_objects_json(domain, index, kind=kind)
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


def describe_object(domain: str, index: int, name: str) -> str:
    """
    Agent-facing object description (formerly describe_object_nl).
    """
    o = describe_object_json(domain, index, name)
    lines = [f"Name: {o.get('name', '')}", f"Type: {o.get('kind', '')}"]
    if o.get("summary"):
        lines.append(f"Summary: {o['summary']}")
    attrs = o.get("attributes") or {}
    if isinstance(attrs, dict) and attrs:
        lines.append("Attributes:")
        for k, v in attrs.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines).strip()


def list_action_types(domain: str) -> str:
    """
    Agent-facing action list (formerly list_action_types_nl / list_actions_nl).
    """
    action_map = _get_action_map(domain)
    if not action_map:
        return "No action types defined for this domain in prompts.json."

    lines: list[str] = []
    for name in sorted(action_map.keys()):
        a = action_map[name]
        lines.append(f"Action: {a.name}")
        if a.parameters:
            lines.append("Parameters (in order): " + ", ".join(p.name for p in a.parameters))
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
    """
    Agent-facing single-action schema description.
    """
    action_map = _get_action_map(domain)
    aname = (action_name or "").strip().lower()
    if aname not in action_map:
        return f"Unknown action '{action_name}'. Use list_action_types(domain) to see valid names."
    a = action_map[aname]

    lines = [f"Action: {a.name}"]
    if a.parameters:
        lines.append("Parameters (in order): " + ", ".join(p.name for p in a.parameters))
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


def _parse_step(line: str) -> tuple[str, dict[str, str] | list[str]]:
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
    if "with:" in after.lower():
        payload = re.split(r"with:", after, flags=re.IGNORECASE, maxsplit=1)[1].strip()
    elif after.startswith(":"):
        payload = after[1:].strip()
    else:
        payload = after.strip()

    if not payload:
        return act_name, []

    # kwargs style "X=A, Y=B" or "X: A"
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


def compile_plan(domain: str, raw: str) -> tuple[str, list[str]]:
    """
    Compile NL-ish step lines into PDDL lines using prompts.json action schemas.
    Returns (plan_text, errors).
    """
    action_map = _get_action_map(domain)
    text = _strip_code_fences(raw)
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    errors: list[str] = []
    pddl_lines: list[str] = []

    for ln in lines:
        try:
            act_name, args = _parse_step(ln)
            if act_name not in action_map:
                raise ValueError(f"Unknown action '{act_name}'")

            schema = action_map[act_name]
            param_names = [p.name for p in schema.parameters]

            ordered: list[str] = []
            if isinstance(args, dict):
                keymap = {_norm_key(k): v for k, v in args.items()}
                for pn in param_names:
                    k = _norm_key(pn)
                    if k not in keymap:
                        raise ValueError(f"missing parameter '{pn}' for action '{schema.name}'")
                    ordered.append(str(keymap[k]).strip())
            else:
                if len(args) != len(param_names):
                    raise ValueError(
                        f"Action '{schema.name}' expects {len(param_names)} args "
                        f"({', '.join(param_names)}), got {len(args)}."
                    )
                ordered = [str(x).strip() for x in args]

            arg_str = " ".join(ordered)
            pddl_lines.append(f"({schema.name} {arg_str})" if arg_str else f"({schema.name})")
        except Exception as e:
            errors.append(f"{ln} -> {e}")

    plan_text = "\n".join(pddl_lines) + ("\n" if pddl_lines else "")
    return plan_text, errors


# -------------------------------
# Stateful execution (episode)
# -------------------------------


@dataclass(slots=True)
class _Episode:
    domain: str | None = None
    index: int | None = None
    domain_path: str | None = None
    problem_path: str | None = None
    val_path: str | None = None
    tolerance: float = 0.001
    pddl_steps: list[str] | None = None

    last_val_stdout: str | None = None
    last_val_steps: int = 0

    def __post_init__(self) -> None:
        self.pddl_steps = []


_EP = _Episode()


def reset_episode(domain: str, index: int, *, val_path: str | None = None, tolerance: float = 0.001) -> str:
    _EP.domain = domain
    _EP.index = int(index)
    _EP.domain_path, _EP.problem_path = _resolve_pddl_paths(domain, index)
    _EP.val_path = val_path
    _EP.tolerance = float(tolerance)
    _EP.pddl_steps = []
    _EP.last_val_stdout = None
    _EP.last_val_steps = 0
    return f"State reset for domain '{domain}', problem {index}. Step counter = 0."


def get_history() -> str:
    if not _EP.pddl_steps:
        return "No actions executed yet."
    out: list[str] = []
    for i, ln in enumerate(_EP.pddl_steps, start=1):
        inner = ln.strip()
        inner = inner[1:-1].strip() if inner.startswith("(") and inner.endswith(")") else inner
        toks = inner.split()
        act = toks[0] if toks else "unknown"
        args = toks[1:]
        out.append(f"{i}) {act}" + (f" {' '.join(args)}" if args else ""))
    return "\n".join(out).strip()


def undo(to_step: int) -> str:
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
    txt = detail_raw.strip()

    m_set = re.search(r"Set\s+(\(.+\))\s+to\s+true", txt, re.IGNORECASE)
    if m_set:
        txt = m_set.group(1)

    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1]

    parts = txt.split()
    if not parts:
        return detail_raw

    pred = parts[0].lower()
    readable_pred = pred.replace("-", " ").replace("_", " ")
    args = parts[1:]

    if len(args) == 0:
        return f"The condition '{readable_pred}' should be met."
    if len(args) == 1:
        return f"{args[0]} should be {readable_pred}."
    if len(args) == 2:
        return f"{args[0]} should be {readable_pred} {args[1]}."
    return f"The relationship '{readable_pred}' should hold for ({', '.join(args)})."


def act(step_text: str) -> str:
    """
    Try to append exactly one action (provided as NL-ish format).
    """
    if not (_EP.domain and _EP.domain_path and _EP.problem_path):
        return "No episode loaded. Call reset_episode(domain, index) first."

    plan_text, errors = compile_plan(_EP.domain, step_text)
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


_INIT_ASSIGN_RE = re.compile(r"^\(\s*=\s*\(\s*([^)]+)\s*\)\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$")


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
            func_name = func_inner.split()[0].lower()
            try:
                funcs[func_name] = float(m.group(2))
            except Exception:
                pass
            continue
        facts.add(e.lower())

    return facts, funcs


def _apply_trace(facts: set[str], adds: list[str], deletes: list[str]) -> None:
    for d in deletes or []:
        dd = re.sub(r"\s+", " ", (d or "").strip()).lower()
        if dd:
            facts.discard(dd)
    for a in adds or []:
        aa = re.sub(r"\s+", " ", (a or "").strip()).lower()
        if aa:
            facts.add(aa)


def get_state(max_facts: int = 200) -> str:
    """
    Read-only debug: reconstruct current facts from problem :init + VAL trace deltas.
    Re-runs VAL on the current prefix.
    """
    if not (_EP.domain and _EP.domain_path and _EP.problem_path):
        return "No episode loaded. Call reset_episode(domain, index) first."

    plan_text = "\n".join(_EP.pddl_steps or []) + ("\n" if (_EP.pddl_steps or []) else "")
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

    facts_sorted = sorted(facts)
    if max_facts and len(facts_sorted) > max_facts:
        facts_sorted = facts_sorted[:max_facts] + [f"... ({len(facts) - max_facts} more facts)"]

    lines: list[str] = []
    step_count = len(_EP.pddl_steps or [])
    lines.append(f"Episode: domain={_EP.domain}, problem={_EP.index}, steps={step_count}")
    if funcs:
        lines.append("Functions:")
        for k in sorted(funcs.keys()):
            lines.append(f"- {k} = {funcs[k]}")
    lines.append("Facts:")
    for fct in facts_sorted:
        lines.append(f"- {_translate_fact(fct)}")
    return "\n".join(lines)


# =============================================================================
# ONE submit tool (episode-only)
# =============================================================================


def submit(*, check_redundancy: bool = False) -> str:
    """
    Validate the CURRENT episode prefix as a full plan (goal satisfaction enforced).

    IMPORTANT:
    - This tool does NOT accept an arbitrary plan.
    - It only validates actions executed so far via act(...).
    """
    if not (_EP.domain and _EP.domain_path and _EP.problem_path):
        return "No episode loaded. Call reset_episode(domain, index) first."

    plan_text = "\n".join(_EP.pddl_steps or []) + ("\n" if (_EP.pddl_steps or []) else "")
    if not plan_text.strip():
        return "Episode plan is empty."

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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases (optional, but helps existing code)
# ---------------------------------------------------------------------------

# Old *_nl names -> new agent-facing names
get_task_overview_nl = 1
list_objects_nl = list_objects
describe_object_nl = describe_object
list_action_types_nl = list_action_types
compile_nl_plan = compile_plan
reset_episode_nl = reset_episode
act_nl = act
get_history_nl = get_history
get_state_nl = get_state
submit_episode_nl = submit

# Keep undo alias name for older code
undo_nl = undo

__all__ = [
    "ToolSignature",
    "SigField",
    "TOOL_SIGNATURES",
    "load_problem_spec",
    "get_task_overview_json",
    "list_objects_json",
    "describe_object_json",
    "get_action_schemas",
    "get_task_overview",
    "list_objects",
    "describe_object",
    "list_action_types",
    "describe_action",
    "compile_plan",
    "reset_episode",
    "act",
    "undo",
    "get_history",
    "get_state",
    "submit",
]
