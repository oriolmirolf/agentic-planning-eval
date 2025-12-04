# /agentic-planning-eval/green_agent/tools_backend.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import compute_metrics

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
    """
    Expected prompts.json structure (per-domain):

      {
        "domain_prompt": "...",   # global description (optional, for plain-NL agents)
        "actions": [...],         # see _make_actions()
        "problems": [
          {
            "id": "p01",
            "prompt": "...",      # plain NL description for baseline prompting
            "overview": {
              "description": "...",
              "initial_state": "...",
              "goal": "..."
            },
            ...
          },
          ...
        ]
      }

    We fall back to domain_prompt + prompt when overview fields are missing,
    so you can incrementally migrate.
    """
    domain_prompt = (data.get("domain_prompt") or "").strip()
    overview = entry.get("overview") or {}

    # Description: overview.description OR domain_prompt + problem prompt
    desc = (
        overview.get("description")
        or "\n\n".join(
            p
            for p in (domain_prompt, (entry.get("prompt") or "").strip())
            if p
        ).strip()
    )

    # Initial state / goal: prefer structured fields; fall back to empty.
    init = (overview.get("initial_state") or entry.get("initial_state") or "").strip()
    goal = (overview.get("goal") or entry.get("goal") or "").strip()

    return TaskOverview(description=desc, initial_state=init, goal=goal)


def _make_actions(data: dict[str, Any]) -> list[ActionSchema]:
    """
    Expected domain-level actions section:

      "actions": [
        {
          "name": "move-vm",
          "signature": "move-vm(vm, from-server, to-server)",
          "parameters": [
            {"name": "vm", "kind": "vm"},
            {"name": "from-server", "kind": "server"},
            {"name": "to-server", "kind": "server"}
          ],
          "preconditions": [
            "VM is currently on from-server.",
            "to-server has enough free capacity.",
            ...
          ],
          "effects": [
            "VM is now on to-server.",
            "Capacity usage updated.",
            ...
          ],
          "cost": 1
        },
        ...
      ]
    """
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
        cost_raw = a.get("cost", 1)
        try:
            cost = float(cost_raw)
        except Exception:
            cost = 1.0

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
    """
    Expected per-problem objects structure:

      "objects": [
        {
          "name": "srv-eu-1",
          "kind": "server",
          "summary": "EU production server 1.",
          "attributes": {
            "region": "eu",
            "role": "prod",
            "capacity_cores": 16
          }
        },
        ...
      ]
    """
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
        objs.append(
            ObjectSpec(name=name, kind=kind, summary=summary, attributes=attrs)
        )
    return objs


def _resolve_pddl_paths(domain: str, index: int) -> tuple[str, str]:
    """
    Resolve domain.pddl and problem{index}.pddl.
    This duplicates the logic in green_agent.cli._resolve_paths but avoids imports.
    """
    base = Path("examples") / domain
    domain_pddl = base / "domain.pddl"
    problems_dir = base / "problems_pddl"

    if not domain_pddl.exists():
        raise FileNotFoundError("domain.pddl not found for "
                                "domain '{domain}' at {domain_pddl}")
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
    """
    Read-only tool: high-level description of the problem.

    Returns:
      {
        "description": "...",
        "initial_state": "...",
        "goal": "..."
      }
    """
    spec = load_problem_spec(domain, index)
    return {
        "description": spec.overview.description,
        "initial_state": spec.overview.initial_state,
        "goal": spec.overview.goal,
    }


def list_objects(
    domain: str, index: int, kind: str | None = None
) -> dict[str, Any]:
    """
    Read-only tool: list objects in this instance.

    Returns:
      {
        "kind": "<requested kind or 'all'>",
        "objects": [
          {
            "name": "...",
            "kind": "...",
            "summary": "...",
            "attributes": {...}
          },
          ...
        ]
      }
    """
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
    """
    Read-only tool: detailed info for a single object.

    Returns:
      {
        "name": "...",
        "kind": "...",
        "summary": "...",
        "attributes": {...}
      }

    Raises:
      KeyError if the object name is unknown.
    """
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
    """
    Read-only tool: list all available actions for this domain.

    Returns:
      {
        "actions": [
          {
            "name": "...",
            "signature": "move-vm(vm, from, to)",
            "parameters": [
              {"name": "vm", "kind": "vm"},
              ...
            ],
            "preconditions": ["...", ...],
            "effects": ["...", ...],
            "cost": 1.0
          },
          ...
        ]
      }
    """
    data = _load_prompts(domain)
    actions = _make_actions(data)
    return {
        "actions": [
            {
                "name": a.name,
                "signature": a.signature,
                "parameters": [
                    {"name": p.name, "kind": p.kind} for p in a.parameters
                ],
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
    """
    Write tool: evaluate a candidate plan.

    Args:
      domain: domain name (e.g. "blocks", "balancer", "hospital").
      index: problem index.
      steps: list of {"step": int?, "action": str, "args": [..]} dicts.
             'step' is ignored here; order is given by list position.

    Returns a compact summary suitable as a tool result:

      {
        "accepted": true/false,
        "length": int,
        "cost_value": float | null,
        "first_failure_at": int | null,
        "first_failed_action": str | null,
        "first_failure_reason": str | null,
        "first_failure_detail": str | null,
        "unsat_count": int,
        "redundant_indices": [int] | null,
        "advice_count": int,
        "advice_top_predicates": [[str, int], ...],
        "val_attempts": int,
        "val_warning": str | null
      }
    """
    domain_path, problem_path = _resolve_pddl_paths(domain, index)

    # Build VAL-compatible plan text from tool arguments.
    lines: list[str] = []
    for s in steps:
        action = str(s.get("action") or "").strip()
        if not action:
            continue
        args = s.get("args") or []
        arg_str = " ".join(str(a).strip() for a in args if str(a).strip())
        # "(move-vm vm1 srv-a srv-b)" etc.
        if arg_str:
            lines.append(f"({action} {arg_str})")
        else:
            lines.append(f"({action})")

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
