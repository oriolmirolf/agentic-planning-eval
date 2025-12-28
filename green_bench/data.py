from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    index: int
    prompt: str
    optimal_cost: float | None
    difficulty: str | None


@dataclass(frozen=True)
class DomainSpec:
    name: str
    domain_prompt: str
    problems: list[ProblemSpec]
    prompts_path: Path
    domain_pddl: Path
    problems_dir: Path


_ID_NUM_RE = re.compile(r"(\d+)")


def problem_index_from_id(pid: str) -> int:
    """
    Convert a problem id like 'p01' or '01' or 'problem01' into an int index.

    Assumes the PDDL file is named problems_pddl/problem{index}.pddl.
    """
    m = _ID_NUM_RE.findall(str(pid))
    if not m:
        raise ValueError(f"Could not parse numeric index from problem id: {pid!r}")
    return int(m[-1])


def load_domain(examples_dir: Path, domain_name: str) -> DomainSpec:
    dom_dir = examples_dir / domain_name
    prompts_path = dom_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing prompts.json: {prompts_path}")

    with prompts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"prompts.json must be an object: {prompts_path}")

    domain_prompt = str(data.get("domain_prompt") or "").strip()
    raw_problems = data.get("problems") or []
    if not isinstance(raw_problems, list):
        raise ValueError(f"'problems' must be a list in: {prompts_path}")

    problems: list[ProblemSpec] = []
    for p in raw_problems:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        idx = problem_index_from_id(pid)
        prompt = str(p.get("prompt") or "").strip()
        if not prompt:
            continue
        oc = p.get("optimal_cost", None)
        try:
            optimal_cost = float(oc) if oc is not None else None
        except Exception:
            optimal_cost = None
        difficulty = str(p.get("difficulty")).strip() if p.get("difficulty") else None

        problems.append(
            ProblemSpec(
                problem_id=pid,
                index=idx,
                prompt=prompt,
                optimal_cost=optimal_cost,
                difficulty=difficulty,
            )
        )

    domain_pddl = dom_dir / "domain.pddl"
    problems_dir = dom_dir / "problems_pddl"
    if not domain_pddl.exists():
        raise FileNotFoundError(
            f"Missing domain.pddl for domain '{domain_name}': {domain_pddl}"
        )
    if not problems_dir.exists():
        raise FileNotFoundError(
            f"Missing problems_pddl/ for domain '{domain_name}': {problems_dir}"
        )

    # Sanity-check: each referenced problem PDDL exists
    for pr in problems:
        pddl_path = problems_dir / f"problem{pr.index}.pddl"
        if not pddl_path.exists():
            raise FileNotFoundError(
                f"Missing PDDL for {domain_name}/{pr.problem_id}: {pddl_path}"
            )

    return DomainSpec(
        name=domain_name,
        domain_prompt=domain_prompt,
        problems=problems,
        prompts_path=prompts_path,
        domain_pddl=domain_pddl,
        problems_dir=problems_dir,
    )


def discover_domains(examples_dir: Path) -> list[str]:
    """
    Discover domains as direct children of examples_dir that contain prompts.json
    and domain.pddl.
    """
    out: list[str] = []
    if not examples_dir.exists():
        return out
    for p in sorted(examples_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "prompts.json").exists() and (p / "domain.pddl").exists():
            out.append(p.name)
    return out
