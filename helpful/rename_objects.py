#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_TOKEN_CHARCLASS = r"A-Za-z0-9_\-"
_PDDL_PROBLEM_RE = re.compile(r"^problem(\d+)\.pddl$", re.IGNORECASE)


@dataclass(frozen=True)
class DomainPaths:
    name: str
    root: Path
    domain_pddl: Path
    prompts_json: Path
    problems_dir: Path
    problem_pddls: list[Path]


def _require(p: Path, what: str) -> None:
    if not p.exists():
        raise SystemExit(f"Missing {what}: {p}")


def load_domain_paths(examples_root: Path, domain: str) -> DomainPaths:
    root = examples_root / domain
    domain_pddl = root / "domain.pddl"
    prompts_json = root / "prompts.json"
    problems_dir = root / "problems_pddl"

    _require(root, "domain folder")
    _require(domain_pddl, "domain.pddl")
    _require(prompts_json, "prompts.json")
    _require(problems_dir, "problems_pddl/")

    problem_pddls = sorted(problems_dir.glob("*.pddl"))
    if not problem_pddls:
        raise SystemExit(f"No .pddl files found in: {problems_dir}")

    return DomainPaths(
        name=domain,
        root=root,
        domain_pddl=domain_pddl,
        prompts_json=prompts_json,
        problems_dir=problems_dir,
        problem_pddls=problem_pddls,
    )


def _extract_objects_block(problem_text: str) -> str | None:
    m = re.search(r"(?is)\b:objects\b(.*?)(\b:init\b|\b:goal\b)", problem_text)
    if not m:
        return None
    return m.group(1)


def parse_objects_types_from_problem(problem_text: str) -> dict[str, str]:
    block = _extract_objects_block(problem_text)
    if not block:
        return {}

    cleaned = block.replace("(", " ").replace(")", " ")
    toks = [t for t in cleaned.split() if t]

    out: dict[str, str] = {}
    buf: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "-":
            if i + 1 < len(toks):
                typ = toks[i + 1]
                for obj in buf:
                    out[obj] = typ
                buf = []
                i += 2
                continue
        if t.startswith(":"):
            break
        buf.append(t)
        i += 1

    for obj in buf:
        out.setdefault(obj, "obj")
    return out


def objects_from_prompts_json(prompts: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for prob in prompts.get("problems", []):
        for obj in prob.get("objects", []):
            name = obj.get("name")
            kind = obj.get("kind") or "obj"
            if isinstance(name, str) and name:
                out[name] = str(kind)
    return out


def _abbr_from_type(type_name: str) -> str:
    letters = "".join(ch for ch in type_name.lower() if ch.isalpha())
    abbr = letters[:3] if letters else "obj"
    if not abbr or not abbr[0].isalpha():
        abbr = "obj"
    return abbr


def build_mapping(obj_to_type: dict[str, str], seed: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for old, typ in sorted(obj_to_type.items()):
        abbr = _abbr_from_type(typ)
        h = hashlib.sha1(f"{seed}:{old}".encode("utf-8")).hexdigest()

        for k in (6, 8, 10, 12):
            new = f"{abbr}_{h[:k]}"
            if new not in used:
                mapping[old] = new
                used.add(new)
                break
        else:
            raise RuntimeError(f"Could not create unique rename for {old}")
    return mapping


def compile_replacer(mapping: dict[str, str]) -> re.Pattern[str]:
    keys = sorted(mapping.keys(), key=len, reverse=True)
    union = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"(?<![{_TOKEN_CHARCLASS}])({union})(?![{_TOKEN_CHARCLASS}])")


def replace_text(text: str, pat: re.Pattern[str], mapping: dict[str, str]) -> str:
    return pat.sub(lambda m: mapping[m.group(1)], text)


def transform_json(obj: Any, pat: re.Pattern[str], mapping: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: transform_json(v, pat, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [transform_json(v, pat, mapping) for v in obj]
    if isinstance(obj, str):
        return replace_text(obj, pat, mapping)
    return obj


def _pddl_problem_indices(problems_dir: Path) -> list[int]:
    idxs: list[int] = []
    for p in problems_dir.iterdir():
        if not p.is_file():
            continue
        m = _PDDL_PROBLEM_RE.match(p.name)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(idxs))
    if not idxs:
        raise SystemExit(f"No problemN.pddl files found in: {problems_dir}")
    return idxs


def normalize_problem_ids(prompts: dict[str, Any], problems_dir: Path, *, prefix: str = "p") -> None:
    probs = prompts.get("problems", [])
    if not isinstance(probs, list) or not probs:
        raise SystemExit("prompts.json has no 'problems' list to normalize ids for.")

    idxs = _pddl_problem_indices(problems_dir)

    if len(idxs) != len(probs):
        raise SystemExit(
            f"Mismatch: prompts.json has {len(probs)} problems but {problems_dir} has {len(idxs)} problemN.pddl files.\n"
            f"Fix this first (don’t guess alignment). Found indices: {idxs}"
        )

    # Keep list order, but ensure the id matches problem{idx}.pddl deterministically.
    for i, idx in enumerate(idxs):
        entry = probs[i]
        if isinstance(entry, dict):
            entry["id"] = f"{prefix}{idx:02d}"

    prompts["problems"] = probs


def anonymize_domain_tree(domain_root: Path, seed: str) -> None:
    """
    Mutates files inside domain_root in-place:
      - prompts.json
      - domain.pddl
      - problems_pddl/problem*.pddl
    plus:
      - rewrites prompts["problems"][i]["id"] to p01..pNN based on existing problemN.pddl files
      - writes object_rename_map.json
    """
    prompts_path = domain_root / "prompts.json"
    domain_pddl_path = domain_root / "domain.pddl"
    problems_dir = domain_root / "problems_pddl"

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    obj_to_type: dict[str, str] = {}

    for p in sorted(problems_dir.glob("*.pddl")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        obj_to_type.update(parse_objects_types_from_problem(txt))

    if not obj_to_type:
        obj_to_type.update(objects_from_prompts_json(prompts))

    if not obj_to_type:
        raise SystemExit(f"No objects found to rename in: {domain_root}")

    mapping = build_mapping(obj_to_type, seed=seed)
    pat = compile_replacer(mapping)

    (domain_root / "object_rename_map.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    prompts_new = transform_json(prompts, pat, mapping)

    # ✅ CRITICAL: normalize problem ids so green_bench doesn't look for problem98.pddl
    normalize_problem_ids(prompts_new, problems_dir, prefix="p")

    prompts_path.write_text(json.dumps(prompts_new, indent=2, ensure_ascii=False), encoding="utf-8")

    domain_txt = domain_pddl_path.read_text(encoding="utf-8", errors="ignore")
    domain_pddl_path.write_text(replace_text(domain_txt, pat, mapping), encoding="utf-8")

    for p in sorted(problems_dir.glob("*.pddl")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        p.write_text(replace_text(txt, pat, mapping), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Back up examples/<domain> then swap in an anonymized version in-place."
    )
    ap.add_argument("--examples-root", default="examples", help="Root containing domain folders.")
    ap.add_argument("--domains", nargs="+", required=True, help="Domains to anonymize (e.g., gripper logistics).")
    ap.add_argument("--seed", default="42", help="Deterministic renaming seed.")
    ap.add_argument(
        "--backup-root",
        default=None,
        help="Where to store originals. Default: examples/_original_domains_<timestamp>/",
    )
    args = ap.parse_args()

    examples_root = Path(args.examples_root).resolve()
    _require(examples_root, "examples root directory")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = (
        Path(args.backup_root).resolve()
        if args.backup_root
        else (examples_root / f"_original_domains_{timestamp}")
    )
    backup_root.mkdir(parents=True, exist_ok=True)

    for d in args.domains:
        paths = load_domain_paths(examples_root, d)
        src = paths.root
        backup = backup_root / d
        tmp = examples_root / f".tmp_anonymized_{d}_{timestamp}"

        if backup.exists():
            raise SystemExit(f"Backup path already exists: {backup}")
        if tmp.exists():
            shutil.rmtree(tmp)

        shutil.move(str(src), str(backup))
        shutil.copytree(backup, tmp)

        anonymize_domain_tree(tmp, seed=str(args.seed))

        shutil.move(str(tmp), str(src))
        print(f"[OK] {d}: originals -> {backup}, anonymized -> {src}")

    print(f"\nDone. Originals stored at: {backup_root}")


if __name__ == "__main__":
    main()
