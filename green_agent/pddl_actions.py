# green_agent/pddl_actions.py
from __future__ import annotations
import re
from typing import List, Tuple, Optional

ACTION_START_RE = re.compile(r":action\s+([^\s\)]+)", re.IGNORECASE)
PARAMS_RE = re.compile(r":parameters\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)

# New: extract PRE and EFF blobs inside each :action … end of block
PRECOND_RE = re.compile(r":precondition\s*(\([^\)]*(?:\)[^\)]*)*\))", re.IGNORECASE | re.DOTALL)
EFFECT_RE  = re.compile(r":effect\s*(\([^\)]*(?:\)[^\)]*)*\))", re.IGNORECASE | re.DOTALL)

def _strip_types(params_blob: str) -> str:
    tokens = params_blob.replace("\n", " ").split()
    return " ".join(tok for tok in tokens if tok.startswith("?"))

def extract_action_signatures(domain_text: str) -> List[str]:
    signatures: List[str] = []
    for m in ACTION_START_RE.finditer(domain_text):
        name = m.group(1)
        tail = domain_text[m.end():]
        pm = PARAMS_RE.search(tail)
        if pm:
            params = _strip_types(pm.group(1))
            if params.strip():
                signatures.append(f"({name} {params})")
            else:
                signatures.append(f"({name})")
        else:
            signatures.append(f"({name})")
    # dedupe
    seen = set(); out: List[str] = []
    for s in signatures:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def actions_from_domain(domain_path: str) -> str:
    try:
        with open(domain_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return "(No extra hints.)"
    sigs = extract_action_signatures(text)
    if not sigs:
        return "(No extra hints.)"
    return "\n".join(sigs) + "\n"

# ---------- NEW: simple semantics extraction ----------

def _clean_sexpr(s: str) -> str:
    # collapse whitespace but keep parentheses as-is
    s = re.sub(r"\s+", " ", s.strip())
    # drop redundant outer "(and ...)" for readability
    s = s[1:-1].strip() if s.startswith("(") and s.endswith(")") else s
    if s.lower().startswith("and "):
        s = s[4:].strip()
    return s

def extract_action_semantics(domain_text: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Returns list of (action_name, precond_str, effect_str).
    precond_str/effect_str are compact S-exprs (whitespace-collapsed), or None.
    """
    results: List[Tuple[str, Optional[str], Optional[str]]] = []
    # Walk each :action block
    for m in ACTION_START_RE.finditer(domain_text):
        name = m.group(1)
        # block runs until the next :action or end of text
        start = m.end()
        next_m = ACTION_START_RE.search(domain_text, pos=start)
        end   = next_m.start() if next_m else len(domain_text)
        block = domain_text[start:end]

        pre, eff = None, None
        p = PRECOND_RE.search(block)
        e = EFFECT_RE.search(block)
        if p:
            pre = _clean_sexpr(p.group(1))
        if e:
            eff = _clean_sexpr(e.group(1))
        results.append((name, pre, eff))
    return results

def semantics_from_domain(domain_path: str) -> str:
    """
    Pretty, low-risk semantics text to append to the prompt.
    Keeps predicates in PDDL S-expr style for precision (no lossy NL paraphrases).
    """
    try:
        with open(domain_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return ""
    rows = extract_action_semantics(text)
    if not rows:
        return ""
    out_lines: List[str] = []
    for name, pre, eff in rows:
        out_lines.append(f"{name}():")
        if pre:
            out_lines.append(f"  PRE: {pre}")
        if eff:
            out_lines.append(f"  EFF: {eff}")
    return "\n".join(out_lines) + "\n"
