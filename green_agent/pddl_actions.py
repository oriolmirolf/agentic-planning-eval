from __future__ import annotations
import re
from typing import List

ACTION_START_RE = re.compile(r":action\s+([^\s\)]+)", re.IGNORECASE)
PARAMS_RE = re.compile(r":parameters\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)

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
