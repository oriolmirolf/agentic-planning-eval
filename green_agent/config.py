from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalConfig:
    domain_path: str
    problem_path: str
    out_dir: str = "out"
    val_path: Optional[str] = None
    val_flags: tuple[str, ...] = ("-v", "-e")   # verbose + error/advice
    tolerance: float = 0.001                    # VAL -t (epsilon) recommended
    purple_kind: str = "openai"                 # 'openai' | 'http' | 'file'
    purple_url: Optional[str] = None
    prompt_path: Optional[str] = None
    openai_model: Optional[str] = None
    temperature: float = 0.0
    attempts: int = 3
    check_redundancy: bool = False
