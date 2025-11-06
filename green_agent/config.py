from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EvalConfig:
    domain_path: str
    problem_path: str
    out_dir: str = "out"

    # Optional override for where to write this run's artifacts.
    run_dir: Optional[str] = None

    val_path: Optional[str] = None
    # DEFAULT: "-v" only (no "-e").
    val_flags: Tuple[str, ...] = ("-v",)
    tolerance: float = 0.001

    # Purple agent selection
    purple_kind: str = "openai"                 # 'openai' | 'a2a' | 'react' | 'langchain-react'
    purple_url: Optional[str] = None            # a2a endpoint (if purple_kind='a2a')

    # Prompt sources
    prompt_path: Optional[str] = None
    prompt_text: Optional[str] = None

    # Model + provider config (for OpenAI-compatible providers)
    openai_model: Optional[str] = None
    llm_base_url: Optional[str] = None          # e.g., http://localhost:8000/v1
    llm_api_key: Optional[str] = None           # API key for that endpoint (if needed)

    check_redundancy: bool = False

    # Scoring
    optimal_cost: Optional[float] = None

    # Printing control
    print_card: bool = True
