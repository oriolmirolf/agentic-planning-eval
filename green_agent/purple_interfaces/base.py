from __future__ import annotations
from typing import Protocol

class PurpleAgent(Protocol):
    def generate_plan(self, *, problem_nl: str, actions_nl: str, formatting_instructions: str) -> str: ...
