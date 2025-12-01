from __future__ import annotations

from typing import Protocol

# we should substitute all this with A2A later on

class PurpleAgent(Protocol):
    def generate_plan(self, *, problem_nl: str) -> str: ...
