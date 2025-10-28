from __future__ import annotations
from purple_agent.base import PurpleAgent

class FilePurpleAgent(PurpleAgent):
    def __init__(self, path: str) -> None:
        self.path = path
    def generate_plan(self, *, problem_nl: str, actions_nl: str, formatting_instructions: str) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()
