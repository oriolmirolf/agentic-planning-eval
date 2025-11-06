# /Oriol-TFM/purple_agent/strategy_agent.py
from __future__ import annotations
from typing import Optional, Dict, Any
from .base import PurpleAgent
from .llm_registry import LLMRegistry
from . import strategies as S

class StrategyPurpleAgent(PurpleAgent):
    """
    One-shot orchestrator: selects a technique and runs its internal pipeline
    with one or more LLM roles. Returns the final plan text.
    """
    def __init__(self, *, strategy_name: str, roles: Dict[str, Dict[str, Any]], settings: Optional[Dict[str, Any]] = None):
        self.strategy_name = (strategy_name or "base").lower()
        self.registry = LLMRegistry(roles or {})
        self.settings = settings or {}

    def generate_plan(self, *, problem_nl: str) -> str:
        sn = self.strategy_name
        st = self.settings

        if sn == "base":
            return S.run_direct(self.registry.get("planner"), problem_nl)

        if sn == "cot":
            return S.run_cot(self.registry.get("planner"), problem_nl)

        if sn == "cot_sc":
            return S.run_cot_sc(
                self.registry.get("planner"),
                self.registry.get("judge"),
                problem_nl,
                samples=int(st.get("samples", 6)),
                temperature=float(st.get("temperature", 0.7))
            )

        if sn == "ltm":
            return S.run_ltm(self.registry.get("planner"), problem_nl)

        if sn == "react":
            return S.run_react(self.registry.get("controller"), problem_nl,
                               max_steps=int(st.get("max_steps", 16)))

        if sn == "debate":
            return S.run_debate(self.registry.get("proponent_a"),
                                self.registry.get("proponent_b"),
                                self.registry.get("judge"), problem_nl)

        if sn == "verifier":
            return S.run_verifier(self.registry.get("planner"),
                                  self.registry.get("verifier"), problem_nl)

        if sn == "tot":
            return S.run_tot(self.registry.get("planner"),
                             self.registry.get("judge"), problem_nl,
                             depth=int(st.get("depth", 4)),
                             branch=int(st.get("branch", 3)),
                             beam=int(st.get("beam", 3)))

        if sn == "ensemble":
            judge = self.registry.get("judge") if "judge" in self.registry._clients else None
            return S.run_ensemble(self.registry.get("planner"),
                                  self.registry.get("synth"), judge, problem_nl,
                                  n=int(st.get("n", 6)), temp=float(st.get("temp", 0.8)))

        # default
        return S.run_direct(self.registry.get("planner"), problem_nl)
