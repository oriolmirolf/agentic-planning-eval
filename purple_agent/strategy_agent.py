# /Oriol-TFM/purple_agent/strategy_agent.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os

from . import strategies as S

# Minimal interface expected by strategies.py
class LLMClient:
    def generate(self, prompt: str, temperature: float = 0.0) -> str: ...

class OpenAICompatClient(LLMClient):
    """
    Thin client around OpenAI's Responses API with optional base_url.
    Works for:
      - provider: "openai"        (uses OPENAI_API_KEY or cfg["api_key"])
      - provider: "openai_compat" (uses cfg["base_url"] + dummy api_key)
    Required cfg keys:
      - "model"
    Optional:
      - "api_key" or "api_key_env"
      - "base_url"
    """
    def __init__(self, *, model: str, api_key: Optional[str], base_url: Optional[str]) -> None:
        from openai import OpenAI  # lazy import
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # Prefer Responses API; fall back cleanly if temperature is not supported
        base = {"model": self._model, "input": prompt}
        try:
            resp = self._client.responses.create(**{**base, "temperature": float(temperature)})
        except Exception as e:
            if "Unsupported parameter" in str(e) and "temperature" in str(e):
                resp = self._client.responses.create(**base)
            else:
                raise
        # robust text extraction (like your openai_agent.py)
        txt = getattr(resp, "output_text", None)
        if txt:
            return txt
        try:
            d = resp.to_dict() if hasattr(resp, "to_dict") else resp.__dict__
        except Exception:
            d = getattr(resp, "__dict__", {}) or {}
        out_chunks = []
        for item in d.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    t = c.get("text")
                    if isinstance(t, dict) and t.get("value"):
                        out_chunks.append(t["value"])
                    elif isinstance(t, str):
                        out_chunks.append(t)
        if out_chunks:
            return "".join(out_chunks)
        if "choices" in d and d["choices"]:
            msg = d["choices"][0].get("message", {})
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
        return ""

def _build_llm_client(cfg: Dict[str, Any]) -> LLMClient:
    provider = (cfg.get("provider") or "").strip().lower()
    model = cfg.get("model")
    if not model:
        raise ValueError("LLM config missing 'model'")

    # Resolve API key from env if needed
    api_key = cfg.get("api_key")
    if not api_key and cfg.get("api_key_env"):
        api_key = os.getenv(cfg["api_key_env"])

    if provider in ("openai", "openai_compat"):
        return OpenAICompatClient(
            model=model,
            api_key=api_key,
            base_url=cfg.get("base_url"),
        )

    # You can extend here for "anthropic", "google", etc., when needed.
    raise ValueError(f"Unsupported provider '{provider}'. Supported: openai, openai_compat")

class StrategyPurpleAgent:
    """
    Wraps strategies.run_strategy(..) with planner (+ optional judge for cot_sc).
    roles: {"planner": <llm_cfg>, "judge": <llm_cfg>?}
    """
    def __init__(self, *, strategy_name: str, roles: Dict[str, Dict[str, Any]], settings: Optional[Dict[str, Any]] = None):
        self.strategy_name = strategy_name
        self.settings = settings or {}
        if "planner" not in roles:
            raise ValueError("roles must include 'planner'")
        self.planner = _build_llm_client(roles["planner"])
        self.judge = _build_llm_client(roles["judge"]) if roles.get("judge") else None

    def generate_plan(self, *, problem_nl: str) -> str:
        return S.run_strategy(
            self.strategy_name,
            planner=self.planner,
            judge=self.judge,
            problem_nl=problem_nl,
            settings=self.settings,
        )
