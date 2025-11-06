# /Oriol-TFM/purple_agent/llm_registry.py
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Optional, Dict, Any

# OpenAI client
from openai import OpenAI  # Official OpenAI SDK (Responses API)  # See docs: platform.openai.com
# Anthropic client
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None  # optional
# Google GenAI SDK
try:
    from google import genai
except Exception:
    genai = None  # optional

class LLMError(RuntimeError): pass

@dataclass
class LLMConfig:
    provider: str                # "openai" | "openai_compat" | "anthropic" | "google"
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2048

class LLMClient:
    """Minimal interface: single-shot text generation."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

# -------------------- OpenAI + OpenAI-compatible (vLLM, etc.) --------------------

class OpenAICompatClient(LLMClient):
    """
    Uses OpenAI Responses API when available; otherwise auto-falls back to Chat Completions.
    This is important for local OpenAI-compatible servers (e.g., vLLM) that don't implement Responses yet.
    """
    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        kwargs = {}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        self.client = OpenAI(**kwargs)  # type: ignore

    def _try_responses(self, system: Optional[str], prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        try:
            # Responses API: single-string input is enough; we include system prefix inline if provided
            if system:
                input_text = f"{system.strip()}\n\n{prompt.strip()}"
            else:
                input_text = prompt
            resp = self.client.responses.create(
                model=self.cfg.model,
                input=input_text,
                temperature=temperature
            )
            # Normalized extractor
            txt = getattr(resp, "output_text", None)
            if txt:
                return txt
            # Fallback extraction
            try:
                d = resp.to_dict()
            except Exception:
                d = getattr(resp, "__dict__", {}) or {}
            chunks = []
            for item in d.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        t = c.get("text")
                        if isinstance(t, dict) and t.get("value"):
                            chunks.append(t["value"])
                        elif isinstance(t, str):
                            chunks.append(t)
            return "".join(chunks) if chunks else None
        except Exception as e:
            # Heuristic: 404/Not found or "responses not supported"
            msg = str(e).lower()
            if ("responses" in msg and ("not" in msg and ("support" in msg or "implement" in msg))) or "404" in msg or "unknown path" in msg:
                return None
            # Azure/OpenAI errors might suggest chat.completions; if so, fallback
            if "chat.completions.create" in msg or "use chat.completions" in msg:
                return None
            # Otherwise re-raise
            raise

    def _chat_completions(self, system: Optional[str], prompt: str, temperature: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        t = float(self.cfg.temperature if temperature is None else temperature)
        mt = int(self.cfg.max_tokens if max_tokens is None else max_tokens)
        # Try Responses API first
        out = self._try_responses(system, prompt, t, mt)
        if out is not None:
            return out.strip()
        # Fallback to Chat Completions (works well for local OpenAI-compatible servers)
        return self._chat_completions(system, prompt, t, mt).strip()

# -------------------- Anthropic --------------------

class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        if Anthropic is None:
            raise LLMError("anthropic package not installed. pip install anthropic")
        super().__init__(cfg)
        self.client = Anthropic(api_key=cfg.api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        t = float(self.cfg.temperature if temperature is None else temperature)
        mt = int(self.cfg.max_tokens if max_tokens is None else max_tokens)
        # Some SDK versions accept 'system='; to be maximally compatible, inline system text at the top.
        combined = f"{system.strip()}\n\n{prompt.strip()}" if system else prompt
        msg = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=mt,
            temperature=t,
            messages=[{"role": "user", "content": combined}],
        )
        # Extract text parts
        chunks = []
        for part in getattr(msg, "content", []) or []:
            if getattr(part, "type", None) == "text":
                chunks.append(getattr(part, "text", "") or "")
        return "".join(chunks).strip()

# -------------------- Google GenAI (Gemini) --------------------

class GoogleGenAIClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        if genai is None:
            raise LLMError("google-genai package not installed. pip install google-genai")
        super().__init__(cfg)
        # The SDK reads GEMINI_API_KEY / GOOGLE_API_KEY env vars too; we pass explicitly if provided
        self.client = genai.Client(api_key=cfg.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        t = float(self.cfg.temperature if temperature is None else temperature)
        mt = int(self.cfg.max_tokens if max_tokens is None else max_tokens)
        combined = f"{system.strip()}\n\n{prompt.strip()}" if system else prompt
        resp = self.client.models.generate_content(
            model=self.cfg.model,
            contents=combined,
        )
        # Most SDK versions provide .text
        return getattr(resp, "text", "") or ""

# -------------------- Registry --------------------

class LLMRegistry:
    """
    Build LLM clients from simple dict specs, e.g.:
      {"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."}
      {"provider": "openai_compat", "model": "Qwen2.5-7B-Instruct", "base_url": "http://localhost:5678/v1", "api_key": "dummy"}
      {"provider": "anthropic", "model": "claude-3-7-sonnet", "api_key": "..."}
      {"provider": "google", "model": "gemini-2.5-pro", "api_key": "..."}
    """
    def __init__(self, role_to_cfg: Dict[str, Dict[str, Any]]):
        self._clients: Dict[str, LLMClient] = {}
        for role, raw in role_to_cfg.items():
            cfg = LLMConfig(
                provider=raw.get("provider", "openai").lower(),
                model=raw["model"],
                base_url=raw.get("base_url"),
                api_key=raw.get("api_key"),
                temperature=float(raw.get("temperature", 0.2)),
                max_tokens=int(raw.get("max_tokens", 2048)),
            )
            if cfg.provider in ("openai", "openai_compat"):
                self._clients[role] = OpenAICompatClient(cfg)
            elif cfg.provider == "anthropic":
                self._clients[role] = AnthropicClient(cfg)
            elif cfg.provider == "google":
                self._clients[role] = GoogleGenAIClient(cfg)
            else:
                raise LLMError(f"Unknown provider: {cfg.provider}")

    def get(self, role: str) -> LLMClient:
        if role not in self._clients:
            raise LLMError(f"Missing LLM role: {role}")
        return self._clients[role]
