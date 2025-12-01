from __future__ import annotations

import os

from purple_agent.base import PurpleAgent


def _extract_text_from_responses(resp) -> str:
    # (unchanged extractor)
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


class OpenAIPurpleAgent(PurpleAgent):
    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(temperature)

        # Prefer OPENAI_BASE_URL/OPENAI_API_KEY envs if not provided
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")

        from openai import OpenAI  # type: ignore
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)

    def _responses_create(self, prompt: str) -> str:
        base = {"model": self.model, "input": prompt}
        try:
            resp = self._client.responses.create(**{**base, "temperature": self.temperature})
        except Exception as e:
            if "Unsupported parameter" in str(e) and "temperature" in str(e):
                resp = self._client.responses.create(**base)
            else:
                raise
        return _extract_text_from_responses(resp)

    def generate_plan(self, *, problem_nl: str) -> str:
        prompt = problem_nl
        return self._responses_create(prompt)
