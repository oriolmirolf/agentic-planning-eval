from __future__ import annotations
import os
from typing import Optional
from purple_agent.base import PurpleAgent

def _extract_text_from_responses(resp) -> str:
    # 1) The official property on Responses API objects
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt

    # 2) Fallback: walk the 'output' array (SDK 2.x)
    try:
        d = resp.to_dict() if hasattr(resp, "to_dict") else resp.__dict__
    except Exception:
        d = getattr(resp, "__dict__", {}) or {}

    out_chunks = []
    for item in d.get("output", []):
        # Each item has 'content' — look for type=='output_text'
        for c in item.get("content", []):
            t = c.get("type")
            if t == "output_text":
                text_obj = c.get("text")
                if isinstance(text_obj, dict):
                    val = text_obj.get("value")
                    if val:
                        out_chunks.append(val)
                elif isinstance(text_obj, str):
                    out_chunks.append(text_obj)
    if out_chunks:
        return "".join(out_chunks)

    # 3) Last resort: Chat Completions-like shape (choices[0].message.content)
    if "choices" in d and d["choices"]:
        msg = d["choices"][0].get("message", {})
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                return c

    return ""  # nothing found


class OpenAIPurpleAgent(PurpleAgent):
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(temperature)

        # Prefer the modern Responses API (OpenAI SDK 2.x)
        from openai import OpenAI  # type: ignore
        self._client = OpenAI()
        
    def _responses_create(self, prompt: str) -> str:
        # Official pattern: pass a simple string via 'input', then read 'output_text'
        # https://platform.openai.com/docs/guides/text  and  https://platform.openai.com/docs/api-reference/responses
        base = {"model": self.model, "input": prompt}

        try:
            # First try WITH temperature (some models accept it)…
            resp = self._client.responses.create(**{**base, "temperature": self.temperature})
        except Exception as e:
            # If the model rejects 'temperature', retry WITHOUT it
            msg = str(e)
            if "Unsupported parameter" in msg and "temperature" in msg:
                resp = self._client.responses.create(**base)
            else:
                raise

        return _extract_text_from_responses(resp)

    def generate_plan(self, *, problem_nl: str) -> str:
        prompt = problem_nl
        return self._responses_create(prompt)
