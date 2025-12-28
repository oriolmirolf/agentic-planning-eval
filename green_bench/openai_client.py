from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _normalize_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    # OpenAI Python SDK expects base_url like http://host:port/v1
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _extract_text_from_responses(resp: Any) -> str:
    """
    Best-effort extraction across OpenAI SDK versions and OpenAI-compatible servers.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt

    # pydantic-ish objects
    try:
        d = resp.to_dict() if hasattr(resp, "to_dict") else resp.__dict__
    except Exception:
        d = getattr(resp, "__dict__", {}) or {}

    # Responses API
    out_chunks: list[str] = []
    for item in d.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                t = c.get("text")
                if isinstance(t, dict) and isinstance(t.get("value"), str):
                    out_chunks.append(t["value"])
                elif isinstance(t, str):
                    out_chunks.append(t)
    if out_chunks:
        return "".join(out_chunks)

    # Chat Completions
    choices = d.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content

    # Older completion shapes (very rare here)
    if isinstance(d.get("text"), str):
        return d["text"]

    return ""


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    model: str
    temperature: float
    max_tokens: int


class OpenAICompatClient:
    """
    OpenAI SDK wrapper that works against OpenAI-compatible local inference servers.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        request_timeout_s: float = 180.0,
    ) -> None:
        from openai import OpenAI  # type: ignore

        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.request_timeout_s = float(request_timeout_s)

        # OpenAI SDK uses httpx under the hood; timeout can be passed via client options
        # However, depending on SDK version, the kwarg name may differ.
        try:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.request_timeout_s,
            )
        except TypeError:
            # older SDK
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, req: LLMRequest) -> str:
        """
        Attempt Responses API first, then fallback to Chat Completions.
        """
        prompt = req.prompt
        model = req.model
        temperature = float(req.temperature)
        max_tokens = int(req.max_tokens)

        # --- Try Responses API ---
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            try:
                resp = self._client.responses.create(**kwargs)
            except Exception as e:
                # Some servers don't accept temperature or max_output_tokens
                msg = str(e)
                if (
                    "Unsupported parameter" in msg
                    or "unexpected keyword" in msg
                    or "max_output_tokens" in msg
                ):
                    kwargs.pop("temperature", None)
                    kwargs.pop("max_output_tokens", None)
                    # Some servers accept max_tokens instead
                    kwargs["max_tokens"] = max_tokens
                    resp = self._client.responses.create(**kwargs)
                else:
                    raise
            return _extract_text_from_responses(resp).strip()
        except Exception:
            pass

        # --- Fallback: Chat Completions API ---
        kwargs2: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp2 = self._client.chat.completions.create(**kwargs2)
        except Exception as e:
            msg = str(e)
            if (
                "Unsupported parameter" in msg
                or "unexpected keyword" in msg
                or "temperature" in msg
            ):
                kwargs2.pop("temperature", None)
                resp2 = self._client.chat.completions.create(**kwargs2)
            else:
                raise
        return _extract_text_from_responses(resp2).strip()


def try_list_models(
    *,
    base_url: str,
    api_key: str = "EMPTY",
    request_timeout_s: float = 30.0,
) -> list[str] | None:
    """Best-effort model listing for OpenAI-compatible servers.

    This is useful in "manual/tunnel" mode where the caller may not know the
    served model id.

    Returns:
      - list[str] of model ids if available
      - None if the endpoint does not support listing or the call fails
    """
    from openai import OpenAI  # type: ignore

    base_url_n = _normalize_base_url(base_url)
    try:
        try:
            client = OpenAI(
                base_url=base_url_n, api_key=api_key, timeout=float(request_timeout_s)
            )
        except TypeError:
            client = OpenAI(base_url=base_url_n, api_key=api_key)
        res = client.models.list()
    except Exception:
        return None

    data = getattr(res, "data", None)
    if not data:
        # Some servers may return a raw dict-like payload
        try:
            d = (
                res.to_dict()
                if hasattr(res, "to_dict")
                else getattr(res, "__dict__", {})
            )
            data = d.get("data")
        except Exception:
            data = None

    ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            mid = None
            if isinstance(item, dict):
                mid = item.get("id")
            else:
                mid = getattr(item, "id", None)
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
    return ids or None
