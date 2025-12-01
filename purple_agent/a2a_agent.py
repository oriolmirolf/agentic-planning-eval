# /Oriol-TFM/purple_agent/a2a_agent.py
from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

from .base import PurpleAgent


def _new_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def _merge_text(parts: list[Part]) -> str:
    chunks: list[str] = []
    for p in parts or []:
        if isinstance(p.root, TextPart) and isinstance(p.root.text, str):
            t = p.root.text.strip()
            if t:
                chunks.append(t)
        elif isinstance(p.root, DataPart) and isinstance(p.root.data, str):
            dt = p.root.data.strip()
            if dt:
                chunks.append(dt)
    return "\n".join(chunks)


class _AsyncA2AClient:
    """Collects any text the purple emits (status messages or artifacts)."""
    def __init__(self, url: str, timeout: float = 120.0) -> None:
        self.url = url
        self.timeout = timeout
        self.context_id: str | None = None

    async def send(self, text: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.url)
            card = await resolver.get_agent_card()
            client = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True)).create(card)

            plan_text: str = ""  # keep the latest non-empty text we see

            async for event in client.send_message(_new_message(text, self.context_id)):
                # Final message events:
                if isinstance(event, Message):
                    self.context_id = event.context_id
                    txt = _merge_text(event.parts)
                    if txt:
                        plan_text = txt
                    continue

                # Tuple events: (task, update/artifact)
                task, update = event
                self.context_id = task.context_id

                if isinstance(update, TaskStatusUpdateEvent):
                    msg = update.status.message
                    if msg:
                        txt = _merge_text(msg.parts)
                        if txt:
                            plan_text = txt

                elif isinstance(update, TaskArtifactUpdateEvent):
                    # Prefer artifacts named like "plan", but accept any text
                    art = update.artifact
                    txt = _merge_text(art.parts)
                    if txt:
                        plan_text = txt

            return (plan_text or "").strip()


def _run_async_in_thread(coro):
    """Run an async coroutine even if we're already inside an event loop."""
    try:
        asyncio.get_running_loop()
        result: dict[str, str] = {}
        def runner():
            result["v"] = asyncio.run(coro)
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        return result["v"]
    except RuntimeError:
        return asyncio.run(coro)


class A2APurpleAgent(PurpleAgent):
    def __init__(self, url: str, timeout: float = 180.0) -> None:
        self._client = _AsyncA2AClient(url, timeout=timeout)

    def generate_plan(self, *, problem_nl: str) -> str:
        return _run_async_in_thread(self._client.send(problem_nl))
