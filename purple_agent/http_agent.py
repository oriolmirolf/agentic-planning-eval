from __future__ import annotations
import json, urllib.request
from purple_agent.base import PurpleAgent

class HTTPPurpleAgent(PurpleAgent):
    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout

    # Match the PurpleAgent protocol used by runner.py
    def generate_plan(self, *, problem_nl: str) -> str:
        payload = {"problem": problem_nl}
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("plan", "")
