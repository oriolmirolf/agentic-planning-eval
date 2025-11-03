# /Oriol-TFM/green_agent/a2a_server.py
from __future__ import annotations
import argparse
import json
import logging
from typing import Any, Optional

import uvicorn
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    InvalidParamsError,
    InternalError,
    TaskState,
    UnsupportedOperationError,
    Part,
    TextPart,
)
from a2a.utils import new_task, new_agent_text_message

from .config import EvalConfig
from .runner import evaluate_once
from .cli import _resolve_paths  # reuse your path resolver

log = logging.getLogger("pddl_green")


# ---- IO schema (minimal) -----------------------------------------------------

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]  # e.g. {"planner": "http://..."}
    config: dict[str, Any]            # e.g. {"example": "blocks", "index": 1, "check_redundancy": true}


# ---- Minimal green agent interface -------------------------------------------

class GreenAgent:
    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None: ...
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]: ...


class PDDLGreen(GreenAgent):
    """Wraps your evaluate_once() with A2A plumbing."""

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        parts = request.participants or {}
        if not parts:
            return False, "participants mapping is required (e.g., {'planner': 'http://host:port'})"
        if "planner" not in parts and len(parts) != 1:
            return False, "include a 'planner' role or provide exactly one participant"
        cfg = request.config or {}
        if "example" not in cfg:
            return False, "config.example is required (name under ./examples)"
        if "index" not in cfg:
            return False, "config.index is required (problem number)"
        try:
            int(cfg["index"])
        except Exception as e:
            return False, f"config.index must be an int: {e}"
        return True, "ok"

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        cfg_in = request.config or {}
        role = "planner" if "planner" in request.participants else list(request.participants.keys())[0]
        purple_url = str(request.participants[role])

        # Resolve domain/problem/prompt using your existing helper
        auto = _resolve_paths(cfg_in.get("example"), int(cfg_in.get("index")))
        if not auto["domain"] or not auto["problem"]:
            raise ValueError("Could not resolve domain/problem; check config.example and config.index")

        # Emit a short status update
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting PDDL evaluation: example='{cfg_in.get('example')}', index={cfg_in.get('index')}\n"
                f"Planner (A2A): {purple_url}"
            ),
        )

        cfg = EvalConfig(
            domain_path=auto["domain"],
            problem_path=auto["problem"],
            out_dir=cfg_in.get("out_dir", "out"),
            val_path=cfg_in.get("val_path"),
            val_flags=tuple(cfg_in.get("val_flags", ("-v", "-e"))),
            tolerance=float(cfg_in.get("tolerance", 0.001)),
            purple_kind="a2a",
            purple_url=purple_url,
            prompt_path=auto["prompt"],
            openai_model=None,
            check_redundancy=bool(cfg_in.get("check_redundancy", False)),
        )

        # Run your existing evaluator (synchronous)
        record = evaluate_once(cfg)

        # Provide a readable update and a JSON artifact
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Evaluation finished.\n"
                f"Valid: {record['valid']}\n"
                f"Length: {record['length']}\n"
                f"Cost/Value: {record['cost_value']}\n"
                f"Unsat preconds: {record['unsat_count']}\n"
                f"First failure: {record['first_failure_detail'] or '—'}"
            ),
        )

        await updater.add_artifact(
            parts=[
                Part(TextPart(text=json.dumps(record, indent=2))),
            ],
            name="pddl_evaluation_result",
        )


# ---- Executor wiring (thin shim copied from tutorial pattern) ----------------

class GreenExecutor(AgentExecutor):
    def __init__(self, agent: GreenAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = context.get_user_input()
        try:
            req = EvalRequest.model_validate_json(request_text)
            ok, msg = self.agent.validate_request(req)
            if not ok:
                raise InvalidParamsError(message=msg)
        except ValidationError as e:
            raise InvalidParamsError(message=e.json())

        msg = context.message
        if not msg:
            raise InvalidParamsError(message="Missing message.")

        task = new_task(msg)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.update_status(TaskState.working, new_agent_text_message("Assessment received."))

        try:
            await self.agent.run_eval(req, updater)
            await updater.complete()
        except InvalidParamsError as e:
            await updater.failed(new_agent_text_message(f"Invalid params: {e.message}"))
            raise
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Agent error: {e}"))
            raise InternalError(message=str(e))

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        raise UnsupportedOperationError()


def _agent_card(card_url: str) -> AgentCard:
    skill = AgentSkill(
        id="pddl_plan_benchmark",
        name="PDDL Plan Benchmark",
        description="Evaluates a purple planning agent by validating a plan with VAL for a selected example/index.",
        tags=["planning", "pddl", "validation"],
        examples=[json.dumps({
            "participants": {"planner": "http://127.0.0.1:9020/"},
            "config": {"example": "blocks", "index": 1, "check_redundancy": True}
        }, indent=2)],
    )
    return AgentCard(
        name="TFM-Green-PDDL",
        description="Green agent that evaluates PDDL plans produced by a purple agent (A2A).",
        url=card_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(description="Run the A2A Green server for the PDDL benchmark.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, help="External URL to publish in the agent card")
    args = parser.parse_args()

    agent = PDDLGreen()
    executor = GreenExecutor(agent)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=_agent_card(args.card_url or f"http://{args.host}:{args.port}/"),
                                  http_handler=handler).build()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
