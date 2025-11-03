from __future__ import annotations
import argparse, os
import uvicorn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    TaskState,
    UnsupportedOperationError,
    Part,
    TextPart,
)
from a2a.utils import new_task, new_agent_text_message


class DummyPlanner(AgentExecutor):
    """Minimal purple agent that returns a static or dummy plan."""

    def __init__(self, static_plan: str | None):
        self.static_plan = static_plan  # already wrapped in ``` fences if provided

    def _make_plan(self, prompt: str) -> str:
        if self.static_plan:
            return self.static_plan
        # Simple dummy plan to test wiring; VAL will likely mark it invalid.
        return "```\n(noop)\n```"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        task = new_task(msg)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        prompt = context.get_user_input() or ""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Received prompt. Generating plan..."),
        )

        plan_block = self._make_plan(prompt)

        # Stream the plan
        await updater.update_status(TaskState.working, new_agent_text_message(plan_block))

        # Also attach as an artifact
        await updater.add_artifact(parts=[Part(root=TextPart(text=plan_block))], name="plan")

        await updater.complete()

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        raise UnsupportedOperationError()


def build_app(card_url: str):
    static_plan_text = None
    plan_path = os.getenv("STATIC_PLAN_PATH")
    if plan_path and os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8") as f:
            body = f.read().strip()
        static_plan_text = f"```\n{body}\n```"

    skill = AgentSkill(
        id="pddl_plan_producer",
        name="PDDL Plan Producer (Testing)",
        description="Returns a plan in a code block. Use STATIC_PLAN_PATH to serve a real .plan file.",
        tags=["planning", "pddl", "test"],
        examples=["Responds with:\n```\n(move a b)\n(stack a c)\n```"],
    )

    card = AgentCard(
        name="DummyPlanner",
        description="Minimal A2A purple agent that returns a static or dummy plan.",
        url=card_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    executor = DummyPlanner(static_plan=static_plan_text)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    return app


def main():
    ap = argparse.ArgumentParser(description="Run minimal A2A purple planner.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9020)
    ap.add_argument("--card-url", help="Public URL to publish in the Agent Card (use with tunnels)")
    args = ap.parse_args()

    url = args.card_url or f"http://{args.host}:{args.port}/"
    app = build_app(url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()