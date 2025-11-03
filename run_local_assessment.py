# run_local_assessment.py
import asyncio, json, httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import Message, Part, TextPart, DataPart, Role, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

GREEN_URL = "http://127.0.0.1:9009"
REQ = {
  "participants": {"planner": "http://127.0.0.1:9020"},
  "config": {"example": "blocks"}
}

def _merge_parts(parts):
    out = []
    for p in parts or []:
        if isinstance(p.root, TextPart):
            out.append(p.root.text.strip())
        elif isinstance(p.root, DataPart) and isinstance(p.root.data, str):
            out.append(p.root.data.strip())
    return "\n".join(x for x in out if x)

async def main():
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(kind="text", text=json.dumps(REQ)))],
        message_id=uuid4().hex,
    )
    async with httpx.AsyncClient(timeout=180) as hc:
        card = await A2ACardResolver(httpx_client=hc, base_url=GREEN_URL).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=hc, streaming=True)).create(card)

        final_json = None
        print("→ sending assessment_request")
        async for event in client.send_message(msg):
            if isinstance(event, Message):
                print(f"[final message]\n{_merge_parts(event.parts)}")

            else:
                task, update = event
                if isinstance(update, TaskStatusUpdateEvent):
                    state = update.status.state.value
                    text = _merge_parts(update.status.message.parts) if update.status.message else ""
                    if text:
                        print(f"[{state}] {text}")
                    else:
                        print(f"[{state}]")
                elif isinstance(update, TaskArtifactUpdateEvent):
                    text = _merge_parts(update.artifact.parts)
                    print(f"[artifact:{update.artifact.name}]")
                    # Try to pretty-print if it's JSON
                    try:
                        final_json = json.loads(text)
                        print(json.dumps(final_json, indent=2))
                    except Exception:
                        print(text)

        print("\n✓ done.")
        if final_json:
            print("\nSummary:")
            print(f"  valid: {final_json.get('valid')}")
            print(f"  length: {final_json.get('length')}")
            print(f"  value: {final_json.get('cost_value')}")
            print(f"  run_dir: {final_json.get('run_dir')}")

if __name__ == "__main__":
    asyncio.run(main())
