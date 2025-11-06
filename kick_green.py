import asyncio, httpx, json
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import Role, Message, Part, TextPart

async def main():
    base_url = "http://127.0.0.1:9009/"  # green server
    # This is the *text* your green expects (EvalRequest as JSON string):
    eval_request = {
        "participants": {"planner": "http://127.0.0.1:9020/"},
        "config": {"domain": "blocks", "index": 1, "check_redundancy": True}
    }
    text = json.dumps(eval_request)

    async with httpx.AsyncClient(timeout=120) as hc:
        card = await A2ACardResolver(httpx_client=hc, base_url=base_url).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=hc, streaming=True)).create(card)
        msg = Message(kind="message", role=Role.user, parts=[Part(TextPart(text=text))])

        async for event in client.send_message(msg):
            print(event)  # you'll see TaskStatus updates + the final Message/artifact

asyncio.run(main())
