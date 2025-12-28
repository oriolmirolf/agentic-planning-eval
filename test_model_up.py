from openai import OpenAI

BASE_URL = "http://localhost:5679/v1"
MODEL = "Qwen3-VL-30B-A3B-Thinking"

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

# 1) Try listing models
try:
    ms = client.models.list()
    ids = [m.id for m in ms.data]
    print("models.list OK. First IDs:", ids[:5])
except Exception as e:
    print("models.list failed:", e)

# 2) Try chat completion (most compatible)
try:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
        max_tokens=10,
    )
    print("chat.completions OK:", r.choices[0].message.content)
except Exception as e:
    print("chat.completions failed:", e)

# 3) Try Responses API (optional)
try:
    r = client.responses.create(
        model=MODEL,
        input="Reply with exactly: OK",
        max_output_tokens=10,
    )
    print("responses OK:", getattr(r, "output_text", None))
except Exception as e:
    print("responses failed:", e)
