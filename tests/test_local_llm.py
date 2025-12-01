# tests/test_local_llm.py
import os

import pytest

from purple_agent.llm_registry import LLMRegistry


# Skip these tests unless explicitly requested or if env var not set
# You can run them locally via: uv run pytest -m local_llm
@pytest.mark.local_llm
def test_local_llm_connection():
    """
    Verifies that we can actually talk to the local Qwen/Llama instance.
    This test requires a server running at the base_url.
    """
    # Adjust these defaults to match your local setup script
    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:5679/v1")
    model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-72B-Instruct")

    print(f"\nConnecting to Local LLM at {base_url} with model {model_name}...")

    registry = LLMRegistry(
        {
            "planner": {
                "provider": "openai_compat",
                "model": model_name,
                "base_url": base_url,
                "api_key": "dummy",
                "temperature": 0.1,
                "max_tokens": 50,
            }
        }
    )

    client = registry.get("planner")

    # Simple generation
    prompt = "What is the capital of France? Answer with just the city name."
    response = client.generate(prompt)

    print(f"LLM Response: {response}")

    assert response is not None
    assert len(response) > 0
    assert "Paris" in response or "paris" in response


@pytest.mark.local_llm
def test_local_llm_system_prompt():
    """Ensure system prompts are respected by the local model."""
    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:5679/v1")
    model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-72B-Instruct")

    registry = LLMRegistry(
        {
            "planner": {
                "provider": "openai_compat",
                "model": model_name,
                "base_url": base_url,
                "api_key": "dummy",
            }
        }
    )
    client = registry.get("planner")

    system = "You are a pirate. Always speak like a pirate."
    prompt = "Say hello."

    response = client.generate(prompt, system=system)
    print(f"Pirate Response: {response}")

    # Loose check for pirate lingo
    keywords = ["ahoy", "matey", "arr", "shiver"]
    assert any(k in response.lower() for k in keywords), (
        f"Response didn't sound like a pirate: {response}"
    )
