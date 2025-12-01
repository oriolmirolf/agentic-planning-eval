# /Oriol-TFM/experiments/models.py
# One place to list *all* models you want to try.
# - For hosted APIs, prefer api_key_env so you keep secrets out of the file.
# - For local vLLM, use base_url and a dummy api_key.
# - You can safely add/remove entries; the experiment runner will iterate them.

MODELS = {
    # ======================
    # OpenAI (Responses API)
    # ======================
    # GPT-5 family (flagship + smaller variants)
    # "gpt5": {
    #     "provider": "openai",
    #     "model": "gpt-5",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "temperature": 0.2,
    # },
    # "gpt5-mini": {
    #     "provider": "openai",
    #     "model": "gpt-5-mini",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "temperature": 0.2,
    # },
    # "gpt5-nano": {
    #     "provider": "openai",
    #     "model": "gpt-5-nano",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "temperature": 0.2,
    # },
    # # Useful legacy/baseline OpenAI models (still widely referenced)
    # "gpt41": {
    #     "provider": "openai",
    #     "model": "gpt-4.1",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "temperature": 0.2,
    # },
    # "gpt4o": {
    #     "provider": "openai",
    #     "model": "gpt-4o",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "temperature": 0.2,
    # },
    # # ======================
    # # Anthropic (Claude API)
    # # ======================
    # # Use current snapshot IDs published by Anthropic docs.
    # "claude-sonnet-4.5": {
    #     "provider": "anthropic",
    #     "model": "claude-sonnet-4-5-20250929",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    #     "temperature": 0.2,
    # },
    # "claude-opus-4.1": {
    #     "provider": "anthropic",
    #     "model": "claude-opus-4-1-20250805",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    #     "temperature": 0.2,
    # },
    # "claude-haiku-4.5": {
    #     "provider": "anthropic",
    #     "model": "claude-haiku-4-5-20251001",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    #     "temperature": 0.2,
    # },
    # # ======================
    # # Google (Gemini API)
    # # ======================
    # "gemini25-pro": {
    #     "provider": "google",
    #     "model": "gemini-2.5-pro",
    #     "api_key_env": "GEMINI_API_KEY",
    #     "temperature": 0.2,
    # },
    # "gemini25-flash": {
    #     "provider": "google",
    #     "model": "gemini-2.5-flash",
    #     "api_key_env": "GEMINI_API_KEY",
    #     "temperature": 0.2,
    # },
    # =======================================
    # Local OpenAI-compatible (vLLM via tunnel)
    # =======================================
    # Qwen2.5 7B Instruct served behind your launch_and_tunnel.sh (localhost:5678/v1)
    "qwen7b-local": {
        "provider": "openai_compat",  # use OpenAI-compatible client with base_url
        "model": "Qwen2.5-72B-Instruct",
        "base_url": "http://localhost:5679/v1",
        "api_key": "dummy",
        "temperature": 0.2,
    },
}
