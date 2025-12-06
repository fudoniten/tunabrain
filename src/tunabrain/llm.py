from __future__ import annotations

"""Utilities for constructing LangChain chat models with project defaults."""

from langchain.chat_models import init_chat_model

from tunabrain.config import get_settings


def get_chat_model():
    """Return a configured chat model instance based on environment settings."""

    settings = get_settings()
    init_kwargs = {}

    # Explicitly pass the API key when available so deployments can rely on the
    # OpenAI environment variable or overrides without extra wiring.
    if settings.llm_provider == "openai" and settings.openai_api_key:
        init_kwargs["api_key"] = settings.openai_api_key

    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        **init_kwargs,
    )
