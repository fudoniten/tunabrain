from __future__ import annotations

"""Utilities for constructing LangChain chat models with project defaults."""

import logging

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from tunabrain.config import get_settings


logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_model():
    """Return a configured chat model instance based on environment settings."""

    settings = get_settings()

    logger.info(
        "Initializing chat model: provider=%s model=%s", settings.llm_provider, settings.llm_model
    )

    if settings.llm_provider == "openrouter":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )

    init_kwargs = {}
    if settings.llm_provider == "openai" and settings.openai_api_key:
        init_kwargs["api_key"] = settings.openai_api_key

    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        **init_kwargs,
    )
