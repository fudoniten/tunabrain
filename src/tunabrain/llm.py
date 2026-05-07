from __future__ import annotations

"""Utilities for constructing LangChain chat models with project defaults."""

import logging
from enum import Enum

from langchain.chat_models import init_chat_model

from tunabrain.config import get_settings


logger = logging.getLogger(__name__)


class LLMTask(Enum):
    """Task categories for LLM routing."""
    DEFAULT = "default"
    SHOW_TAGGING = "show_tagging"
    EPISODE_FLAGGING = "episode_flagging"
    SCHEDULING = "scheduling"
    CATEGORIZATION = "categorization"
    BUMPERS = "bumpers"


def get_chat_model(task: LLMTask = LLMTask.DEFAULT):
    """Return a configured chat model instance based on task and environment settings.
    
    Supports task-specific model selection via environment variables:
    - TUNABRAIN_SHOW_LLM_MODEL: for show tagging
    - TUNABRAIN_EPISODE_LLM_MODEL: for episode special flags
    - TUNABRAIN_SCHEDULE_LLM_MODEL: for scheduling
    
    Falls back to TUNABRAIN_LLM_MODEL if task-specific override not set.
    """

    settings = get_settings()
    init_kwargs = {}

    # Determine which model to use based on task
    model_to_use = settings.llm_model  # fallback

    if task == LLMTask.SHOW_TAGGING and settings.show_llm_model:
        model_to_use = settings.show_llm_model
    elif task == LLMTask.EPISODE_FLAGGING and settings.episode_llm_model:
        model_to_use = settings.episode_llm_model
    elif task == LLMTask.SCHEDULING and settings.schedule_llm_model:
        model_to_use = settings.schedule_llm_model

    # Explicitly pass the API key when available so deployments can rely on the
    # OpenAI environment variable or overrides without extra wiring.
    if settings.llm_provider == "openai" and settings.openai_api_key:
        init_kwargs["api_key"] = settings.openai_api_key

    logger.info(
        "Initializing chat model for task=%s: provider=%s model=%s",
        task.value, settings.llm_provider, model_to_use
    )
    return init_chat_model(
        model=model_to_use,
        model_provider=settings.llm_provider,
        **init_kwargs,
    )
