from __future__ import annotations

"""Runtime configuration helpers for TunaBrain.

This module centralizes environment-driven settings so that LLM provider and model
selection can be controlled without code changes. Defaults favor OpenAI since the
project ships with prompts tuned for that provider.
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache


logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings derived from environment variables."""

    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    debug_enabled: bool


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean flag controlled by an environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on", "y"}


@lru_cache
def get_settings() -> Settings:
    """Load settings from the environment with sensible defaults."""

    settings = Settings(
        llm_provider=os.getenv("TUNABRAIN_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("TUNABRAIN_LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        debug_enabled=_env_flag("TUNABRAIN_DEBUG", False),
    )
    logger.info(
        "Loaded settings: provider=%s model=%s debug=%s", 
        settings.llm_provider,
        settings.llm_model,
        settings.debug_enabled,
    )
    return settings


def is_debug_enabled(request_debug: bool = False) -> bool:
    """Resolve whether debug logging should be enabled for a request.

    Debug logging can be toggled either via the request payload or by setting the
    ``TUNABRAIN_DEBUG`` environment variable. The environment variable ensures
    logging is still available when request parsing fails before the payload can
    be inspected.
    """

    return request_debug or get_settings().debug_enabled
