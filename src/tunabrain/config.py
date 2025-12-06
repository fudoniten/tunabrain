from __future__ import annotations

"""Runtime configuration helpers for TunaBrain.

This module centralizes environment-driven settings so that LLM provider and model
selection can be controlled without code changes. Defaults favor OpenAI since the
project ships with prompts tuned for that provider.
"""

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass
class Settings:
    """Application settings derived from environment variables."""

    llm_provider: str
    llm_model: str
    openai_api_key: str | None


@lru_cache
def get_settings() -> Settings:
    """Load settings from the environment with sensible defaults."""

    return Settings(
        llm_provider=os.getenv("TUNABRAIN_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("TUNABRAIN_LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
