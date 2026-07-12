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
    openrouter_api_key: str | None
    debug_enabled: bool
    
    # Task-specific model overrides (optional; fall back to llm_model if not set)
    show_llm_model: str | None = None        # Full tagging for shows
    episode_llm_model: str | None = None     # Lightweight episode special flags
    schedule_llm_model: str | None = None    # Schedule building
    review_llm_model: str | None = None      # Schedule review / critique loop
    bumpers_llm_model: str | None = None     # Bumper prompt generation

    # Max number of (schedulable) shows to enumerate in scheduling prompts. The
    # catalog shape is always summarized; this caps the per-show detail list so a
    # large library stays within a sensible prompt size. Long-context models
    # (e.g. Claude Opus) can afford a high value.
    schedule_max_shows: int = 300

    # --- Grout long-form enrichment (STT + keyframes) ---
    # Both STT backends exist in the cluster with different performance profiles
    # (see the enrich spec §4). The orchestrator is pluggable: pick a default
    # here, override per-request via EnrichLongFormOptions.stt_backend.
    stt_whisper_url: str = "http://whisper-http.wyoming.svc.cluster.local:10301"
    stt_subgen_url: str = "http://subgen.arr.svc.cluster.local:9000"
    stt_default_backend: str = "auto"
    # whisper-http only has "turbo" registered in the current deployment; do NOT
    # default to large-v3 (the server rejects it). Overridable for other deploys.
    stt_whisper_model: str = "turbo"
    scratch_dir: str = "/tmp/tunabrain-scratch"
    enrich_long_timeout: int = 900

    # --- Grounding / context resolution ---
    # When True, media with no supplied grounding falls back to a Wikipedia
    # auto-search on the title. Most of Grout's free-form media has no notable
    # Wikipedia page (anything with an IMDB id goes to Jellyfin instead), so a
    # deployment can disable the auto-search wholesale and rely on transcript /
    # keyframe / operator-supplied grounding only. The per-search relevance gate
    # already discards bad matches when this is left on.
    enable_wikipedia_search: bool = True


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
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        debug_enabled=_env_flag("TUNABRAIN_DEBUG", False),
        show_llm_model=os.getenv("TUNABRAIN_SHOW_LLM_MODEL"),
        episode_llm_model=os.getenv("TUNABRAIN_EPISODE_LLM_MODEL"),
        schedule_llm_model=os.getenv("TUNABRAIN_SCHEDULE_LLM_MODEL"),
        review_llm_model=os.getenv("TUNABRAIN_REVIEW_LLM_MODEL"),
        bumpers_llm_model=os.getenv("TUNABRAIN_BUMPERS_LLM_MODEL"),
        schedule_max_shows=int(os.getenv("TUNABRAIN_SCHEDULE_MAX_SHOWS", "300")),
        stt_whisper_url=os.getenv(
            "TUNABRAIN_STT_WHISPER_URL",
            "http://whisper-http.wyoming.svc.cluster.local:10301",
        ),
        stt_subgen_url=os.getenv(
            "TUNABRAIN_STT_SUBGEN_URL", "http://subgen.arr.svc.cluster.local:9000"
        ),
        stt_default_backend=os.getenv("TUNABRAIN_STT_DEFAULT_BACKEND", "auto"),
        stt_whisper_model=os.getenv("TUNABRAIN_STT_WHISPER_MODEL", "turbo"),
        scratch_dir=os.getenv("TUNABRAIN_SCRATCH_DIR", "/tmp/tunabrain-scratch"),
        enrich_long_timeout=int(os.getenv("TUNABRAIN_ENRICH_LONG_TIMEOUT", "900")),
        enable_wikipedia_search=_env_flag("TUNABRAIN_ENABLE_WIKIPEDIA_SEARCH", True),
    )
    logger.info(
        "Loaded settings: provider=%s model=%s (shows=%s episodes=%s schedule=%s review=%s bumpers=%s) debug=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.show_llm_model or "default",
        settings.episode_llm_model or "default",
        settings.schedule_llm_model or "default",
        settings.review_llm_model or "default",
        settings.bumpers_llm_model or "default",
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
