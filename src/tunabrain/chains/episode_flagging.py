"""Episode special flag generation chain using constrained vocabulary."""
from __future__ import annotations

import logging
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tunabrain.api.models import MediaItem, EPISODE_SPECIAL_FLAGS
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model, LLMTask

logger = logging.getLogger(__name__)


class EpisodeFlaggingResult(BaseModel):
    """Structured response for episode special flags."""
    
    flags: list[str] = Field(
        description="List of special flags from the constrained vocabulary"
    )
    reasoning: str | None = Field(
        None,
        description="Brief explanation of why these flags were selected"
    )


async def generate_episode_flags(
    media: MediaItem,
    parent_title: str | None = None,
    existing_flags: list[str] | None = None,
    *,
    debug: bool = False,
) -> list[str]:
    """Generate constrained special flags for an episode.
    
    Uses a lightweight LLM to identify special episode characteristics
    from a fixed vocabulary (christmas, crossover, musical, etc.).
    
    Args:
        media: The episode media item
        parent_title: Title of the parent show for context
        existing_flags: Existing flags to preserve
        debug: Enable debug logging
    
    Returns:
        List of special flags (subset of EPISODE_SPECIAL_FLAGS)
    """
    
    if existing_flags is None:
        existing_flags = []
    
    logger.info(
        "Generating special flags for episode '%s' (parent: %s)",
        media.title,
        parent_title or "unknown"
    )
    
    # Get episode-specific lightweight model
    model = get_chat_model(task=LLMTask.EPISODE_FLAGGING)
    
    # Build the prompt
    allowed_flags_str = ", ".join(sorted(EPISODE_SPECIAL_FLAGS))
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert TV episode classifier. Analyze the episode and identify any special characteristics from the following allowed flags:

Allowed flags: {allowed_flags}

Episode Title: {episode_title}
Parent Show: {parent_title}
Season: {season}
Episode: {episode_number}
Description: {description}
Existing Tags: {existing_tags}

Identify which flags apply to this episode. Only use flags from the allowed list above.
Consider the episode description, title, and context to identify special characteristics.

Return a JSON object with:
- flags: list of applicable flags (empty list if none apply)
- reasoning: brief explanation of your selections"""
    )
    
    parser = PydanticOutputParser(pydantic_object=EpisodeFlaggingResult)
    
    chain = prompt | model | parser
    
    try:
        result = await chain.ainvoke({
            "allowed_flags": allowed_flags_str,
            "episode_title": media.title,
            "parent_title": parent_title or "Unknown",
            "season": media.season_number or "Unknown",
            "episode": media.episode_number or "Unknown",
            "description": media.description or "No description available",
            "existing_tags": ", ".join(existing_flags) if existing_flags else "None",
        })
        
        # Validate flags are in allowed set
        valid_flags = [f for f in result.flags if f in EPISODE_SPECIAL_FLAGS]
        
        logger.info(
            "Generated %d flags for '%s': %s",
            len(valid_flags),
            media.title,
            ", ".join(valid_flags) if valid_flags else "(none)"
        )
        
        if debug:
            logger.debug(
                "Episode flagging reasoning: %s",
                result.reasoning or "(not provided)"
            )
        
        return valid_flags
        
    except Exception as e:
        logger.error("Failed to generate episode flags: %s", e)
        if debug:
            logger.exception("Full error trace:")
        return []
