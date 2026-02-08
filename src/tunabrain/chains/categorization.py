from __future__ import annotations

import asyncio
import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field

from tunabrain.api.models import (
    CategoryDefinition,
    Channel,
    ChannelMapping,
    DimensionSelection,
    MediaItem,
)
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model
from tunabrain.tools.wikipedia import WikipediaLookup


logger = logging.getLogger(__name__)


class CategorizationResult(BaseModel):
    """Structured response capturing dimension choices and optional channel mapping."""

    dimensions: list[DimensionSelection] = Field(
        description="Selected values for each scheduling dimension",
    )
    channel_mappings: list[ChannelMapping] = Field(
        default_factory=list,
        description="Optional channel mapping suggestions when channels are provided",
    )


class SingleDimensionResult(BaseModel):
    """Structured response for a single dimension categorization."""

    dimension: DimensionSelection = Field(
        description="Selected values for the scheduling dimension",
    )


def _fallback_dimension(name: str, definition: CategoryDefinition) -> DimensionSelection:
    """Return a fallback selection for a single category."""
    return DimensionSelection(
        dimension=name,
        values=[definition.values[0]] if definition.values else [],
        notes=["Default selection used because structured LLM output was unavailable."],
    )


def _fallback_dimensions(categories: dict[str, CategoryDefinition]) -> list[DimensionSelection]:
    return [_fallback_dimension(name, defn) for name, defn in categories.items()]


async def _categorize_single(
    *,
    llm: RunnableSerializable,
    media: MediaItem,
    category_name: str,
    category_definition: CategoryDefinition,
    wikipedia_summary: str,
    debug: bool,
) -> DimensionSelection:
    """Send a single category to the LLM and return its dimension selection."""
    parser = PydanticOutputParser(pydantic_object=SingleDimensionResult)

    value_block = "\n".join(f"  - {v}" for v in category_definition.values)
    formatted_category = f"- {category_name}: {category_definition.description}\n{value_block}"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a scheduling strategist selecting structured attributes for media. "
                "You will be given exactly one scheduling dimension with its candidate values. "
                "Choose 1-3 values from the candidates that best describe the media. "
                "The dimension MUST have at least one value.",
            ),
            (
                "human",
                "Media details:\n"
                "- Title: {title}\n"
                "- Description: {description}\n"
                "- Genres: {genres}\n"
                "- Runtime (minutes): {duration}\n"
                "- Rating: {rating}\n"
                "- Current tags: {current_tags}\n\n"
                "Wikipedia summary: {wikipedia_summary}\n\n"
                "Scheduling dimension to categorize:\n{category}\n\n"
                "Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )

    inputs = {
        "title": media.title,
        "description": media.description or "Not provided",
        "genres": ", ".join(media.genres) if media.genres else "Unknown",
        "duration": media.duration_minutes or "Unknown",
        "rating": media.rating or "Unknown",
        "current_tags": ", ".join(media.current_tags) if media.current_tags else "None",
        "wikipedia_summary": wikipedia_summary,
        "category": formatted_category,
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }

    if debug:
        logger.debug("LLM request (categorization/%s): %s", category_name, inputs)

    messages = prompt.format_messages(**inputs)
    response = await llm.ainvoke(messages)

    if debug:
        logger.debug("LLM raw response (categorization/%s): %s", category_name, response)

    result = await parser.ainvoke(response)
    dim = result.dimension

    # Ensure the dimension name matches the requested category.
    dim.dimension = category_name

    return dim


async def _categorize_single_safe(
    *,
    llm: RunnableSerializable,
    media: MediaItem,
    category_name: str,
    category_definition: CategoryDefinition,
    wikipedia_summary: str,
    debug: bool,
) -> DimensionSelection:
    """Wrapper around ``_categorize_single`` that returns a fallback on failure."""
    try:
        dim = await _categorize_single(
            llm=llm,
            media=media,
            category_name=category_name,
            category_definition=category_definition,
            wikipedia_summary=wikipedia_summary,
            debug=debug,
        )
        # Guarantee at least one value.
        if not dim.values:
            logger.warning(
                "LLM returned empty values for dimension '%s', applying fallback",
                category_name,
            )
            return _fallback_dimension(category_name, category_definition)
        return dim
    except OutputParserException as exc:
        logger.error(
            "Failed to parse categorization response for '%s'. llm_output=%s",
            category_name,
            getattr(exc, "llm_output", "<missing>"),
        )
        return _fallback_dimension(category_name, category_definition)
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("LLM categorization failed for '%s': %s", category_name, exc)
        return _fallback_dimension(category_name, category_definition)


async def categorize_media(
    media: MediaItem,
    categories: dict[str, CategoryDefinition],
    channels: list[Channel] | None = None,
    *,
    debug: bool = False,
    llm: RunnableSerializable | None = None,
) -> CategorizationResult:
    """Categorize media across caller-provided scheduling dimensions.

    Each category is sent as an individual LLM request so that every dimension
    is guaranteed to receive at least one value.  Requests are dispatched
    concurrently.  If channels are provided, channel mapping is handled via a
    dedicated chain after all dimensions are resolved.
    """

    debug_enabled = is_debug_enabled(debug)
    llm_instance = llm or get_chat_model()
    channels_list = channels or []

    logger.info(
        "Categorizing media '%s' with %s dimensions and %s channels",
        media.title,
        len(categories),
        len(channels_list),
    )

    # --- Wikipedia enrichment (shared across all category requests) ---
    wikipedia = WikipediaLookup(debug=debug_enabled, llm=llm_instance)
    wikipedia_summary = "Wikipedia summary not available."
    try:
        summary = await wikipedia.lookup_async(
            name=media.title,
            year=None,
            imdb_id=getattr(media, "imdb_id", None),
            llm=llm_instance,
        )
        if summary:
            wikipedia_summary = summary
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("Wikipedia lookup failed: %s", exc)

    # --- Per-category LLM calls (concurrent) ---
    if categories:
        dimensions = list(
            await asyncio.gather(
                *(
                    _categorize_single_safe(
                        llm=llm_instance,
                        media=media,
                        category_name=name,
                        category_definition=defn,
                        wikipedia_summary=wikipedia_summary,
                        debug=debug_enabled,
                    )
                    for name, defn in categories.items()
                )
            )
        )
    else:
        dimensions = []

    # --- Channel mapping (separate chain) ---
    channel_mappings: list[ChannelMapping] = []
    if channels_list:
        channel_mappings = await map_media_to_channels(
            media,
            channels_list,
            debug=debug_enabled,
            llm=llm_instance,
        )

    result = CategorizationResult(
        dimensions=dimensions,
        channel_mappings=channel_mappings,
    )

    logger.info(
        "Categorization complete for '%s' with %s dimensions and %s channel mappings",
        media.title,
        len(result.dimensions),
        len(result.channel_mappings),
    )
    return result

