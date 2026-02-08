from __future__ import annotations

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


def _format_value_sets(categories: dict[str, CategoryDefinition]) -> str:
    if not categories:
        return "No categories were provided."

    lines: list[str] = []
    for name, definition in categories.items():
        value_block = "\n".join(f"  - {value}" for value in definition.values)
        lines.append(f"- {name}: {definition.description}\n{value_block}")
    return "\n".join(lines)


def _fallback_dimensions(categories: dict[str, CategoryDefinition]) -> list[DimensionSelection]:
    fallback: list[DimensionSelection] = []
    for name, definition in categories.items():
        fallback.append(
            DimensionSelection(
                dimension=name,
                values=[definition.values[0]] if definition.values else [],
                notes=[
                    "Default selection used because structured LLM output was unavailable."
                ],
            )
        )
    return fallback


async def _call_llm(
    *,
    llm: RunnableSerializable,
    media: MediaItem,
    categories: dict[str, CategoryDefinition],
    channels: list[Channel],
    wikipedia_summary: str,
    parser: PydanticOutputParser,
    debug: bool,
) -> CategorizationResult:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a scheduling strategist selecting structured attributes for media. "
                "Use the provided dimensions and value sets as anchors. Always include every "
                "caller-provided dimension with 1-3 chosen values. Every dimension MUST have "
                "at least one value. If channels are supplied, suggest the top 1-3 channels "
                "with brief reasons. You may propose up to two additional scheduling dimensions "
                "only when they provide clear programming value, with concise names and 1-3 "
                "reasonable values for each.",
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
                "Scheduling categories (always include them):\n{value_sets}\n\n"
                "Available channels (optional):\n{channels}\n\n"
                "Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )

    channel_block = (
        "\n".join(
            f"- {channel.name}: {channel.description or 'No description provided'}"
            for channel in channels
        )
        if channels
        else "None provided"
    )

    inputs = {
        "title": media.title,
        "description": media.description or "Not provided",
        "genres": ", ".join(media.genres) if media.genres else "Unknown",
        "duration": media.duration_minutes or "Unknown",
        "rating": media.rating or "Unknown",
        "current_tags": ", ".join(media.current_tags) if media.current_tags else "None",
        "wikipedia_summary": wikipedia_summary,
        "value_sets": _format_value_sets(categories),
        "channels": channel_block,
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }

    if debug:
        logger.debug("LLM request (categorization): %s", inputs)

    messages = prompt.format_messages(**inputs)
    response = await llm.ainvoke(messages)

    if debug:
        logger.debug("LLM raw response (categorization): %s", response)

    return await parser.ainvoke(response)


async def categorize_media(
    media: MediaItem,
    categories: dict[str, CategoryDefinition],
    channels: list[Channel] | None = None,
    *,
    debug: bool = False,
    llm: RunnableSerializable | None = None,
) -> CategorizationResult:
    """Categorize media across caller-provided scheduling dimensions.

    Categories and their allowable values are supplied by the caller. The media metadata
    is enriched with a Wikipedia summary when possible. If channels are provided, the LLM
    will also suggest channel mappings. Fallbacks are provided when parsing fails.
    """

    debug_enabled = is_debug_enabled(debug)
    parser = PydanticOutputParser(pydantic_object=CategorizationResult)
    llm_instance = llm or get_chat_model()
    channels_list = channels or []

    logger.info(
        "Categorizing media '%s' with %s dimensions and %s channels",
        media.title,
        len(categories),
        len(channels_list),
    )

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

    try:
        result = await _call_llm(
            llm=llm_instance,
            media=media,
            categories=categories,
            channels=channels_list,
            wikipedia_summary=wikipedia_summary,
            parser=parser,
            debug=debug_enabled,
        )
    except OutputParserException as exc:
        logger.error(
            "Failed to parse categorization response. llm_output=%s",
            getattr(exc, "llm_output", "<missing>"),
        )
        result = CategorizationResult(
            dimensions=_fallback_dimensions(categories),
            channel_mappings=[],
        )
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("LLM categorization failed: %s", exc)
        result = CategorizationResult(
            dimensions=_fallback_dimensions(categories),
            channel_mappings=[],
        )

    if channels_list and not result.channel_mappings:
        result.channel_mappings = await map_media_to_channels(
            media,
            channels_list,
            debug=debug_enabled,
            llm=llm_instance,
        )

    if not result.dimensions:
        result.dimensions = _fallback_dimensions(categories)

    logger.info(
        "Categorization complete for '%s' with %s dimensions and %s channel mappings",
        media.title,
        len(result.dimensions),
        len(result.channel_mappings),
    )
    return result

