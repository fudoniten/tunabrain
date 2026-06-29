from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field

from tunabrain.api.models import Channel, ChannelMapping, MediaItem
from tunabrain.chains.validation import format_invalid_feedback, partition_values
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model

logger = logging.getLogger(__name__)

# Number of times to re-prompt the LLM after it selects channels that are not in
# the provided option set before giving up and filtering them out.
_MAX_VALIDATION_RETRIES = 2


class ChannelMappingResult(BaseModel):
    """DEPRECATED: Hardcoded channel mapping result. Not dimension-aware.

    Channels are a dimension now. Use /categorize with a "channel" dimension
    which returns DimensionSelection instead.
    See TS DIMENSION_CLEANUP.md for the full migration plan.
    """

    mappings: list[ChannelMapping] = Field(
        description="Chosen channels with human-readable justification.",
    )


def _format_channels(channels: Iterable[Channel]) -> str:
    return "\n".join(
        f"- {channel.name}: {channel.description or 'No description provided'}"
        for channel in channels
    )


async def _call_llm(
    *,
    llm: RunnableSerializable,
    media: MediaItem,
    channels: list[Channel],
    parser: PydanticOutputParser,
    debug: bool,
) -> ChannelMappingResult:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a programming director who assigns media to existing channels. " +
                "Use broad knowledge of genre, tone, setting, and audience to find the " +
                "best fit. Always pick 1-3 channels, even if descriptions are sparse. "
                "Provide concise reasons rooted in the media's content (not just "
                "matching keywords).",
            ),
            (
                "human",
                "Media details:\n"
                "- Title: {title}\n"
                "- Description: {description}\n"
                "- Genres: {genres}\n"
                "- Runtime (minutes): {duration}\n"
                "- Rating: {rating}\n\n"
                "Available channels:\n{channels}\n\n"
                "Choose the top 1-3 channels that best fit the media. Provide a short reason "
                "for each selection. Return only the JSON dictated by the format instructions."
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
        "channels": _format_channels(channels),
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }

    if debug:
        logger.debug("LLM request (channel mapping): %s", inputs)

    messages = prompt.format_messages(**inputs)

    allowed_names = [channel.name for channel in channels]

    result: ChannelMappingResult | None = None
    # Re-prompt whenever the LLM selects a channel outside the option set so it
    # can correct itself; invalid selections are filtered out below regardless.
    for attempt in range(_MAX_VALIDATION_RETRIES + 1):
        response = await llm.ainvoke(messages)

        if debug:
            logger.debug(
                "LLM raw response (channel mapping, attempt %s): %s", attempt + 1, response
            )

        result = await parser.ainvoke(response)

        selected = [mapping.channel_name for mapping in result.mappings]
        _, invalid = partition_values(selected, allowed_names)
        if not invalid:
            break

        logger.warning(
            "LLM selected invalid channel(s): %s (valid options: %s)",
            invalid,
            allowed_names,
        )

        if attempt < _MAX_VALIDATION_RETRIES:
            messages = [
                *messages,
                response,
                HumanMessage(content=format_invalid_feedback(invalid, allowed_names)),
            ]

    assert result is not None  # the loop always runs at least once

    # Final safety net: never return channels outside the provided option set.
    allowed_set = set(allowed_names)
    kept = [mapping for mapping in result.mappings if mapping.channel_name in allowed_set]
    if len(kept) != len(result.mappings):
        logger.warning(
            "Dropping invalid channel selection(s) after %s attempt(s)",
            _MAX_VALIDATION_RETRIES + 1,
        )
    result.mappings = kept

    return result


def _fallback_mapping(channels: list[Channel]) -> list[ChannelMapping]:
    if not channels:
        return []

    first = channels[0]
    return [
        ChannelMapping(
            channel_name=first.name,
            reasons=["Selected as the closest available option when LLM mapping was unavailable."],
        )
    ]


async def map_media_to_channels(
    media: MediaItem,
    channels: list[Channel],
    *,
    debug: bool = False,
    llm: RunnableSerializable | None = None,
) -> list[ChannelMapping]:
    """DEPRECATED: Hardcoded channel mapping. Not dimension-aware.

    This function asks the LLM to pick from a fixed list of Channel objects,
    reading media.genres (a hardcoded field). In the dimension model,
    channels are just one dimension among many, returned by /categorize.

    See TS DIMENSION_CLEANUP.md for the full migration plan.
    """

    if not channels:
        return []

    logger.info(
        "Mapping media '%s' to %s channels", media.title, len(channels)
    )
    debug_enabled = is_debug_enabled(debug)
    parser = PydanticOutputParser(pydantic_object=ChannelMappingResult)

    llm_instance = llm or get_chat_model()

    try:
        result = await _call_llm(
            llm=llm_instance,
            media=media,
            channels=channels,
            parser=parser,
            debug=debug_enabled,
        )
        mappings = result.mappings[:3]
        logger.info("LLM returned %s channel mappings for '%s'", len(mappings), media.title)
    except OutputParserException as exc:
        logger.error(
            "Failed to parse LLM channel mapping response. llm_output=%s",
            getattr(exc, "llm_output", "<missing>"),
        )
        mappings = _fallback_mapping(channels)
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("LLM channel mapping failed: %s", exc)
        mappings = _fallback_mapping(channels)

    if mappings:
        logger.info(
            "Returning %s channel mappings for '%s'", len(mappings), media.title
        )
    return mappings if mappings else _fallback_mapping(channels)

