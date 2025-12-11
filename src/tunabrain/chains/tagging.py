from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from tunabrain.api.models import MediaItem
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model
from tunabrain.tools.wikipedia import WikipediaLookupTool


logger = logging.getLogger(__name__)


class TaggingResult(BaseModel):
    """Structured response capturing the final tag set."""

    tags: list[str] = Field(
        description="Scheduling-friendly tags after reviewing current and existing taxonomy"
    )


async def generate_tags(
    media: MediaItem, existing_tags: list[str] | None = None, *, debug: bool = False
) -> list[str]:
    """Generate scheduling-friendly tags for the provided media item.

    This function should orchestrate LangChain components to build a set of
    concise tags that help place the media into thematic schedules.
    """

    debug_enabled = is_debug_enabled(debug)

    llm = get_chat_model()
    llm_with_tools = llm.bind_tools([WikipediaLookupTool(debug=debug_enabled)])

    parser = PydanticOutputParser(pydantic_object=TaggingResult)

    chunk_parser = PydanticOutputParser(
        pydantic_object=TaggingResult,
    )

    chunk_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are vetting tags for a media library. Only choose tags that accurately apply "
                "to the media based on the provided metadata. Do not invent new tags; you may only "
                "select from the candidates in this batch. Return a JSON list of applicable tags.",
            ),
            (
                "human",
                "Media metadata for evaluation:\n"
                "- Title: {title}\n"
                "- Description: {description}\n"
                "- Genres: {genres}\n"
                "- Runtime (minutes): {duration}\n"
                "- Rating: {rating}\n"
                "- Current tags: {current_tags}\n\n"
                "Candidate tags (batch): {candidate_tags}\n\n"
                "If the synopsis is unclear, call the wikipedia_media_lookup tool."
                " Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )

    async def evaluate_tag_batches(tags: Iterable[str]) -> list[str]:
        selected: list[str] = []
        tag_list = list(tags)
        if not tag_list:
            return selected

        batch_size = 75
        for i in range(0, len(tag_list), batch_size):
            batch = tag_list[i : i + batch_size]
            chain = chunk_prompt | llm_with_tools | chunk_parser
            batch_inputs = {
                "title": media.title,
                "description": media.description or "Not provided",
                "genres": ", ".join(media.genres) if media.genres else "Unknown",
                "duration": media.duration_minutes or "Unknown",
                "rating": media.rating or "Unknown",
                "current_tags": ", ".join(media.current_tags)
                if media.current_tags
                else "None",
                "candidate_tags": ", ".join(batch),
                "format_instructions": f"\n\n{chunk_parser.get_format_instructions()}",
            }
            if debug_enabled:
                logger.debug("LLM request (tag batch %s): %s", i // batch_size + 1, batch_inputs)
            result: TaggingResult = await chain.ainvoke(batch_inputs)
            if debug_enabled:
                logger.debug(
                    "LLM response (tag batch %s): %s",
                    i // batch_size + 1,
                    result.model_dump(),
                )
            for tag in result.tags:
                if tag not in selected:
                    selected.append(tag)

        return selected

    vetted_existing_tags = await evaluate_tag_batches(existing_tags or [])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a scheduling assistant that assigns concise, reusable tags to media. "
                "Prefer tags from the vetted existing list to avoid synonyms. "
                "Keep 5-15 tags that describe genre, tone, audience, and programming value. "
                "Remove tags that are irrelevant to scheduling or inaccurate.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            (
                "human",
                "Media metadata for tagging:\n"
                "- Title: {title}\n"
                "- Description: {description}\n"
                "- Genres: {genres}\n"
                "- Runtime (minutes): {duration}\n"
                "- Rating: {rating}\n"
                "- Current tags (review for removal): {current_tags}\n"
                "- Vetted existing tags to reuse: {existing_tags}\n\n"
                "If the synopsis is unclear, call the wikipedia_media_lookup tool."
                " Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm_with_tools | parser

    final_inputs = {
        "title": media.title,
        "description": media.description or "Not provided",
        "genres": ", ".join(media.genres) if media.genres else "Unknown",
        "duration": media.duration_minutes or "Unknown",
        "rating": media.rating or "Unknown",
        "current_tags": ", ".join(media.current_tags)
        if media.current_tags
        else "None",
        "existing_tags": ", ".join(vetted_existing_tags) if vetted_existing_tags else "None",
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }
    if debug_enabled:
        logger.debug("LLM request (final tags): %s", final_inputs)
    result: TaggingResult = await chain.ainvoke(final_inputs)
    if debug_enabled:
        logger.debug("LLM response (final tags): %s", result.model_dump())

    return result.tags

