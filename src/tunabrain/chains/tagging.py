from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from tunabrain.api.models import MediaContext, MediaItem
from tunabrain.chains.context import resolve_media_context
from tunabrain.chains.validation import (
    format_kebab_feedback,
    partition_kebab_case,
)
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model

logger = logging.getLogger(__name__)


# Number of times to re-prompt the LLM after it returns non-kebab-case tags
# before giving up and filtering them out.  Mirrors _MAX_VALIDATION_RETRIES in
# chains/categorization.py so the two chains behave consistently.
_KEBAB_CASE_MAX_RETRIES = 2


class TaggingResult(BaseModel):
    """Free-form tag result."""

    tags: list[str] = Field(
        description="Free-form tags after reviewing current and existing taxonomy"
    )


# Shared instruction fragment appended to the system prompt of every free-form
# tag generation call.  Echoed in the human message and in the retry feedback so
# the model sees the format requirement up-front and on every re-prompt.
_KEBAB_CASE_INSTRUCTION = (
    "All tags MUST be in kebab-case format: lowercase words joined by single "
    "hyphens (e.g. 'action-and-adventure', 'sci-fi', 'documentary'). Do not "
    "use spaces, ampersands, capitals, or other special characters. Tags that "
    "do not follow this format will be rejected."
)


async def generate_tags(
    media: MediaItem,
    existing_tags: list[str] | None = None,
    *,
    debug: bool = False,
    task=None,
    context: MediaContext | None = None,
) -> tuple[list[str], MediaContext]:
    """Generate free-form tags for the provided media item.

    This function should orchestrate LangChain components to build a set of
    concise tags that help place the media into thematic schedules.

    Args:
        media: The media item to tag
        existing_tags: Existing tags to reuse when available
        debug: Enable debug logging
        task: LLMTask enum for task-specific model selection (default: inferred from media.is_episode)
        context: Optional grounding context to override the Wikipedia auto-search

    Returns:
        A ``(tags, resolved_context)`` tuple. The resolved context echoes back
        the reference (e.g. Wikipedia page) that grounded the tags so the caller
        can store and correct it.
    """

    # Infer task from media type if not specified
    if task is None:
        from tunabrain.llm import LLMTask

        task = LLMTask.EPISODE_FLAGGING if media.is_episode else LLMTask.SHOW_TAGGING

    logger.info(
        "Generating tags for '%s' (task=%s)",
        media.title,
        task.value if hasattr(task, "value") else task,
    )
    debug_enabled = is_debug_enabled(debug)

    llm = get_chat_model(task=task)  # Use task-specific model
    # Resolve the grounding context: a caller-supplied override (to correct a
    # bad match) if present, otherwise the automatic Wikipedia search. The
    # resolved context is echoed back so the caller can see and correct it.
    resolved = await resolve_media_context(media, context, llm=llm, debug=debug_enabled)
    wikipedia_context = resolved.grounding_text

    parser = PydanticOutputParser(pydantic_object=TaggingResult)

    chunk_parser = PydanticOutputParser(
        pydantic_object=TaggingResult,
    )

    async def invoke_prompt(prompt: ChatPromptTemplate, inputs: dict, parser: PydanticOutputParser):
        messages = prompt.format_messages(**inputs)
        response = await llm.ainvoke(messages)
        if debug_enabled:
            logger.debug("LLM raw response: %s", response)
        return await parser.ainvoke(response)

    chunk_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are vetting tags for a media library. Only choose tags that accurately apply "
                "to the media based on the provided metadata. Do not invent new tags; you may only "
                "select from the candidates in this batch. Return a JSON list of applicable tags.\n\n"
                f"{_KEBAB_CASE_INSTRUCTION}",
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
                "Wikipedia summary: {wikipedia_summary}\n\n"
                "Candidate tags (batch): {candidate_tags}\n\n"
                "Use the Wikipedia synopsis as needed to validate tags."
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
            batch_inputs = {
                "title": media.title,
                "description": media.description or "Not provided",
                "genres": ", ".join(media.genres) if media.genres else "Unknown",
                "duration": media.duration_minutes or "Unknown",
                "rating": media.rating or "Unknown",
                "current_tags": ", ".join(media.current_tags) if media.current_tags else "None",
                "wikipedia_summary": wikipedia_context,
                "candidate_tags": ", ".join(batch),
                "format_instructions": f"\n\n{chunk_parser.get_format_instructions()}",
            }
            if debug_enabled:
                logger.debug("LLM request (tag batch %s): %s", i // batch_size + 1, batch_inputs)
            try:
                result: TaggingResult = await invoke_prompt(
                    chunk_prompt, batch_inputs, chunk_parser
                )
            except OutputParserException as exc:
                logger.error(
                    "Failed to parse tagging batch %s. llm_output=%s",
                    i // batch_size + 1,
                    getattr(exc, "llm_output", "<missing>"),
                )
                raise
            if debug_enabled:
                logger.debug(
                    "LLM response (tag batch %s): %s",
                    i // batch_size + 1,
                    result.model_dump(),
                )
            for tag in result.tags:
                if tag not in selected:
                    selected.append(tag)

        # Safety net: the batch prompt instructs kebab-case, but the LLM can
        # still echo raw Jellyfin genre strings (e.g. "Action & Adventure") from
        # the candidate set when one of those is the best fit.  Filter them out
        # here so non-kebab-case values never feed the final prompt as
        # "vetted existing tags".  We do not retry the batch (the candidates
        # are fixed by the caller) — drop and move on.
        valid, invalid = partition_kebab_case(selected)
        if invalid:
            logger.warning(
                "Dropping non-kebab-case tag(s) from vetted-existing-tag set: %s",
                invalid,
            )
        return valid

    vetted_existing_tags = await evaluate_tag_batches(existing_tags or [])

    _EPISODE_VOCAB = (
        ":christmas, :halloween, :holiday, :finale, :premiere, :pilot, "
        ":musical, :crossover, :bottle-episode, :clip-show, :flashback, "
        ":anniversary, :standalone, :two-parter, :special"
    )

    if media.is_episode:
        episode_label = ""
        if media.season_number is not None and media.episode_number is not None:
            episode_label = f"Season {media.season_number}, Episode {media.episode_number}"
        elif media.season_number is not None:
            episode_label = f"Season {media.season_number}"
        elif media.episode_number is not None:
            episode_label = f"Episode {media.episode_number}"
        else:
            episode_label = "Unknown position"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a scheduling assistant tagging a specific TV episode. "
                    "The series already has genre and tone tags; do NOT re-derive series-level "
                    "tags like the show's genre or overall audience. "
                    "Instead, focus on what makes THIS episode distinctive within the series: "
                    "special themes, unusual format, narrative significance, or seasonal hooks "
                    "that a scheduler would use to choose this episode over others. "
                    "Prefer tags from the vetted existing list to avoid synonyms. "
                    "Keep 3-10 tags. Prioritise episode-specific vocabulary where applicable: "
                    f"{_EPISODE_VOCAB}. "
                    "Remove tags that are inaccurate or not useful for scheduling decisions.\n\n"
                    f"{_KEBAB_CASE_INSTRUCTION}",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                (
                    "human",
                    "Episode metadata for tagging:\n"
                    "- Title: {title}\n"
                    "- Position: {episode_label}\n"
                    "- Description: {description}\n"
                    "- Runtime (minutes): {duration}\n"
                    "- Rating: {rating}\n"
                    "- Current tags (review for removal): {current_tags}\n"
                    "- Vetted existing tags to reuse: {existing_tags}\n\n"
                    "Wikipedia summary: {wikipedia_summary}\n\n"
                    "Use the Wikipedia synopsis to confirm episode-specific details. "
                    "Return only the JSON dictated by the format instructions."
                    "{format_instructions}",
                ),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a scheduling assistant that assigns concise, reusable tags to media. "
                    "Prefer tags from the vetted existing list to avoid synonyms. "
                    "Keep 5-15 tags that describe genre, tone, audience, and programming value. "
                    "Remove tags that are irrelevant to scheduling or inaccurate.\n\n"
                    f"{_KEBAB_CASE_INSTRUCTION}",
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
                    "Wikipedia summary: {wikipedia_summary}\n\n"
                    "Use the Wikipedia synopsis to ensure accuracy."
                    " Return only the JSON dictated by the format instructions."
                    "{format_instructions}",
                ),
            ]
        )

    final_inputs = {
        "title": media.title,
        "description": media.description or "Not provided",
        "genres": ", ".join(media.genres) if media.genres else "Unknown",
        "duration": media.duration_minutes or "Unknown",
        "rating": media.rating or "Unknown",
        "current_tags": ", ".join(media.current_tags) if media.current_tags else "None",
        "existing_tags": ", ".join(vetted_existing_tags) if vetted_existing_tags else "None",
        "wikipedia_summary": wikipedia_context,
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }
    if media.is_episode:
        final_inputs["episode_label"] = episode_label

    # Re-prompt the LLM whenever it returns tags that are not in kebab-case so
    # it can re-format them.  After the retries are exhausted we filter below
    # so a non-kebab-case value never propagates downstream.  Mirrors the
    # option-set validation in chains/categorization._categorize_single.
    messages = prompt.format_messages(**final_inputs)
    result: TaggingResult | None = None
    response = None
    for attempt in range(_KEBAB_CASE_MAX_RETRIES + 1):
        response = await llm.ainvoke(messages)
        if debug_enabled:
            logger.debug(
                "LLM raw response (final tags, attempt %s): %s",
                attempt + 1,
                response,
            )
        try:
            result = await parser.ainvoke(response)
        except OutputParserException as exc:
            logger.error(
                "Failed to parse final tagging response (attempt %s). llm_output=%s",
                attempt + 1,
                getattr(exc, "llm_output", "<missing>"),
            )
            raise

        valid, invalid = partition_kebab_case(result.tags)
        if not invalid:
            result.tags = valid
            break

        logger.warning(
            "LLM returned non-kebab-case tag(s) (attempt %s): %s",
            attempt + 1,
            invalid,
        )
        if attempt < _KEBAB_CASE_MAX_RETRIES:
            # Feed the rejected tags back so the model can re-format them.
            messages = [
                *messages,
                response,
                HumanMessage(content=format_kebab_feedback(invalid)),
            ]

    assert result is not None  # the loop always runs at least once

    # Final safety net: never return tags outside kebab-case.
    valid, invalid = partition_kebab_case(result.tags)
    if invalid:
        logger.warning(
            "Dropping non-kebab-case tag(s) after %s attempt(s): %s",
            _KEBAB_CASE_MAX_RETRIES + 1,
            invalid,
        )
    result.tags = valid

    if debug_enabled:
        logger.debug("LLM response (final tags): %s", result.model_dump())

    logger.info("Generated %s tags for '%s'", len(result.tags), media.title)
    return result.tags, resolved.output
