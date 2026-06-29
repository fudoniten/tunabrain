from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from tunabrain.api.models import MediaItem
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model
from tunabrain.tools.wikipedia import WikipediaLookup


logger = logging.getLogger(__name__)


class TaggingResult(BaseModel):
    """Free-form tag result."""

    tags: list[str] = Field(
        description="Free-form tags after reviewing current and existing taxonomy"
    )


async def generate_tags(
    media: MediaItem, existing_tags: list[str] | None = None, *, debug: bool = False, task=None
) -> list[str]:
    """Generate free-form tags for the provided media item.

    This function should orchestrate LangChain components to build a set of
    concise tags that help place the media into thematic schedules.

    Args:
        media: The media item to tag
        existing_tags: Existing tags to reuse when available
        debug: Enable debug logging
        task: LLMTask enum for task-specific model selection (default: inferred from media.is_episode)
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
    wikipedia = WikipediaLookup(debug=debug_enabled, llm=llm)
    wikipedia_summary: str | None = None
    try:
        wikipedia_summary = await wikipedia.lookup_async(
            name=media.title,
            year=None,
            imdb_id=getattr(media, "imdb_id", None),
            llm=llm,
        )
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("Wikipedia lookup failed: %s", exc)

    wikipedia_context = wikipedia_summary or "Wikipedia summary not available."

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

        return selected

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
                    "Remove tags that are inaccurate or not useful for scheduling decisions.",
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
    if debug_enabled:
        logger.debug("LLM request (final tags): %s", final_inputs)
    try:
        result: TaggingResult = await invoke_prompt(prompt, final_inputs, parser)
    except OutputParserException as exc:
        logger.error(
            "Failed to parse final tagging response. llm_output=%s",
            getattr(exc, "llm_output", "<missing>"),
        )
        raise
    if debug_enabled:
        logger.debug("LLM response (final tags): %s", result.model_dump())

    logger.info("Generated %s tags for '%s'", len(result.tags), media.title)
    return result.tags
