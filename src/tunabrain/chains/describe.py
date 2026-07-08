"""Describe-only enrichment: derive a display title and short description.

Grout's free-form media arrives with only a rough working title — usually a
filename, an on-disk path, or the literal ``"Unknown"``. This chain refines
that title into something fit for a TV guide / browse UI and synthesises a
one-sentence description from whatever grounding is available (operator context,
a supplied summary, or the Wikipedia auto-search shared with /tags and
/categorize).

It is deliberately small: a single LLM call grounded on the resolved
:class:`MediaContext`. The endpoint never invents a title from nothing — the
caller must supply one (validated on :class:`EnrichDescribeRequest`) — and it
never hard-fails on a title it cannot improve. When the model errors or returns
unusable output, the working title is returned unchanged with a warning, mirroring
the "return whatever title we were given" contract in the spec.

``/enrich/short-form`` and ``/enrich/long-form`` will call :func:`describe_media`
internally as a follow-up so their responses can carry a title and description.
"""

from __future__ import annotations

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field

from tunabrain.api.models import (
    CostEstimate,
    DescribeMedia,
    EnrichDescribeResponse,
    MediaContext,
    MediaItem,
)
from tunabrain.chains.context import resolve_media_context
from tunabrain.config import get_settings, is_debug_enabled
from tunabrain.llm import get_chat_model
from tunabrain.scheduling.cost import calculate_cost

logger = logging.getLogger(__name__)


class DescribeResult(BaseModel):
    """Structured LLM output for a describe request."""

    title: str = Field(
        description=(
            "A clean, display-ready title derived from the working title and the "
            "available context. Never empty."
        ),
    )
    description: str | None = Field(
        None,
        description=(
            "A single-sentence description of the media, or null when a "
            "description would be noise (e.g. a short bumper or ident)."
        ),
    )


def _estimate_cost(llm_calls: int) -> CostEstimate:
    """Build a rough CostEstimate for ``llm_calls`` describe call(s).

    Mirrors the estimation style in ``chains/enrich_short``: a fixed per-call
    token budget priced against the configured model. Describe is a single
    small prompt, so the budget is modest.
    """
    model = get_settings().llm_model
    prompt_tokens = 900 * llm_calls
    completion_tokens = 150 * llm_calls
    cost_usd = calculate_cost(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return CostEstimate(
        estimated_cost_usd=cost_usd,
        llm_calls_used=llm_calls,
        estimated_tokens=f"~{prompt_tokens + completion_tokens:,}",
        model=model,
    )


_SYSTEM_PROMPT = (
    "You are a media librarian preparing a title and short description for a TV "
    "guide and browse UI. You are given a rough working title (which may be a "
    "filename, an on-disk path, or the literal 'Unknown') plus whatever "
    "grounding context is available.\n\n"
    "Rules:\n"
    "- Refine the working title into a clean, human-readable one. Strip filename "
    "cruft (dates, resolution, codec, release tags, extensions) and fix casing. "
    "Do NOT invent facts that the title and context do not support; when you "
    "cannot improve the title, return it as-is (cleaned of obvious cruft).\n"
    "- Write the description as a single concise sentence grounded in the title "
    "and context. Set description to null when a description would be noise, e.g. "
    "for a very short bumper, ident, or filler with no real content.\n"
    "- The grounding context (a transcript, keyframe captions, or operator notes) "
    "describes the ACTUAL media. When it conflicts with what the raw title seems "
    "to imply, trust the context and describe what the media actually is.\n"
    "- Keep both fields free of markup and quoting."
)


async def describe_media(
    media: MediaItem,
    context: MediaContext | None = None,
    *,
    debug: bool = False,
    llm: RunnableSerializable | None = None,
) -> EnrichDescribeResponse:
    """Derive a display title and short description for a media item.

    Resolves grounding context (a caller override, else the Wikipedia
    auto-search shared with /tags and /categorize), then makes a single LLM
    call to refine the title and synthesise a one-sentence description. The
    resolved context is echoed back for storage and correction.

    The endpoint always returns a non-empty title: if the LLM call fails or
    returns unusable output, the working title is returned unchanged and the
    problem is surfaced in ``warnings`` rather than raised.

    ``media``/``context`` are taken directly (rather than the request object) so
    the composite ``/enrich/short-form`` and ``/enrich/long-form`` chains can
    call this with the context they have already resolved — passing that summary
    back in reuses it verbatim and avoids a second Wikipedia lookup.
    """
    debug = is_debug_enabled(debug)
    logger.info("Describe for title='%s' (id=%s)", media.title, media.id)

    warnings: list[str] = []
    llm_instance = llm or get_chat_model()

    resolved = await resolve_media_context(
        media, context, llm=llm_instance, debug=debug
    )

    parser = PydanticOutputParser(pydantic_object=DescribeResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                "Media details:\n"
                "- Working title: {title}\n"
                "- Existing description: {description}\n"
                "- Duration (minutes): {duration}\n"
                "- Rating: {rating}\n"
                "- Is episode: {is_episode}\n"
                "- Current tags: {current_tags}\n\n"
                "Grounding context: {grounding}\n\n"
                "Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )
    inputs = {
        "title": media.title,
        "description": media.description or "Not provided",
        "duration": media.duration_minutes if media.duration_minutes is not None else "Unknown",
        "rating": media.rating or "Unknown",
        "is_episode": media.is_episode,
        "current_tags": ", ".join(media.current_tags) if media.current_tags else "None",
        "grounding": resolved.grounding_text,
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }
    if debug:
        logger.debug("LLM request (describe): %s", inputs)

    # The working title is the guaranteed fallback: the endpoint never returns
    # an empty title, and "I can't improve this" degrades to returning it
    # unchanged (with a warning) rather than a hard failure.
    title = media.title
    description = media.description
    try:
        messages = prompt.format_messages(**inputs)
        response = await llm_instance.ainvoke(messages)
        if debug:
            logger.debug("LLM raw response (describe): %s", response)
        result = await parser.ainvoke(response)

        refined = result.title.strip() if result.title else ""
        if refined:
            title = refined
        else:
            warnings.append("model returned an empty title; kept the working title")
        description = result.description.strip() if result.description else None
    except OutputParserException as exc:
        logger.error(
            "Failed to parse describe response for '%s'. llm_output=%s",
            media.title,
            getattr(exc, "llm_output", "<missing>"),
        )
        warnings.append(f"describe parse failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("Describe LLM call failed for '%s': %s", media.title, exc)
        warnings.append(f"describe failed: {exc}")

    response = EnrichDescribeResponse(
        media=DescribeMedia(id=media.id, title=title, description=description),
        context=resolved.output or MediaContext(),
        cost_estimate=_estimate_cost(1),
        warnings=warnings,
    )
    logger.info(
        "Describe complete for '%s' -> title='%s', description=%s, %s warnings",
        media.title,
        title,
        "set" if description else "null",
        len(warnings),
    )
    return response
