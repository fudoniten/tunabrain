"""Resolve the reference context that grounds tagging/categorization.

Both ``/tags`` and ``/categorize`` feed a short reference summary into the LLM
prompt. Historically that summary came from an automatic Wikipedia search whose
result was invisible to the caller, so a wrong match (an ambiguous title landing
on the wrong article) silently produced bad tags.

This module centralises that resolution and makes it two-way:

- If the caller supplies context (``summary``, ``text``, or ``links``), it is
  used to ground the model and the auto-search is skipped — this is how an
  operator corrects a bad match.
- Whatever is resolved is returned as a populated :class:`MediaContext` so the
  caller can see (and store, and correct) which reference drove the result.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from tunabrain.api.models import MediaContext, MediaItem
from tunabrain.config import get_settings
from tunabrain.tools.titles import is_placeholder_title
from tunabrain.tools.wikipedia import (
    WikipediaLookup,
    page_title_from_url,
)

logger = logging.getLogger(__name__)


# Grounding text substituted into the prompt when nothing could be resolved.
# Kept identical to the historical placeholder so prompts are unchanged when no
# context is available.
NO_CONTEXT_TEXT = "Wikipedia summary not available."


class ResolvedContext:
    """The outcome of resolving a request's grounding context.

    Attributes:
        grounding_text: The reference text to splice into the LLM prompt.
        output: The :class:`MediaContext` to echo back on the response.
    """

    __slots__ = ("grounding_text", "output")

    def __init__(self, grounding_text: str, output: MediaContext) -> None:
        self.grounding_text = grounding_text
        self.output = output


async def _resolve_from_links(
    links: list[str], *, llm: BaseChatModel | None, debug: bool
) -> tuple[str, list[str]] | None:
    """Resolve grounding text from caller-supplied links.

    Only Wikipedia links are fetched; other links are preserved in the returned
    link list but do not contribute grounding text. Returns ``(summary, links)``
    when at least one Wikipedia link resolved, else ``None`` so the caller can
    fall back to the auto-search.
    """
    wikipedia = WikipediaLookup(debug=debug, llm=llm)
    summaries: list[str] = []
    resolved_links: list[str] = []
    for link in links:
        title = page_title_from_url(link)
        if not title:
            # Non-Wikipedia reference: echo it, but we don't fetch arbitrary URLs.
            resolved_links.append(link)
            continue
        try:
            summary, url = await wikipedia.summarize_title_async(title, llm=llm)
        except Exception as exc:  # pragma: no cover - defensive catch for external service
            logger.warning("Failed to resolve context link %s: %s", link, exc)
            resolved_links.append(link)
            continue
        summaries.append(summary)
        resolved_links.append(url)

    if not summaries:
        return None
    return "\n\n".join(summaries), resolved_links


async def resolve_media_context(
    media: MediaItem,
    context: MediaContext | None = None,
    *,
    llm: BaseChatModel | None = None,
    debug: bool = False,
) -> ResolvedContext:
    """Resolve the grounding context for a tagging/categorization request.

    Precedence (first match wins; a caller override always skips the search):

    1. ``context.summary`` — reused verbatim (the corrected, stored summary).
    2. ``context.text`` — free-form operator notes, used directly.
    3. ``context.links`` — Wikipedia links fetched and summarized in place.
    4. Automatic Wikipedia search on the media title (historical behavior).

    The returned :class:`ResolvedContext` always carries both the text to feed
    the model and a populated :class:`MediaContext` to echo back.
    """
    context = context or MediaContext()

    # 1. Verbatim summary — the operator's stored/corrected grounding.
    if context.summary and context.summary.strip():
        grounding = context.summary.strip()
        return ResolvedContext(
            grounding,
            MediaContext(
                text=context.text,
                links=context.links,
                summary=grounding,
                source="provided-summary",
            ),
        )

    # 2. Free-form operator text.
    if context.text and context.text.strip():
        grounding = context.text.strip()
        return ResolvedContext(
            grounding,
            MediaContext(
                text=context.text,
                links=context.links,
                summary=grounding,
                source="provided-text",
            ),
        )

    # 3. Caller-supplied links.
    if context.links:
        resolved = await _resolve_from_links(context.links, llm=llm, debug=debug)
        if resolved is not None:
            grounding, resolved_links = resolved
            return ResolvedContext(
                grounding,
                MediaContext(
                    text=context.text,
                    links=resolved_links,
                    summary=grounding,
                    source="provided-link",
                ),
            )
        logger.info(
            "No Wikipedia links resolved from supplied context; falling back to auto-search"
        )

    # 4. Automatic Wikipedia search (only when no usable override was given).
    # Two guards keep the auto-search from inventing bad grounding:
    #   - a deployment can disable it wholesale (most of Grout's media is not on
    #     Wikipedia, so the search is noise there); and
    #   - a placeholder title ("Unknown", "<unnamed>", a bare filename that
    #     reduces to nothing) must never drive a search — that is exactly how
    #     "<unnamed>" ended up matching the anime "Unnamed Memory".
    if not get_settings().enable_wikipedia_search:
        logger.info(
            "Wikipedia auto-search disabled; no external grounding for %r", media.title
        )
        return ResolvedContext(NO_CONTEXT_TEXT, MediaContext(source="none"))
    if is_placeholder_title(media.title):
        logger.info(
            "Skipping Wikipedia auto-search for placeholder title %r", media.title
        )
        return ResolvedContext(NO_CONTEXT_TEXT, MediaContext(source="none"))

    wikipedia = WikipediaLookup(debug=debug, llm=llm)
    try:
        result = await wikipedia.resolve_async(
            name=media.title,
            year=None,
            imdb_id=getattr(media, "imdb_id", None),
            llm=llm,
        )
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("Wikipedia lookup failed: %s", exc)
        result = None

    if result is not None:
        summary, url = result
        return ResolvedContext(
            summary,
            MediaContext(links=[url], summary=summary, source="wikipedia"),
        )

    return ResolvedContext(NO_CONTEXT_TEXT, MediaContext(source="none"))
