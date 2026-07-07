"""Short-form enrichment: orchestrate /categorize + /tags in one round trip.

Grout's short-form media (bumpers, fillers, idents, ads, music videos) has no
audio worth transcribing — the only signals are the filename, duration, and any
operator-supplied context. This chain simply wraps the two existing building
blocks (:func:`categorize_media` and :func:`generate_tags`) so a caller gets
structured dimensions and free-form tags from a single request, with the
grounding context propagated from categorize into tags.

Each sub-call degrades gracefully: if categorize fails, tags are still
attempted (and vice versa), and the failure is surfaced as a warning rather
than a 500.
"""

from __future__ import annotations

import logging

from tunabrain.api.models import (
    CostEstimate,
    DescribeMedia,
    EnrichShortFormRequest,
    EnrichShortFormResponse,
    MediaContext,
)
from tunabrain.chains.categorization import categorize_media
from tunabrain.chains.describe import describe_media
from tunabrain.chains.tagging import generate_tags
from tunabrain.config import get_settings, is_debug_enabled
from tunabrain.scheduling.cost import calculate_cost

logger = logging.getLogger(__name__)


def _estimate_cost(llm_calls: int) -> CostEstimate:
    """Build a rough CostEstimate for an enrichment made of ``llm_calls`` calls.

    We don't have real token accounting from the LLM responses yet, so this
    mirrors the estimation style used by the scheduling routes: a fixed
    per-call token budget priced against the configured model.
    """
    model = get_settings().llm_model
    prompt_tokens = 1200 * llm_calls
    completion_tokens = 400 * llm_calls
    cost_usd = calculate_cost(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return CostEstimate(
        estimated_cost_usd=cost_usd,
        llm_calls_used=llm_calls,
        estimated_tokens=f"~{prompt_tokens + completion_tokens:,}",
        model=model,
    )


async def run_enrich_short_form(request: EnrichShortFormRequest) -> EnrichShortFormResponse:
    """Enrich short-form media by orchestrating categorize + tags.

    The categories catalog is forwarded verbatim to /categorize. The context
    resolved by categorize is fed into /tags so both calls ground on the same
    reference (and the Wikipedia auto-search runs at most once).
    """
    debug = is_debug_enabled(request.debug)
    logger.info(
        "Enrich short-form for title='%s' (%s dimensions, %s existing tags)",
        request.media.title,
        len(request.categories),
        len(request.existing_tags),
    )

    warnings: list[str] = []

    dimensions = []
    # The context to propagate into tags: seeded with the request's context and
    # upgraded to categorize's resolved context when that call succeeds.
    resolved_context: MediaContext | None = request.context
    categorize_calls = 0
    try:
        categorization = await categorize_media(
            media=request.media,
            categories=request.categories,
            channels=request.channels,
            debug=debug,
            context=request.context,
        )
        dimensions = categorization.dimensions
        resolved_context = categorization.context
        # One LLM call per dimension (plus channel mapping when channels given).
        categorize_calls = len(request.categories) + (1 if request.channels else 0)
    except Exception as exc:  # pragma: no cover - defensive; categorize is robust internally
        logger.warning("Enrich short-form: categorize failed for '%s': %s", request.media.title, exc)
        warnings.append(f"categorize failed: {exc}")

    tags: list[str] = []
    tags_calls = 0
    try:
        tags, tag_context = await generate_tags(
            request.media,
            request.existing_tags,
            debug=debug,
            context=resolved_context,
        )
        # generate_tags always echoes back the grounding it used; prefer it so
        # the stored context reflects the last call.
        resolved_context = tag_context
        tags_calls = 1
    except Exception as exc:  # pragma: no cover - defensive; tagging is robust internally
        logger.warning("Enrich short-form: tags failed for '%s': %s", request.media.title, exc)
        warnings.append(f"tags failed: {exc}")

    # Derive a display title + short description. Reuse the context resolved by
    # categorize/tags so describe grounds on the same reference (and the
    # Wikipedia auto-search never runs a second time).
    describe: DescribeMedia | None = None
    describe_calls = 0
    try:
        describe_result = await describe_media(
            request.media, resolved_context, debug=debug
        )
        describe = describe_result.media
        describe_calls = describe_result.cost_estimate.llm_calls_used
        warnings.extend(describe_result.warnings)
        # Only adopt describe's echoed context when categorize/tags resolved
        # nothing; otherwise keep the upstream context so its provenance (e.g.
        # source='wikipedia') isn't flattened to 'provided-summary'.
        if not (resolved_context and resolved_context.summary):
            resolved_context = describe_result.context
    except Exception as exc:  # pragma: no cover - defensive; describe is robust internally
        logger.warning("Enrich short-form: describe failed for '%s': %s", request.media.title, exc)
        warnings.append(f"describe failed: {exc}")

    llm_calls = max(1, categorize_calls + tags_calls + describe_calls)
    response = EnrichShortFormResponse(
        media=request.media,
        describe=describe,
        dimensions=dimensions,
        tags=tags,
        context=resolved_context,
        cost_estimate=_estimate_cost(llm_calls),
        warnings=warnings,
    )
    logger.info(
        "Enrich short-form complete for '%s': %s dimensions, %s tags, %s warnings",
        request.media.title,
        len(dimensions),
        len(tags),
        len(warnings),
    )
    return response
