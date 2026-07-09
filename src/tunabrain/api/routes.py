from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter

from tunabrain.api.models import (
    BumperRequest,
    BumperResponse,
    CategorizationRequest,
    CategorizationResponse,
    ChannelMappingRequest,
    ChannelMappingResponse,
    EnrichDescribeRequest,
    EnrichDescribeResponse,
    EnrichLongFormRequest,
    EnrichLongFormResponse,
    EnrichShortFormRequest,
    EnrichShortFormResponse,
    DaypartSkeletonRequest,
    DaypartSkeletonResponse,
    EpisodeSpecialFlagRequest,
    EpisodeSpecialFlagResponse,
    MonthlyOverridesRequest,
    MonthlyOverridesResponse,
    MonthlyStrategyRequest,
    MonthlyStrategyResponse,
    QuarterlyGridRepairRequest,
    QuarterlyGridRepairResponse,
    QuarterlyGridRequest,
    QuarterlyGridResponse,
    QuarterlyStrategyRequest,
    QuarterlyStrategyResponse,
    ScheduleRequest,
    ScheduleResponse,
    StripFillRequest,
    StripFillResponse,
    TagAuditRequest,
    TagAuditResponse,
    TaggingRequest,
    TaggingResponse,
    TagTriageRequest,
    TagTriageResponse,
)
from tunabrain.chains.bumpers import generate_bumpers
from tunabrain.chains.categorization import categorize_media
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.chains.describe import describe_media
from tunabrain.chains.enrich_long import run_enrich_long_form
from tunabrain.chains.enrich_short import run_enrich_short_form
from tunabrain.chains.episode_flagging import generate_episode_flags
from tunabrain.chains.scheduling import build_schedule
from tunabrain.chains.tag_governance import audit_tags, triage_tags
from tunabrain.chains.tagging import generate_tags
from tunabrain.config import is_debug_enabled
from tunabrain.scheduling.cost import calculate_cost
from tunabrain.scheduling.monthly_overrides import propose_monthly_overrides
from tunabrain.scheduling.monthly_strategy import generate_monthly_strategy_agent_loop
from tunabrain.scheduling.quarterly_grid import (
    propose_daypart_skeleton,
    propose_quarterly_grid,
    propose_strip_fill,
    repair_quarterly_grid,
)
from tunabrain.scheduling.quarterly_strategy import generate_quarterly_strategy
from tunabrain.version import get_git_info

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok"}


@router.get("/api/version")
async def get_version() -> dict[str, str | None]:
    """Get version information including git commit and timestamp."""
    logger.debug("Version info requested")
    return get_git_info()


@router.post("/tags", response_model=TaggingResponse)
async def tag_media(request: TaggingRequest) -> TaggingResponse:
    """Generate free-form tags for a media item.

    Tags are free-form metadata, separate from dimensions. Use /categorize
    for structured dimension-based categorization (channel, genre, etc.).
    Both are valid: tags for arbitrary keywords, dimensions for controlled
    vocabulary scheduling attributes.
    """
    logger.info("Processing tagging request for title='%s'", request.media.title)
    tags, context = await generate_tags(
        request.media,
        request.existing_tags,
        debug=is_debug_enabled(request.debug),
        context=request.context,
    )
    logger.info("Generated %s tags for title='%s'", len(tags), request.media.title)
    return TaggingResponse(tags=tags, context=context)


# DEPRECATED: Hardcoded channel mapping. Channels are a dimension now.
# Use /categorize with a "channel" dimension instead.
# See TS DIMENSION_CLEANUP.md for the full migration plan.
@router.post("/channel-mapping", response_model=ChannelMappingResponse, deprecated=True)
async def channel_mapping(request: ChannelMappingRequest) -> ChannelMappingResponse:
    logger.info(
        "Processing channel mapping request with %s media items and %s channels",
        len(request.media),
        len(request.channels),
    )
    mappings = await map_media_to_channels(
        media=request.media,
        channels=request.channels,
        debug=is_debug_enabled(request.debug),
    )
    logger.info("Generated %s channel mappings", len(mappings))
    return ChannelMappingResponse(mappings=mappings)


@router.post("/categorize", response_model=CategorizationResponse)
async def categorize(request: CategorizationRequest) -> CategorizationResponse:
    logger.info(
        "Processing categorization request for media %s into %s categories",
        request.media.title,
        len(request.categories),
    )
    categorization = await categorize_media(
        media=request.media,
        categories=request.categories,
        channels=request.channels,
        debug=is_debug_enabled(request.debug),
        context=request.context,
    )
    logger.info("Categorization complete with %s dimensions", len(categorization.dimensions))
    return CategorizationResponse(
        dimensions=categorization.dimensions,
        mappings=categorization.channel_mappings,
        context=categorization.context,
    )


@router.post("/enrich/short-form", response_model=EnrichShortFormResponse)
async def enrich_short_form(request: EnrichShortFormRequest) -> EnrichShortFormResponse:
    """Enrich short-form media (bumpers, fillers, ads, music videos) in one call.

    Orchestrates the existing /categorize + /tags building blocks: the categories
    catalog is forwarded verbatim, and the grounding context resolved by
    categorize is propagated into tags. No STT — short-form media has no audio
    worth transcribing; filename, duration, and any operator context are enough.
    """
    logger.info("Processing short-form enrichment for title='%s'", request.media.title)
    response = await run_enrich_short_form(request)
    logger.info(
        "Short-form enrichment complete for '%s': %s dimensions, %s tags",
        request.media.title,
        len(response.dimensions),
        len(response.tags),
    )
    return response


@router.post("/enrich/long-form", response_model=EnrichLongFormResponse)
async def enrich_long_form(request: EnrichLongFormRequest) -> EnrichLongFormResponse:
    """Enrich long-form media (documentaries, video essays, interviews) in one call.

    Runs the full pipeline: fetch the media, extract audio, transcribe via the
    cluster's STT service (pluggable; defaults to auto), optionally caption a few
    keyframes, then categorize + tags grounded on the assembled transcript. Every
    stage degrades gracefully and the whole pipeline is bounded by a hard timeout.
    """
    logger.info(
        "Processing long-form enrichment for title='%s' (stt=%s)",
        request.media.title,
        request.options.stt_backend,
    )
    response = await run_enrich_long_form(request)
    logger.info(
        "Long-form enrichment complete for '%s': %s dimensions, %s tags, transcript=%s chars",
        request.media.title,
        len(response.dimensions),
        len(response.tags),
        len(response.transcript),
    )
    return response


@router.post("/enrich/describe", response_model=EnrichDescribeResponse)
async def enrich_describe(request: EnrichDescribeRequest) -> EnrichDescribeResponse:
    """Derive a display-ready title and short description for a media item.

    Takes a media item that already has a rough working ``title`` (a filename,
    an on-disk path, or the literal 'Unknown') and returns a refined title plus
    a one-sentence description synthesised from the resolved grounding context.
    This is the describe-only building block that /enrich/short-form and
    /enrich/long-form will call internally. The endpoint never invents a title
    from nothing (an empty title is rejected with 422) and always returns a
    non-empty title, degrading to the working title with a warning rather than
    failing.
    """
    logger.info("Processing describe enrichment for title='%s'", request.media.title)
    response = await describe_media(request.media, request.context, debug=request.debug)
    logger.info(
        "Describe enrichment complete for '%s' -> title='%s'",
        request.media.title,
        response.media.title,
    )
    return response


@router.post("/schedule", response_model=ScheduleResponse)
async def schedule(request: ScheduleRequest) -> ScheduleResponse:
    """Create a schedule using the autonomous agent.

    This endpoint uses the new LangGraph-based scheduling agent
    (build_schedule_with_agent) internally. The parameter style is
    transitional; the agent itself is current.

    NOTE: The layered grid endpoints (/api/scheduling/*) are planned
    but not yet implemented in this branch. This endpoint remains
    the active scheduling API until those land.
    """
    logger.info(
        "Processing schedule request for channel='%s' with %s media items, "
        "start_date='%s', cost_tier='%s'",
        request.channel.name,
        len(request.media),
        request.start_date.strftime("%Y-%m-%d"),
        request.cost_tier,
    )
    return await build_schedule(
        channel=request.channel,
        media=request.media,
        user_instructions=request.user_instructions,
        scheduling_window_days=request.scheduling_window_days,
        debug=is_debug_enabled(request.debug),
    )


@router.post("/bumpers", response_model=BumperResponse)
async def bumpers(request: BumperRequest) -> BumperResponse:
    logger.info(
        "Processing bumper generation for channel='%s' (duration=%ss)",
        request.channel.name,
        request.duration_seconds,
    )
    bumpers = await generate_bumpers(
        channel=request.channel,
        schedule_overview=request.schedule_overview,
        duration_seconds=request.duration_seconds,
        focus_window=request.focus_window,
        theme=request.theme,
        debug=is_debug_enabled(request.debug),
    )
    logger.info("Generated %s bumpers for channel='%s'", len(bumpers), request.channel.name)
    return BumperResponse(bumpers=bumpers)


@router.post("/tag-governance/triage", response_model=TagTriageResponse)
async def triage_tag_governance(request: TagTriageRequest) -> TagTriageResponse:
    """Tag governance triage for free-form tags.

    Keeps the free-form tag namespace clean. Dimensions use a controlled
    vocabulary and don't need governance.
    """
    logger.info("Processing tag governance triage for %s tags", len(request.tags))
    decisions = await triage_tags(
        request.tags,
        target_limit=request.target_limit,
        debug=is_debug_enabled(request.debug),
    )
    logger.info("Completed governance triage with %s recommendations", len(decisions))
    return TagTriageResponse(decisions=decisions)


@router.post("/tags/audit", response_model=TagAuditResponse)
async def audit_tag_usefulness(request: TagAuditRequest) -> TagAuditResponse:
    """Tag audit for free-form tags.

    Identifies tags that are not useful for scheduling. Free-form tags
    need governance; dimensions use a controlled vocabulary and don't.
    """
    logger.info("Processing tag audit for %s tags", len(request.tags))
    tags_to_delete = await audit_tags(
        request.tags,
        debug=is_debug_enabled(request.debug),
    )
    logger.info(
        "Completed tag audit: %s of %s tags recommended for deletion",
        len(tags_to_delete),
        len(request.tags),
    )
    return TagAuditResponse(tags_to_delete=tags_to_delete)


@router.post("/tags/episode-special-flag", response_model=EpisodeSpecialFlagResponse)
async def flag_episode_special(request: EpisodeSpecialFlagRequest) -> EpisodeSpecialFlagResponse:
    """Generate constrained special flags for an episode.
    
    Uses a lightweight LLM to identify special episode characteristics
    from a fixed vocabulary (christmas, crossover, musical, season-finale, etc.).
    
    This endpoint is designed for efficient bulk processing of episodes with
    cost-effective, lightweight models. Only use on episodes, not full show tagging.
    
    Args:
        request: Episode metadata and context
    
    Returns:
        Special flags for the episode
    """
    
    flags = await generate_episode_flags(
        media=request.media,
        parent_title=request.parent_title,
        existing_flags=request.existing_flags,
        debug=request.debug,
    )
    
    return EpisodeSpecialFlagResponse(flags=flags)


@router.post("/api/scheduling/get-quarterly-strategy", response_model=QuarterlyStrategyResponse)
async def get_quarterly_strategy(request: QuarterlyStrategyRequest) -> QuarterlyStrategyResponse:
    """Generate a quarterly programming strategy.
    
    This endpoint produces a high-level strategic overview for a quarter,
    including per-channel themes, special events, and implied monthly themes
    for guiding monthly planning.
    
    Args:
        request: QuarterlyStrategyRequest with quarter, channels, available media
    
    Returns:
        QuarterlyStrategyResponse with strategy, cost estimate, and next steps
    
    HTTP Responses:
        200: Strategy generated successfully
        400: Invalid request (bad quarter, year range, etc.)
        500: LLM invocation failed or response invalid
    """
    
    logger.info(
        f"Generating quarterly strategy for Q{request.quarter} {request.year} "
        f"({len(request.channels)} channels, {request.media_candidates.available_count} media items)"
    )
    
    try:
        # Generate strategy
        strategy = await generate_quarterly_strategy(request)
        logger.debug(f"Strategy generated: {strategy.quarter}, theme='{strategy.overall_theme}'")
        
        # Estimate cost (mock since we don't have actual token counts from LLM response yet)
        # In production, extract usage_metadata from LLM response
        estimated_tokens = len(strategy.overall_theme) + len(strategy.reasoning) + 500
        cost_usd = calculate_cost(
            model="gpt-4o-mini",  # Default model for now
            prompt_tokens=2000,  # Estimated
            completion_tokens=1500  # Estimated
        )
        
        # Generate strategy ID
        strategy_id = f"quarterly-q{request.quarter[1]}-{request.year}-{uuid.uuid4().hex[:8]}"
        
        return QuarterlyStrategyResponse(
            strategy_id=strategy_id,
            status="success",
            strategy=strategy,
            cost_estimate={
                "estimated_cost_usd": cost_usd,
                "llm_calls_used": 1,
                "estimated_tokens": "~3,500",
                "provider": "openrouter",
                "model": "gpt-4o-mini"
            },
            suggested_next_steps=[
                "Review strategy with content team",
                f"Generate monthly strategies for each month in Q{request.quarter[1]}",
                "Communicate themes to marketing and production",
                "Finalize special events calendar"
            ]
        )
    
    except ValueError as e:
        logger.error(f"Strategy generation validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Strategy generation runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating quarterly strategy: {e}")
        raise


@router.post("/api/scheduling/get-monthly-strategy", response_model=MonthlyStrategyResponse)
async def get_monthly_strategy(request: MonthlyStrategyRequest) -> MonthlyStrategyResponse:
    """Generate a monthly programming strategy using iterative agent refinement.
    
    This endpoint uses a multi-step agent loop (5-8 iterations) to converge on
    optimal monthly themes, time-block recommendations, and content mix.
    
    Args:
        request: MonthlyStrategyRequest with month, channels, media, optional quarterly context
    
    Returns:
        MonthlyStrategyResponse with final strategy, iteration history, and cost estimate
    
    HTTP Responses:
        200: Strategy generated and converged successfully
        400: Invalid request (bad month format, missing channels, etc.)
        500: LLM invocation failed or response invalid
    """
    
    logger.info(
        f"Generating monthly strategy for {request.month} "
        f"({len(request.channels)} channels, {request.media_candidates.available_count} media items, "
        f"max_iterations={request.max_iterations})"
    )
    
    try:
        # Run agent loop
        final_strategy, iterations_history, iteration_count, final_score = (
            await generate_monthly_strategy_agent_loop(request)
        )
        logger.info(
            f"Strategy converged in {iteration_count} iterations "
            f"with score {final_score:.2f}"
        )
        
        # Calculate cost (multi-LLM because of iterations)
        cost_usd = calculate_cost(
            model="gpt-4o-mini",
            prompt_tokens=2000 * iteration_count,  # Approx tokens per iteration
            completion_tokens=1500 * iteration_count
        )
        
        # Generate strategy ID
        strategy_id = f"monthly-{request.month.replace('-', '')}-{uuid.uuid4().hex[:8]}"
        
        return MonthlyStrategyResponse(
            strategy_id=strategy_id,
            status="success",
            strategy=final_strategy,
            iteration_count=iteration_count,
            convergence_score=final_score,
            iterations_history=iterations_history,
            cost_estimate={
                "estimated_cost_usd": cost_usd,
                "llm_calls_used": iteration_count,
                "estimated_tokens": f"~{3500 * iteration_count}",
                "provider": "openrouter",
                "model": "gpt-4o-mini"
            },
            suggested_next_steps=[
                "Review monthly strategy with content team",
                f"Generate weekly schedules for {request.month}",
                "Allocate media to time blocks per recommendations",
                f"Coordinate with marketing on opening tagline: '{final_strategy.opening_tagline}'"
            ]
        )
    
    except ValueError as e:
        logger.error(f"Strategy generation validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Strategy generation runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating monthly strategy: {e}")
        raise


@router.post(
    "/api/scheduling/propose-daypart-skeleton", response_model=DaypartSkeletonResponse
)
async def propose_daypart_skeleton_route(
    request: DaypartSkeletonRequest,
) -> DaypartSkeletonResponse:
    """Propose Pass A only: the coarse dayparting for a channel.

    Split-round-trip alternative to propose-quarterly-grid (see
    DURATION_AWARE_SCHEDULING.md §4.3, Option A): call this first to get real
    daypart bounds, compute a duration-feasible candidate menu per block from
    the catalog's runtime histogram, then call propose-strip-fill once per
    block with that menu attached.

    HTTP Responses:
        200: Skeleton proposed successfully
        400: Invalid request
        500: LLM invocation failed or response invalid
    """
    logger.info(
        "Proposing daypart skeleton for channel='%s' (%s shows in profile)",
        request.channel.name,
        len(request.catalog_profile.shows),
    )
    skeleton, llm_calls = await propose_daypart_skeleton(request)
    cost_usd = calculate_cost(
        model="gpt-4o-mini",
        prompt_tokens=1500 * llm_calls,
        completion_tokens=1000 * llm_calls,
    )
    return DaypartSkeletonResponse(
        skeleton=skeleton,
        cost_estimate={
            "estimated_cost_usd": cost_usd,
            "llm_calls_used": llm_calls,
            "estimated_tokens": f"~{2500 * llm_calls}",
            "provider": "openrouter",
            "model": "gpt-4o-mini",
        },
    )


@router.post("/api/scheduling/propose-strip-fill", response_model=StripFillResponse)
async def propose_strip_fill_route(request: StripFillRequest) -> StripFillResponse:
    """Propose Pass B for ONE daypart block, against its candidate menu.

    Call once per block returned by propose-daypart-skeleton, threading
    `prior_strips` forward for cross-daypart coherence (same role
    propose-quarterly-grid's internal loop plays for the single-call path).

    HTTP Responses:
        200: Strips proposed successfully (an empty list is valid — the
             caller should warn, not fail, same as propose-quarterly-grid)
        400: Invalid request
        500: LLM invocation failed or response invalid
    """
    logger.info(
        "Proposing strip fill for channel='%s' daypart='%s' (%s candidates, %s prior strips)",
        request.channel.name,
        request.block.name,
        len(request.candidates),
        len(request.prior_strips),
    )
    strips, llm_calls = await propose_strip_fill(
        request, request.block, request.prior_strips, candidates=request.candidates
    )
    cost_usd = calculate_cost(
        model="gpt-4o-mini",
        prompt_tokens=1500 * llm_calls,
        completion_tokens=1000 * llm_calls,
    )
    return StripFillResponse(
        strips=strips,
        cost_estimate={
            "estimated_cost_usd": cost_usd,
            "llm_calls_used": llm_calls,
            "estimated_tokens": f"~{2500 * llm_calls}",
            "provider": "openrouter",
            "model": "gpt-4o-mini",
        },
    )


@router.post("/api/scheduling/propose-quarterly-grid", response_model=QuarterlyGridResponse)
async def propose_grid(request: QuarterlyGridRequest) -> QuarterlyGridResponse:
    """Propose one channel's frozen quarterly grid (Phase 4).

    Runs two internal passes - dayparting skeleton, then strip-fill per daypart -
    against the supplied catalog profile. Per-channel by design: Tunarr Scheduler
    loops channels and calls this once each. The grid is a set of recurring rules;
    monthly overrides and weekly expansion happen downstream.

    HTTP Responses:
        200: Grid proposed successfully
        400: Invalid request
        500: LLM invocation failed or response invalid
    """
    logger.info(
        "Proposing quarterly grid for channel='%s' Q%s %s (%s shows in profile)",
        request.channel.name,
        request.quarter[1],
        request.year,
        len(request.catalog_profile.shows),
    )

    grid, skeleton, warnings, llm_calls = await propose_quarterly_grid(request)

    cost_usd = calculate_cost(
        model="gpt-4o-mini",
        prompt_tokens=1500 * llm_calls,
        completion_tokens=1000 * llm_calls,
    )
    grid_id = f"grid-{request.channel.name}-q{request.quarter[1]}-{request.year}-{uuid.uuid4().hex[:8]}"
    grid_id = grid_id.lower().replace(" ", "-")

    return QuarterlyGridResponse(
        grid_id=grid_id,
        status="partial" if warnings else "success",
        grid=grid,
        skeleton=skeleton,
        warnings=warnings,
        cost_estimate={
            "estimated_cost_usd": cost_usd,
            "llm_calls_used": llm_calls,
            "estimated_tokens": f"~{2500 * llm_calls}",
            "provider": "openrouter",
            "model": "gpt-4o-mini",
        },
        suggested_next_steps=[
            "Run the deterministic feasibility checker over this grid",
            "If shortfalls are found, call repair-quarterly-grid with the report",
            "Once feasible, freeze and store the grid in Tunarr Scheduler",
            "Generate monthly overrides on top of the frozen grid",
        ],
    )


@router.post(
    "/api/scheduling/repair-quarterly-grid", response_model=QuarterlyGridRepairResponse
)
async def repair_grid(request: QuarterlyGridRepairRequest) -> QuarterlyGridRepairResponse:
    """Repair a grid against deterministic feasibility findings (Phase 4/5).

    Targeted fix: only the strips named in the feasibility report should change.
    This is the LLM half of the propose -> check -> repair loop that Tunarr
    Scheduler drives.

    HTTP Responses:
        200: Grid repaired successfully
        400: Invalid request
        500: LLM invocation failed or response invalid
    """
    logger.info(
        "Repairing grid for channel='%s' (%s findings)",
        request.channel.name,
        len(request.feasibility_report.strip_findings),
    )

    revised, changes, llm_calls = await repair_quarterly_grid(request)

    cost_usd = calculate_cost(
        model="gpt-4o-mini",
        prompt_tokens=2000 * llm_calls,
        completion_tokens=1500 * llm_calls,
    )
    grid_id = f"grid-repair-{request.channel.name}-{uuid.uuid4().hex[:8]}".lower().replace(" ", "-")

    return QuarterlyGridRepairResponse(
        grid_id=grid_id,
        status="success",
        grid=revised,
        changes=changes,
        cost_estimate={
            "estimated_cost_usd": cost_usd,
            "llm_calls_used": llm_calls,
            "estimated_tokens": f"~{3500 * llm_calls}",
            "provider": "openrouter",
            "model": "gpt-4o-mini",
        },
    )


@router.post("/api/scheduling/propose-monthly-overrides", response_model=MonthlyOverridesResponse)
async def propose_overrides(request: MonthlyOverridesRequest) -> MonthlyOverridesResponse:
    """Propose sparse monthly overrides over a frozen grid (Phase 6).

    The grid is supplied as context so the LLM emits only the month's exceptions
    (special events, one-off marathons, recurring tweaks), never a re-authored
    schedule. An empty override list is a normal, common result.

    HTTP Responses:
        200: Overrides proposed successfully
        400: Invalid request
        500: LLM invocation failed or response invalid
    """
    logger.info(
        "Proposing monthly overrides for channel='%s' month=%s",
        request.channel.name,
        request.month,
    )

    overrides, warnings, llm_calls = await propose_monthly_overrides(request)

    cost_usd = calculate_cost(
        model="gpt-4o-mini",
        prompt_tokens=1500 * llm_calls,
        completion_tokens=800 * llm_calls,
    )
    overrides_id = (
        f"overrides-{request.channel.name}-{request.month.replace('-', '')}-{uuid.uuid4().hex[:8]}"
    ).lower().replace(" ", "-")

    return MonthlyOverridesResponse(
        overrides_id=overrides_id,
        status="partial" if warnings else "success",
        month=request.month,
        overrides=overrides,
        warnings=warnings,
        cost_estimate={
            "estimated_cost_usd": cost_usd,
            "llm_calls_used": llm_calls,
            "estimated_tokens": f"~{2300 * llm_calls}",
            "provider": "openrouter",
            "model": "gpt-4o-mini",
        },
        suggested_next_steps=[
            "Store the overrides in Tunarr Scheduler alongside the frozen grid",
            "Expand each week with the deterministic expander (grid + these overrides)",
            "No Tunabrain call is needed for weekly expansion",
        ],
    )
