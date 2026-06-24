from __future__ import annotations

import logging

from fastapi import APIRouter
import uuid

from tunabrain.api.models import (
    BumperRequest,
    BumperResponse,
    CategorizationRequest,
    CategorizationResponse,
    ChannelMappingRequest,
    ChannelMappingResponse,
    ScheduleRequest,
    ScheduleResponse,
    TagAuditRequest,
    TagAuditResponse,
    TaggingRequest,
    TaggingResponse,
    TagTriageRequest,
    TagTriageResponse,
    EpisodeSpecialFlagRequest,
    EpisodeSpecialFlagResponse,
    QuarterlyStrategyRequest,
    QuarterlyStrategyResponse,
    MonthlyStrategyRequest,
    MonthlyStrategyResponse,
    ErrorResponse,
)
from tunabrain.chains.bumpers import generate_bumpers
from tunabrain.chains.categorization import categorize_media
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.chains.scheduling import build_schedule
from tunabrain.chains.tag_governance import audit_tags, triage_tags
from tunabrain.chains.tagging import generate_tags
from tunabrain.chains.episode_flagging import generate_episode_flags
from tunabrain.scheduling.quarterly_strategy import generate_quarterly_strategy
from tunabrain.scheduling.monthly_strategy import generate_monthly_strategy_agent_loop
from tunabrain.scheduling.cost import calculate_cost
from tunabrain.config import is_debug_enabled
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
    logger.info("Processing tagging request for title='%s'", request.media.title)
    tags = await generate_tags(
        request.media,
        request.existing_tags,
        debug=is_debug_enabled(request.debug),
    )
    logger.info("Generated %s tags for title='%s'", len(tags), request.media.title)
    return TaggingResponse(tags=tags)


@router.post("/channel-mapping", response_model=ChannelMappingResponse)
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
    )
    logger.info("Categorization complete with %s dimensions", len(categorization.dimensions))
    return CategorizationResponse(
        dimensions=categorization.dimensions,
        mappings=categorization.channel_mappings,
    )


@router.post("/schedule", response_model=ScheduleResponse)
async def schedule(request: ScheduleRequest) -> ScheduleResponse:
    logger.info(
        "Processing schedule request for channel='%s' with %s media items, "
        "start_date='%s', cost_tier='%s'",
        request.channel.name,
        len(request.media),
        request.start_date.strftime("%Y-%m-%d"),
        request.cost_tier,
    )
    # Note: build_schedule now accepts old-style params but converts to ScheduleRequest internally
    # In future, we can pass request directly to build_schedule_with_agent
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
        debug=is_debug_enabled(request.debug),
    )
    logger.info("Generated %s bumpers for channel='%s'", len(bumpers), request.channel.name)
    return BumperResponse(bumpers=bumpers)


@router.post("/tag-governance/triage", response_model=TagTriageResponse)
async def triage_tag_governance(request: TagTriageRequest) -> TagTriageResponse:
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
                "estimated_tokens": f"~3,500",
                "provider": "openrouter",
                "model": "gpt-4o-mini"
            },
            suggested_next_steps=[
                f"Review strategy with content team",
                f"Generate monthly strategies for each month in Q{request.quarter[1]}",
                f"Communicate themes to marketing and production",
                f"Finalize special events calendar"
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
                f"Review monthly strategy with content team",
                f"Generate weekly schedules for {request.month}",
                f"Allocate media to time blocks per recommendations",
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
