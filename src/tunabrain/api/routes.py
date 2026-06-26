from __future__ import annotations

import logging

from fastapi import APIRouter

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
)
from tunabrain.chains.bumpers import generate_bumpers
from tunabrain.chains.categorization import categorize_media
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.chains.scheduling import build_schedule
from tunabrain.chains.tag_governance import audit_tags, triage_tags
from tunabrain.chains.tagging import generate_tags
from tunabrain.config import is_debug_enabled

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok"}


@router.post("/tags", response_model=TaggingResponse)
async def tag_media(request: TaggingRequest) -> TaggingResponse:
    """Generate free-form tags for a media item.

    Tags are free-form metadata, separate from dimensions. Use /categorize
    for structured dimension-based categorization (channel, genre, etc.).
    Both are valid: tags for arbitrary keywords, dimensions for controlled
    vocabulary scheduling attributes.
    """
    logger.info("Processing tagging request for title='%s'", request.media.title)
    tags = await generate_tags(
        request.media,
        request.existing_tags,
        debug=is_debug_enabled(request.debug),
    )
    logger.info("Generated %s tags for title='%s'", len(tags), request.media.title)
    return TaggingResponse(tags=tags)


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
        "Processing categorization request with %s media items and %s categories",
        len(request.media),
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
