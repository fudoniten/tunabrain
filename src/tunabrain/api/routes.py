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
    TaggingRequest,
    TaggingResponse,
)
from tunabrain.chains.bumpers import generate_bumpers
from tunabrain.chains.categorization import categorize_media
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.chains.scheduling import build_schedule
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
    logger.info(
        "Processing schedule request for channel='%s' with %s media items",
        request.channel.name,
        len(request.media),
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

