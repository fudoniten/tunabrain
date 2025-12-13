from __future__ import annotations

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


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/tags", response_model=TaggingResponse)
async def tag_media(request: TaggingRequest) -> TaggingResponse:
    tags = await generate_tags(
        request.media,
        request.existing_tags,
        debug=is_debug_enabled(request.debug),
    )
    return TaggingResponse(tags=tags)


@router.post("/channel-mapping", response_model=ChannelMappingResponse)
async def channel_mapping(request: ChannelMappingRequest) -> ChannelMappingResponse:
    mappings = await map_media_to_channels(
        media=request.media,
        channels=request.channels,
        debug=is_debug_enabled(request.debug),
    )
    return ChannelMappingResponse(mappings=mappings)


@router.post("/categorize", response_model=CategorizationResponse)
async def categorize(request: CategorizationRequest) -> CategorizationResponse:
    categorization = await categorize_media(
        media=request.media,
        categories=request.categories,
        channels=request.channels,
        debug=is_debug_enabled(request.debug),
    )
    return CategorizationResponse(
        dimensions=categorization.dimensions,
        mappings=categorization.channel_mappings,
    )


@router.post("/schedule", response_model=ScheduleResponse)
async def schedule(request: ScheduleRequest) -> ScheduleResponse:
    return await build_schedule(
        channel=request.channel,
        media=request.media,
        user_instructions=request.user_instructions,
        scheduling_window_days=request.scheduling_window_days,
        debug=is_debug_enabled(request.debug),
    )


@router.post("/bumpers", response_model=BumperResponse)
async def bumpers(request: BumperRequest) -> BumperResponse:
    bumpers = await generate_bumpers(
        channel=request.channel,
        schedule_overview=request.schedule_overview,
        duration_seconds=request.duration_seconds,
        focus_window=request.focus_window,
        debug=is_debug_enabled(request.debug),
    )
    return BumperResponse(bumpers=bumpers)

