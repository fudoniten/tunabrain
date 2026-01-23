from __future__ import annotations

import logging
from datetime import datetime

from tunabrain.agents.scheduling_agent import build_schedule_with_agent
from tunabrain.api.models import Channel, MediaItem, ScheduleRequest, ScheduleResponse


logger = logging.getLogger(__name__)


async def build_schedule(
    *,
    channel: Channel,
    media: list[MediaItem],
    user_instructions: str | None,
    scheduling_window_days: int,
    debug: bool = False,
) -> ScheduleResponse:
    """Create a schedule for the provided channel and media list.

    This function now uses the autonomous scheduling agent to build schedules
    iteratively using LangGraph.
    """

    logger.info(
        "Schedule generation requested for channel='%s' with %s media items",
        channel.name,
        len(media),
    )

    # Convert old-style parameters to new ScheduleRequest format
    # Note: start_date is required in new format, so we default to now
    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime.now(),
        scheduling_window_days=scheduling_window_days,
        user_instructions=user_instructions,
        debug=debug,
    )

    # Use the autonomous agent to build the schedule
    return await build_schedule_with_agent(request)
