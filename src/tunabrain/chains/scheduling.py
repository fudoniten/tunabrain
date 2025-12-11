from __future__ import annotations

from tunabrain.api.models import Channel, MediaItem, ScheduleResponse


async def build_schedule(
    *,
    channel: Channel,
    media: list[MediaItem],
    user_instructions: str | None,
    scheduling_window_days: int,
    debug: bool = False,
) -> ScheduleResponse:
    """Create a schedule for the provided channel and media list.

    The final implementation should coordinate multi-step prompting, such as
    drafting a monthly overview, refining into weekly themes, and populating
    daily time slots with content that respects the provided instructions.
    """

    raise NotImplementedError("Scheduling chain is not implemented yet")

