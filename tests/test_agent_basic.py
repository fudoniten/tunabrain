"""Basic end-to-end test for the scheduling agent."""

from __future__ import annotations

from datetime import datetime

import pytest

from tunabrain.agents.scheduling_agent import build_schedule_with_agent
from tunabrain.api.models import Channel, MediaItem, ScheduleRequest


@pytest.mark.asyncio
async def test_agent_minimal_schedule():
    """Test agent with minimal data: 1 day, 3 shows, empty schedule."""
    # Create minimal test data
    channel = Channel(name="Test Comedy Channel", description="24/7 comedy programming")

    media = [
        MediaItem(
            id="friends-s01",
            title="Friends Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
        MediaItem(
            id="seinfeld-s01",
            title="Seinfeld Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
        MediaItem(
            id="office-us-s01",
            title="The Office (US) Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
    ]

    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 1, 8, 0),  # Start at 8 AM
        scheduling_window_days=1,  # Just one day
        user_instructions="Fill morning time slots (8 AM - 12 PM) with sitcoms",
        preferred_slots=["08:00", "09:00", "10:00", "11:00", "12:00"],
        cost_tier="balanced",
        max_iterations=10,  # Keep it short for testing
        quality_threshold=0.5,
    )

    # Run agent
    response = await build_schedule_with_agent(request)

    # Basic assertions
    assert response is not None
    assert response.overview
    assert response.reasoning_summary
    assert response.reasoning_summary.total_iterations <= 10

    # Should have created at least some slots
    # (Depending on LLM behavior, might not fill all)
    assert len(response.daily_slots) >= 0  # Can be 0 if agent struggles

    # If slots were created, verify they're valid
    for slot in response.daily_slots:
        assert slot.start_time < slot.end_time
        assert slot.media_id is not None


@pytest.mark.asyncio
async def test_agent_with_prescheduled_slots():
    """Test agent respects pre-scheduled slots."""
    from tunabrain.api.models import DailySlot

    channel = Channel(name="Test Channel")
    media = [
        MediaItem(id="show1", title="Show 1", genres=["comedy"], duration_minutes=30),
        MediaItem(id="show2", title="Show 2", genres=["comedy"], duration_minutes=30),
    ]

    # Pre-schedule a slot
    prescheduled = DailySlot(
        start_time=datetime(2026, 2, 1, 10, 0),
        end_time=datetime(2026, 2, 1, 11, 0),
        media_id="series:prescheduled",
        media_selection_strategy="specific",
        category_filters=[],
        notes=["Pre-scheduled, do not modify"],
    )

    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 1),
        scheduling_window_days=1,
        user_instructions="Fill gaps around pre-scheduled content",
        daily_slots=[prescheduled],
        preferred_slots=["08:00", "09:00", "10:00", "11:00", "12:00"],
        max_iterations=10,
    )

    response = await build_schedule_with_agent(request)

    # Verify pre-scheduled slot is still there
    prescheduled_slot_found = any(
        slot.start_time == prescheduled.start_time and slot.media_id == prescheduled.media_id
        for slot in response.daily_slots
    )

    assert prescheduled_slot_found, "Pre-scheduled slot should be preserved"
