"""Tools for the autonomous scheduling agent."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def identify_schedule_gaps(
    current_schedule: dict[str, list[dict]],
    start_date: str,
    end_date: str,
    immutable_slots: list[str],
    preferred_slots: list[str] | None = None,
) -> list[dict]:
    """Find unfilled time periods in the schedule.

    Args:
        current_schedule: Dict mapping date strings to lists of DailySlot dicts
        start_date: ISO date string for schedule start (e.g., "2026-02-01")
        end_date: ISO date string for schedule end (e.g., "2026-02-08")
        immutable_slots: Slot IDs that cannot be modified (format: "date:start_time")
        preferred_slots: Optional list of preferred slot times (e.g., ["08:00", "12:00"])

    Returns:
        List of gaps with suggested slot boundaries

    Examples:
        >>> identify_schedule_gaps(
        ...     current_schedule={"2026-02-01": [{"start_time": "2026-02-01T17:00:00", ...}]},
        ...     start_date="2026-02-01",
        ...     end_date="2026-02-02",
        ...     immutable_slots=["2026-02-01:2026-02-01T17:00:00"],
        ...     preferred_slots=["08:00", "12:00", "18:00"]
        ... )
        [
            {
                "date": "2026-02-01",
                "gap_start": "06:00",
                "gap_end": "17:00",
                "duration_minutes": 660,
                "suggested_slots": [
                    {"start": "08:00", "end": "12:00"},
                    {"start": "12:00", "end": "17:00"}
                ],
                "context": "Weekday daytime"
            },
            ...
        ]
    """
    gaps = []
    current = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    # Default broadcast day: 6 AM to 2 AM next day
    default_day_start = timedelta(hours=6)
    default_day_end = timedelta(hours=26)  # 2 AM next day

    while current < end:
        day_key = current.strftime("%Y-%m-%d")
        day_schedule = current_schedule.get(day_key, [])

        # Sort slots by start time
        sorted_slots = sorted(day_schedule, key=lambda s: s["start_time"])

        # Get day boundaries
        day_start = current + default_day_start
        day_end = current + default_day_end

        # Find gaps between slots
        if not sorted_slots:
            # Entire day is empty
            gaps.append(
                _create_gap_entry(
                    date=day_key,
                    gap_start=day_start,
                    gap_end=day_end,
                    preferred_slots=preferred_slots,
                    current=current,
                )
            )
        else:
            # Check gap before first slot
            first_slot_start = datetime.fromisoformat(sorted_slots[0]["start_time"])
            if first_slot_start > day_start:
                gaps.append(
                    _create_gap_entry(
                        date=day_key,
                        gap_start=day_start,
                        gap_end=first_slot_start,
                        preferred_slots=preferred_slots,
                        current=current,
                    )
                )

            # Check gaps between slots
            for i in range(len(sorted_slots) - 1):
                slot_end = datetime.fromisoformat(sorted_slots[i]["end_time"])
                next_slot_start = datetime.fromisoformat(sorted_slots[i + 1]["start_time"])

                if next_slot_start > slot_end:
                    gaps.append(
                        _create_gap_entry(
                            date=day_key,
                            gap_start=slot_end,
                            gap_end=next_slot_start,
                            preferred_slots=preferred_slots,
                            current=current,
                        )
                    )

            # Check gap after last slot
            last_slot_end = datetime.fromisoformat(sorted_slots[-1]["end_time"])
            if last_slot_end < day_end:
                gaps.append(
                    _create_gap_entry(
                        date=day_key,
                        gap_start=last_slot_end,
                        gap_end=day_end,
                        preferred_slots=preferred_slots,
                        current=current,
                    )
                )

        current += timedelta(days=1)

    logger.info(f"Identified {len(gaps)} schedule gaps")
    return gaps


def _create_gap_entry(
    date: str,
    gap_start: datetime,
    gap_end: datetime,
    preferred_slots: list[str] | None,
    current: datetime,
) -> dict:
    """Create a gap entry with suggested slot boundaries."""
    duration_minutes = int((gap_end - gap_start).total_seconds() / 60)

    # Generate suggested slots
    suggested_slots = []
    if preferred_slots:
        # Use preferred slot times as boundaries
        slot_start = gap_start
        for slot_time in preferred_slots:
            hour, minute = map(int, slot_time.split(":"))
            slot_boundary = current.replace(hour=hour, minute=minute)

            # Adjust if slot_boundary is the next day
            if slot_boundary < gap_start:
                slot_boundary += timedelta(days=1)

            if gap_start < slot_boundary < gap_end:
                suggested_slots.append(
                    {
                        "start": slot_start.strftime("%H:%M"),
                        "end": slot_boundary.strftime("%H:%M"),
                    }
                )
                slot_start = slot_boundary

        # Add final slot to gap_end if there's remaining time
        if slot_start < gap_end:
            suggested_slots.append(
                {"start": slot_start.strftime("%H:%M"), "end": gap_end.strftime("%H:%M")}
            )
    else:
        # No preferred slots, suggest the entire gap as one slot
        suggested_slots.append(
            {"start": gap_start.strftime("%H:%M"), "end": gap_end.strftime("%H:%M")}
        )

    # Determine context
    weekday = current.strftime("%A")
    hour = gap_start.hour
    if hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    elif hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "late night"

    is_weekend = current.weekday() in [5, 6]
    context = f"{'Weekend' if is_weekend else 'Weekday'} {time_of_day}"

    return {
        "date": date,
        "gap_start": gap_start.strftime("%H:%M"),
        "gap_end": gap_end.strftime("%H:%M"),
        "duration_minutes": duration_minutes,
        "suggested_slots": suggested_slots,
        "context": context,
    }


@tool
def fill_time_slot(
    schedule: dict[str, list[dict]],
    date: str,
    start_time: str,
    end_time: str,
    media_id: str,
    selection_strategy: str = "random",
    category_filters: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict:
    """Add a media item to the schedule in a specific slot.

    Args:
        schedule: Current schedule dict (will be modified in place)
        date: ISO date string (e.g., "2026-02-01")
        start_time: Time string in HH:MM format (e.g., "08:00")
        end_time: Time string in HH:MM format (e.g., "09:00")
        media_id: Media identifier (e.g., "series:friends", "movie:the-matrix", "random:sitcom")
        selection_strategy: "random" | "sequential" | "specific" (default: "random")
        category_filters: Optional list of category tags (e.g., ["comedy", "sitcom"])
        notes: Optional notes for this slot

    Returns:
        Updated schedule dict

    Raises:
        ValueError: If the slot overlaps with an existing slot

    Examples:
        >>> schedule = {}
        >>> fill_time_slot(
        ...     schedule=schedule,
        ...     date="2026-02-01",
        ...     start_time="08:00",
        ...     end_time="09:00",
        ...     media_id="series:friends",
        ...     category_filters=["sitcom"]
        ... )
        {
            "2026-02-01": [{
                "start_time": "2026-02-01T08:00:00",
                "end_time": "2026-02-01T09:00:00",
                "media_id": "series:friends",
                "media_selection_strategy": "random",
                "category_filters": ["sitcom"],
                "notes": []
            }]
        }
    """
    category_filters = category_filters or []
    notes = notes or []

    day_key = date
    if day_key not in schedule:
        schedule[day_key] = []

    # Parse times and create full datetime strings
    new_slot_start_dt = datetime.fromisoformat(f"{date}T{start_time}:00")
    new_slot_end_dt = datetime.fromisoformat(f"{date}T{end_time}:00")

    # Check for overlaps with existing slots
    for existing_slot in schedule[day_key]:
        existing_start = datetime.fromisoformat(existing_slot["start_time"])
        existing_end = datetime.fromisoformat(existing_slot["end_time"])

        # Check if there's any overlap
        if new_slot_start_dt < existing_end and new_slot_end_dt > existing_start:
            raise ValueError(
                f"Slot {start_time}-{end_time} overlaps with existing slot "
                f"{existing_start.strftime('%H:%M')}-{existing_end.strftime('%H:%M')}: "
                f"{existing_slot}"
            )

    # Add new slot
    new_slot = {
        "start_time": f"{date}T{start_time}:00",
        "end_time": f"{date}T{end_time}:00",
        "media_id": media_id,
        "media_selection_strategy": selection_strategy,
        "category_filters": category_filters,
        "notes": notes,
    }

    schedule[day_key].append(new_slot)

    # Sort by start time
    schedule[day_key].sort(key=lambda s: s["start_time"])

    logger.info(
        f"Added slot {date} {start_time}-{end_time}: {media_id} "
        f"(strategy: {selection_strategy}, filters: {category_filters})"
    )

    return schedule
