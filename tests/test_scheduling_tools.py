"""Unit tests for scheduling tools."""

from __future__ import annotations

from datetime import datetime

import pytest

from tunabrain.agents.scheduling_tools import fill_time_slot, identify_schedule_gaps


class TestIdentifyScheduleGaps:
    """Tests for identify_schedule_gaps tool."""

    def test_empty_schedule_single_day(self):
        """Test gap identification with completely empty schedule."""
        gaps = identify_schedule_gaps(
            current_schedule={},
            start_date="2026-02-01",
            end_date="2026-02-02",
            immutable_slots=[],
            preferred_slots=None,
        )

        assert len(gaps) == 1
        assert gaps[0]["date"] == "2026-02-01"
        assert gaps[0]["gap_start"] == "06:00"
        assert gaps[0]["gap_end"] == "02:00"
        assert gaps[0]["duration_minutes"] == 1200  # 20 hours

    def test_empty_schedule_with_preferred_slots(self):
        """Test gap identification with preferred slot boundaries."""
        gaps = identify_schedule_gaps(
            current_schedule={},
            start_date="2026-02-01",
            end_date="2026-02-02",
            immutable_slots=[],
            preferred_slots=["08:00", "12:00", "18:00", "22:00"],
        )

        assert len(gaps) == 1
        gap = gaps[0]
        assert len(gap["suggested_slots"]) > 1  # Should split into multiple slots
        # First suggested slot should start at 06:00 (day start) and end at 08:00 (first preferred)
        assert gap["suggested_slots"][0]["start"] == "06:00"
        assert gap["suggested_slots"][0]["end"] == "08:00"

    def test_partially_filled_schedule(self):
        """Test gap identification with some slots already scheduled."""
        current_schedule = {
            "2026-02-01": [
                {
                    "start_time": "2026-02-01T10:00:00",
                    "end_time": "2026-02-01T11:00:00",
                    "media_id": "series:test",
                },
                {
                    "start_time": "2026-02-01T14:00:00",
                    "end_time": "2026-02-01T15:00:00",
                    "media_id": "series:test2",
                },
            ]
        }

        gaps = identify_schedule_gaps(
            current_schedule=current_schedule,
            start_date="2026-02-01",
            end_date="2026-02-02",
            immutable_slots=[],
            preferred_slots=None,
        )

        # Should have 3 gaps: before 10:00, between 11:00-14:00, after 15:00
        assert len(gaps) == 3
        assert gaps[0]["gap_end"] == "10:00"
        assert gaps[1]["gap_start"] == "11:00"
        assert gaps[1]["gap_end"] == "14:00"
        assert gaps[2]["gap_start"] == "15:00"

    def test_multiple_days(self):
        """Test gap identification across multiple days."""
        current_schedule = {
            "2026-02-01": [
                {
                    "start_time": "2026-02-01T10:00:00",
                    "end_time": "2026-02-01T11:00:00",
                    "media_id": "series:test",
                }
            ]
        }

        gaps = identify_schedule_gaps(
            current_schedule=current_schedule,
            start_date="2026-02-01",
            end_date="2026-02-03",  # 2 days
            immutable_slots=[],
            preferred_slots=None,
        )

        # Day 1: 2 gaps (before and after the slot)
        # Day 2: 1 gap (entire day)
        # Total: 3 gaps
        assert len(gaps) == 3
        dates = [gap["date"] for gap in gaps]
        assert "2026-02-01" in dates
        assert "2026-02-02" in dates

    def test_context_detection(self):
        """Test that context is properly detected for gaps."""
        gaps = identify_schedule_gaps(
            current_schedule={},
            start_date="2026-02-01",  # Saturday
            end_date="2026-02-02",
            immutable_slots=[],
            preferred_slots=None,
        )

        assert len(gaps) == 1
        # Should detect weekend
        assert "Weekend" in gaps[0]["context"]


class TestFillTimeSlot:
    """Tests for fill_time_slot tool."""

    def test_fill_empty_schedule(self):
        """Test filling a slot in an empty schedule."""
        schedule = {}
        result = fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="08:00",
            end_time="09:00",
            media_id="series:friends",
            selection_strategy="random",
            category_filters=["sitcom"],
            notes=["Test slot"],
        )

        assert "2026-02-01" in result
        assert len(result["2026-02-01"]) == 1

        slot = result["2026-02-01"][0]
        assert slot["start_time"] == "2026-02-01T08:00:00"
        assert slot["end_time"] == "2026-02-01T09:00:00"
        assert slot["media_id"] == "series:friends"
        assert slot["media_selection_strategy"] == "random"
        assert slot["category_filters"] == ["sitcom"]
        assert slot["notes"] == ["Test slot"]

    def test_fill_multiple_slots_same_day(self):
        """Test filling multiple non-overlapping slots on same day."""
        schedule = {}

        fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="08:00",
            end_time="09:00",
            media_id="series:friends",
        )

        fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="10:00",
            end_time="11:00",
            media_id="series:seinfeld",
        )

        assert len(schedule["2026-02-01"]) == 2
        # Should be sorted by start time
        assert schedule["2026-02-01"][0]["start_time"] == "2026-02-01T08:00:00"
        assert schedule["2026-02-01"][1]["start_time"] == "2026-02-01T10:00:00"

    def test_overlapping_slots_raises_error(self):
        """Test that overlapping slots raise ValueError."""
        schedule = {}

        fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="08:00",
            end_time="10:00",
            media_id="series:friends",
        )

        # Try to add overlapping slot
        with pytest.raises(ValueError, match="overlaps"):
            fill_time_slot(
                schedule=schedule,
                date="2026-02-01",
                start_time="09:00",
                end_time="11:00",
                media_id="series:seinfeld",
            )

    def test_adjacent_slots_no_overlap(self):
        """Test that adjacent slots (no gap) are allowed."""
        schedule = {}

        fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="08:00",
            end_time="09:00",
            media_id="series:friends",
        )

        # Adjacent slot should work
        fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="09:00",
            end_time="10:00",
            media_id="series:seinfeld",
        )

        assert len(schedule["2026-02-01"]) == 2

    def test_default_parameters(self):
        """Test that default parameters are applied correctly."""
        schedule = {}
        result = fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="08:00",
            end_time="09:00",
            media_id="series:friends",
        )

        slot = result["2026-02-01"][0]
        assert slot["media_selection_strategy"] == "random"  # default
        assert slot["category_filters"] == []  # default
        assert slot["notes"] == []  # default

    def test_schedule_modified_in_place(self):
        """Test that the schedule dict is modified in place."""
        schedule = {}
        result = fill_time_slot(
            schedule=schedule,
            date="2026-02-01",
            start_time="08:00",
            end_time="09:00",
            media_id="series:friends",
        )

        # Result should be the same object
        assert result is schedule
        assert "2026-02-01" in schedule
