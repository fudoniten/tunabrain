#!/usr/bin/env python3
"""Manual test script for the autonomous scheduling agent.

This script demonstrates how to submit scheduling queries and inspect results.
Run this script directly to test the agent with sample data.

Usage:
    python test_scheduling_manual.py

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key (required for GPT models)
    TUNABRAIN_LLM_PROVIDER: LLM provider (default: openai)
    TUNABRAIN_LLM_MODEL: Model to use (default: gpt-4o-mini)
    TUNABRAIN_DEBUG: Enable debug logging (1, true, yes)
"""

import asyncio
import json
from datetime import datetime

from tunabrain.agents.scheduling_agent import build_schedule_with_agent
from tunabrain.api.models import Channel, MediaItem, ScheduleRequest


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


async def test_basic_schedule() -> None:
    """Test basic scheduling with a few sitcoms."""
    print_section("Test 1: Basic Schedule - Morning Sitcoms")

    # Create test channel
    channel = Channel(name="Comedy Central Test", description="24/7 comedy programming")

    # Create sample media library
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
        MediaItem(
            id="parks-rec-s01",
            title="Parks and Recreation Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
        MediaItem(
            id="30-rock-s01",
            title="30 Rock Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
    ]

    # Create scheduling request
    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 1, 8, 0),  # Saturday, 8 AM
        scheduling_window_days=1,  # Just one day for quick testing
        user_instructions="Fill morning time slots (8 AM - 1 PM) with sitcoms. Schedule 2-3 shows back-to-back.",
        preferred_slots=["08:00", "09:00", "10:00", "11:00", "12:00", "13:00"],
        cost_tier="balanced",
        max_iterations=15,
        quality_threshold=0.6,
        debug=False,
    )

    print("Request Configuration:")
    print(f"  Channel: {channel.name}")
    print(f"  Media items: {len(media)}")
    print(f"  Start date: {request.start_date}")
    print(f"  Duration: {request.scheduling_window_days} day(s)")
    print(f"  Instructions: {request.user_instructions}")
    print(f"  Max iterations: {request.max_iterations}")
    print(f"  Cost tier: {request.cost_tier}")

    # Run the agent
    print("\n🤖 Running scheduling agent...\n")
    response = await build_schedule_with_agent(request)

    # Display results
    print_section("Results")

    print(f"📋 Overview:\n{response.overview}\n")

    print("🔍 Reasoning Summary:")
    print(f"  Total iterations: {response.reasoning_summary.total_iterations}")
    print(f"  Completion status: {response.reasoning_summary.completion_status}")
    print(f"  Quality score: {response.reasoning_summary.quality_score:.2f}")
    print(f"  Unfilled slots: {response.reasoning_summary.unfilled_slots_count}")

    if response.reasoning_summary.key_decisions:
        print("\n  Key decisions:")
        for i, decision in enumerate(response.reasoning_summary.key_decisions, 1):
            print(f"    {i}. {decision}")

    if response.reasoning_summary.cost_estimate:
        print(f"\n  Cost estimate:")
        print(f"    {json.dumps(response.reasoning_summary.cost_estimate, indent=6)}")

    print("\n📺 Schedule (sorted by time):")
    if response.daily_slots:
        for slot in sorted(response.daily_slots, key=lambda s: s.start_time):
            print(
                f"  {slot.start_time.strftime('%a %m/%d %H:%M')} - "
                f"{slot.end_time.strftime('%H:%M')}: {slot.media_id}"
            )
    else:
        print("  ⚠️  No slots were scheduled")

    return response


async def test_gap_filling() -> None:
    """Test gap-filling with pre-scheduled content."""
    print_section("Test 2: Gap Filling - Work Around Pre-scheduled Content")

    from tunabrain.api.models import DailySlot

    channel = Channel(name="Prime Time Network")
    media = [
        MediaItem(id="drama1", title="Drama Show 1", genres=["drama"], duration_minutes=42),
        MediaItem(id="comedy1", title="Comedy Show 1", genres=["comedy"], duration_minutes=22),
        MediaItem(id="comedy2", title="Comedy Show 2", genres=["comedy"], duration_minutes=22),
    ]

    # Pre-schedule a primetime block (immutable)
    prescheduled = [
        DailySlot(
            start_time=datetime(2026, 2, 1, 20, 0),  # 8 PM - primetime
            end_time=datetime(2026, 2, 1, 21, 0),
            media_id="series:special-event",
            media_selection_strategy="specific",
            category_filters=[],
            notes=["Pre-scheduled primetime special - DO NOT MODIFY"],
        ),
    ]

    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 1, 18, 0),  # Start at 6 PM
        scheduling_window_days=1,
        user_instructions="Fill evening slots around the pre-scheduled primetime special. Schedule comedies before it, drama after it.",
        daily_slots=prescheduled,
        preferred_slots=["18:00", "19:00", "20:00", "21:00", "22:00"],
        max_iterations=15,
    )

    print("Request Configuration:")
    print(f"  Channel: {channel.name}")
    print(f"  Pre-scheduled slots: {len(prescheduled)}")
    print(f"  Start date: {request.start_date}")
    print(f"  Instructions: {request.user_instructions}")

    # Run the agent
    print("\n🤖 Running scheduling agent...\n")
    response = await build_schedule_with_agent(request)

    # Display results
    print_section("Results")
    print(f"📋 Overview:\n{response.overview}\n")

    print("📺 Schedule (sorted by time):")
    if response.daily_slots:
        for slot in sorted(response.daily_slots, key=lambda s: s.start_time):
            marker = "🔒 [LOCKED]" if "Pre-scheduled" in " ".join(slot.notes) else ""
            print(
                f"  {slot.start_time.strftime('%H:%M')} - "
                f"{slot.end_time.strftime('%H:%M')}: {slot.media_id} {marker}"
            )

    # Verify pre-scheduled slot is preserved
    prescheduled_preserved = any(
        slot.media_id == "series:special-event" for slot in response.daily_slots
    )
    print(f"\n✅ Pre-scheduled slot preserved: {prescheduled_preserved}")

    return response


async def test_multi_day_schedule() -> None:
    """Test scheduling across multiple days."""
    print_section("Test 3: Multi-Day Schedule - Week-Long Programming")

    channel = Channel(name="24/7 Classics", description="Classic TV shows around the clock")

    # Larger media library
    media = [
        MediaItem(
            id="mash-s01", title="M*A*S*H Season 1", genres=["comedy", "drama"], duration_minutes=25
        ),
        MediaItem(
            id="cheers-s01",
            title="Cheers Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
        MediaItem(
            id="frasier-s01",
            title="Frasier Season 1",
            genres=["comedy", "sitcom"],
            duration_minutes=22,
        ),
        MediaItem(id="taxi-s01", title="Taxi Season 1", genres=["comedy"], duration_minutes=22),
        MediaItem(
            id="twilight-zone",
            title="The Twilight Zone",
            genres=["sci-fi", "thriller"],
            duration_minutes=25,
        ),
        MediaItem(
            id="alfred-hitchcock",
            title="Alfred Hitchcock Presents",
            genres=["thriller", "mystery"],
            duration_minutes=25,
        ),
    ]

    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 3, 18, 0),  # Monday evening
        scheduling_window_days=3,  # 3 days
        user_instructions=(
            "Weekday evenings (6-10 PM): Mix of comedies and thrillers. "
            "Start with comedy, end with thriller each night."
        ),
        preferred_slots=["18:00", "19:00", "20:00", "21:00", "22:00"],
        cost_tier="balanced",
        max_iterations=25,  # More iterations for multi-day
        quality_threshold=0.65,
    )

    print("Request Configuration:")
    print(f"  Channel: {channel.name}")
    print(f"  Media items: {len(media)}")
    print(
        f"  Date range: {request.start_date.strftime('%Y-%m-%d')} ({request.scheduling_window_days} days)"
    )
    print(f"  Instructions: {request.user_instructions}")
    print(f"  Max iterations: {request.max_iterations}")

    # Run the agent
    print("\n🤖 Running scheduling agent...\n")
    response = await build_schedule_with_agent(request)

    # Display results
    print_section("Results")
    print(f"📋 Overview:\n{response.overview}\n")

    print(f"🔍 Agent Performance:")
    print(
        f"  Iterations used: {response.reasoning_summary.total_iterations}/{request.max_iterations}"
    )
    print(f"  Completion: {response.reasoning_summary.completion_status}")
    print(f"  Quality: {response.reasoning_summary.quality_score:.2f}")

    print("\n📺 Schedule by Day:")
    if response.daily_slots:
        # Group by day
        from collections import defaultdict

        by_day = defaultdict(list)
        for slot in response.daily_slots:
            day = slot.start_time.strftime("%a %m/%d")
            by_day[day].append(slot)

        for day in sorted(by_day.keys()):
            print(f"\n  {day}:")
            for slot in sorted(by_day[day], key=lambda s: s.start_time):
                print(
                    f"    {slot.start_time.strftime('%H:%M')}-"
                    f"{slot.end_time.strftime('%H:%M')}: {slot.media_id}"
                )
    else:
        print("  ⚠️  No slots were scheduled")

    return response


async def main() -> None:
    """Run all test scenarios."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              TunaBrain Scheduling Agent Test Suite                  ║
║                                                                      ║
║  This script demonstrates how to use the autonomous scheduling       ║
║  agent with different scenarios and query types.                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Check for API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set. Tests may fail.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        print()

    try:
        # Test 1: Basic scheduling
        await test_basic_schedule()

        # Test 2: Gap filling
        await test_gap_filling()

        # Test 3: Multi-day
        await test_multi_day_schedule()

        print_section("All Tests Complete")
        print("✅ All scenarios executed successfully!")
        print("\n💡 Tips:")
        print("  - Modify the media library in this script to test with your content")
        print("  - Adjust user_instructions to experiment with different constraints")
        print("  - Change cost_tier to 'economy' (local models) or 'premium' (GPT-4)")
        print("  - Set TUNABRAIN_DEBUG=1 to see detailed LLM interactions")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
