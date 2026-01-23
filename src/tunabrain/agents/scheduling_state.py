"""State management for the autonomous scheduling agent."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from tunabrain.api.models import Channel, MediaItem


class SchedulingConstraints(TypedDict, total=False):
    """Structured constraints parsed from natural language instructions."""

    content_rules: list[dict]
    repetition_rules: dict
    quality_preferences: dict


class SchedulingState(TypedDict):
    """Complete state tracked throughout the scheduling process."""

    # === User Inputs (immutable) ===
    messages: Annotated[list[BaseMessage], add_messages]
    channel: Channel
    media_library: list[MediaItem]
    user_instructions: str | None
    scheduling_window_days: int
    start_date: datetime
    end_date: datetime
    preferred_slots: list[str] | None
    cost_tier: str  # "economy" | "balanced" | "premium"
    max_iterations: int
    quality_threshold: float

    # === Parsed Constraints (set by parse_constraints tool) ===
    constraints: SchedulingConstraints | None

    # === Working State (mutable) ===
    current_schedule: dict[str, list[dict]]  # Key: "2026-02-01", Value: [slot, slot, ...]
    immutable_slots: set[str]  # Set of slot IDs that cannot be modified

    # === Analysis Cache (performance optimization) ===
    media_analysis: dict | None
    gap_analysis: list[dict] | None

    # === Control Flow ===
    iterations: int
    confidence_score: float  # 0.0 - 1.0
    completion_status: str  # "in_progress" | "complete" | "partial" | "failed"

    # === Reasoning Capture ===
    key_decisions: list[str]
    tool_calls_made: list[dict]
