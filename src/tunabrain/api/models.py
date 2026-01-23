from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    """A piece of media in the Tunarr library."""

    id: str = Field(..., description="Unique identifier for the media item")
    title: str = Field(..., description="Title of the media")
    imdb_id: str | None = Field(
        None, description="IMDB identifier for the media item, e.g. tt0149460"
    )
    description: str | None = None
    genres: list[str] = Field(default_factory=list)
    duration_minutes: int | None = Field(None, description="Runtime in minutes")
    rating: str | None = Field(None, description="Content rating, e.g. TV-14")
    critical_rating: float | None = Field(None, description="Critic rating, from 1 to 10")
    audience_rating: float | None = Field(None, description="Audience rating, from 1 to 10")
    current_tags: list[str] = Field(
        default_factory=list,
        description="Existing tags already assigned to the media that should be reviewed",
    )


class Channel(BaseModel):
    """A Tunarr channel definition."""

    name: str
    description: str | None = None


class TaggingRequest(BaseModel):
    """Request to generate scheduling-oriented tags for a media item."""

    media: MediaItem
    existing_tags: list[str] = Field(
        default_factory=list,
        description="Preferred tags to reuse when generating a final tag set",
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class TaggingResponse(BaseModel):
    tags: list[str]


class TagSample(BaseModel):
    """Metadata about an existing tag for governance review."""

    tag: str = Field(..., description="The original tag value to review")
    usage_count: int = Field(
        0,
        description="Approximate usage count for prioritization; may be zero when unknown",
    )
    example_titles: list[str] = Field(
        default_factory=list, description="Representative titles that use this tag"
    )


class ChannelMappingRequest(BaseModel):
    """Request to map media to one or more channels."""

    media: MediaItem
    channels: list[Channel]
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class ChannelMapping(BaseModel):
    channel_name: str
    reasons: list[str] = Field(default_factory=list)


class ChannelMappingResponse(BaseModel):
    mappings: list[ChannelMapping]


class DimensionSelection(BaseModel):
    """Selected scheduling-friendly attributes for a media item."""

    dimension: str = Field(..., description="Name of the scheduling dimension")
    values: list[str] = Field(default_factory=list, description="Chosen values")
    notes: list[str] = Field(
        default_factory=list,
        description="Short notes or reasons for the chosen values",
    )


class CategoryDefinition(BaseModel):
    """Definition of a categorization dimension provided by the caller."""

    description: str = Field(..., description="What the dimension represents")
    values: list[str] = Field(
        default_factory=list, description="Candidate values the model may choose from"
    )


class CategorizationRequest(BaseModel):
    """Request to categorize media across scheduling-friendly dimensions."""

    media: MediaItem
    categories: dict[str, CategoryDefinition] = Field(
        default_factory=dict,
        description="Dimensions with descriptions and allowable values",
    )
    channels: list[Channel] = Field(
        default_factory=list,
        description="Optional channels to consider for mapping (used for backward compatibility)",
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class CategorizationResponse(BaseModel):
    dimensions: list[DimensionSelection] = Field(
        default_factory=list, description="Scheduling-friendly dimension selections"
    )
    mappings: list[ChannelMapping] = Field(
        default_factory=list,
        description="Optional channel mapping suggestions for compatibility",
    )


class DailySlot(BaseModel):
    """A single time slot on a given day."""

    start_time: datetime
    end_time: datetime
    media_id: str | None = Field(
        None,
        description="Media identifier: 'random:category', 'series:show-id', or 'movie:movie-id'",
    )
    media_selection_strategy: Literal["random", "sequential", "specific"] = Field(
        "random", description="How to select specific content within this slot"
    )
    category_filters: list[str] = Field(
        default_factory=list,
        description="Category tags to filter content (e.g., ['comedy', 'sitcom'])",
    )
    notes: list[str] = Field(default_factory=list)


class ScheduleRequest(BaseModel):
    """Request to build or extend a TV schedule."""

    channel: Channel
    media: list[MediaItem]

    # Scheduling parameters
    start_date: datetime = Field(
        ..., description="Start date/time for the schedule window (server local time)"
    )
    scheduling_window_days: int = Field(7, description="Number of days to schedule", ge=1, le=90)
    end_date: datetime | None = Field(
        None,
        description="Optional explicit end date (calculated from start_date + window_days if not provided)",
    )

    # Instructions and constraints
    user_instructions: str | None = Field(
        None,
        description="Natural language scheduling instructions (e.g., 'Weekday mornings: random sitcoms')",
    )

    # Slot management
    preferred_slots: list[str] | None = Field(
        None,
        description="Preferred slot times in HH:MM format (e.g., ['08:00', '12:00', '18:00'])",
    )
    daily_slots: list[DailySlot] = Field(
        default_factory=list,
        description="Pre-scheduled slots that should not be modified (for gap-filling mode)",
    )

    # Cost and performance
    cost_tier: Literal["economy", "balanced", "premium"] = Field(
        "balanced",
        description="Cost vs quality tradeoff (economy=local models, balanced=GPT-4o-mini, premium=GPT-4o)",
    )
    max_iterations: int = Field(
        50, description="Maximum agent iterations before stopping", ge=10, le=200
    )
    quality_threshold: float = Field(
        0.7,
        description="Minimum quality score to consider schedule complete (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    # Debugging
    debug: bool = Field(
        False,
        description="Enable debug logging for LLM calls and tool executions",
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class ReasoningSummary(BaseModel):
    """Summary of agent's decision-making process."""

    total_iterations: int = Field(..., description="Number of agent iterations executed")
    key_decisions: list[str] = Field(
        default_factory=list,
        description="5-10 most important decisions made by the agent",
    )
    constraints_applied: list[str] = Field(
        default_factory=list,
        description="Parsed constraints that guided scheduling",
    )
    completion_status: Literal["complete", "partial", "failed"] = Field(
        ..., description="Whether schedule was fully completed"
    )
    unfilled_slots_count: int = Field(0, description="Number of time slots that remain unfilled")
    quality_score: float = Field(0.0, description="Overall quality score (0.0-1.0)", ge=0.0, le=1.0)
    cost_estimate: dict = Field(
        default_factory=dict,
        description="Estimated cost breakdown for this schedule generation",
    )


class ScheduleResponse(BaseModel):
    """Response with generated schedule and reasoning."""

    overview: str = Field(..., description="High-level summary of the schedule")
    reasoning_summary: ReasoningSummary = Field(
        ..., description="Agent's decision-making process and results"
    )
    weekly_plan: list[str] = Field(
        default_factory=list, description="Day-by-day summary of scheduled content"
    )
    daily_slots: list[DailySlot] = Field(
        default_factory=list, description="Complete list of scheduled slots"
    )


class BumperRequest(BaseModel):
    """Request to generate bumpers for a channel schedule."""

    channel: Channel
    schedule_overview: str
    duration_seconds: int
    focus_window: str | None = Field(
        None, description="Temporal focus for the bumper, e.g. 'coming up this week'"
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class Bumper(BaseModel):
    title: str
    script: str
    duration_seconds: int


class BumperResponse(BaseModel):
    bumpers: list[Bumper]


class TagDecision(BaseModel):
    """Recommended action for a tag during cleanup/governance."""

    tag: str = Field(..., description="The original tag that was evaluated")
    action: Literal["keep", "drop", "merge", "rename"] = Field(
        ..., description="Governance action to take"
    )
    replacement: str | None = Field(
        None,
        description=(
            "Replacement or canonical tag when the action is merge or rename; null for"
            " keep or drop actions"
        ),
    )
    rationale: str = Field(
        ..., description="Short scheduling-focused reason for the recommendation"
    )


class TagTriageRequest(BaseModel):
    """Request to triage tags for scheduling usefulness and consolidation."""

    tags: list[TagSample] = Field(default_factory=list)
    target_limit: int | None = Field(
        None,
        description="Optional target tag count to help the model consolidate aggressively",
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class TagTriageResponse(BaseModel):
    decisions: list[TagDecision] = Field(
        default_factory=list, description="Per-tag governance recommendations"
    )


class TagAuditRequest(BaseModel):
    """Request to audit tags for scheduling usefulness."""

    tags: list[str] = Field(
        ..., description="List of tag names to audit for scheduling applicability"
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class TagAuditResult(BaseModel):
    """Result indicating whether a tag should be deleted and why."""

    tag: str = Field(..., description="The tag that was audited")
    reason: str = Field(
        ...,
        description=(
            "Reason why this tag should be deleted (e.g., too obscure, too detailed, "
            "too generic, not relevant for TV scheduling)"
        ),
    )


class TagAuditResponse(BaseModel):
    """Response containing tags recommended for deletion."""

    tags_to_delete: list[TagAuditResult] = Field(
        default_factory=list,
        description="Tags that should be deleted because they're not useful for scheduling",
    )
