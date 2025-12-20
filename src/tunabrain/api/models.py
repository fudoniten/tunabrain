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
    media_id: str | None = Field(None, description="Media item scheduled for this slot")
    notes: list[str] = Field(default_factory=list)


class ScheduleRequest(BaseModel):
    """Request to build a schedule for a channel and media set."""

    channel: Channel
    media: list[MediaItem]
    user_instructions: str | None = Field(
        None, description="User guidance for the schedule"
    )
    scheduling_window_days: int = Field(
        30, description="How many days the resulting schedule should cover"
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class ScheduleResponse(BaseModel):
    overview: str
    weekly_plan: list[str] = Field(default_factory=list)
    daily_slots: list[DailySlot] = Field(default_factory=list)


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

