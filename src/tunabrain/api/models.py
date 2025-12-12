from __future__ import annotations

from datetime import datetime

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

