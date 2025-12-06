from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    """A piece of media in the Tunarr library."""

    id: str = Field(..., description="Unique identifier for the media item")
    title: str
    description: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = Field(None, description="Runtime in minutes")
    rating: Optional[str] = Field(None, description="Content rating, e.g. TV-14")


class Channel(BaseModel):
    """A Tunarr channel definition."""

    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class TaggingRequest(BaseModel):
    """Request to generate scheduling-oriented tags for a media item."""

    media: MediaItem


class TaggingResponse(BaseModel):
    tags: List[str]


class ChannelMappingRequest(BaseModel):
    """Request to map media to one or more channels."""

    media: MediaItem
    channels: List[Channel]


class ChannelMapping(BaseModel):
    channel_id: str
    reasons: List[str] = Field(default_factory=list)


class ChannelMappingResponse(BaseModel):
    mappings: List[ChannelMapping]


class DailySlot(BaseModel):
    """A single time slot on a given day."""

    start_time: datetime
    end_time: datetime
    media_id: Optional[str] = Field(None, description="Media item scheduled for this slot")
    notes: List[str] = Field(default_factory=list)


class ScheduleRequest(BaseModel):
    """Request to build a schedule for a channel and media set."""

    channel: Channel
    media: List[MediaItem]
    user_instructions: Optional[str] = Field(None, description="User guidance for the schedule")
    scheduling_window_days: int = Field(
        30, description="How many days the resulting schedule should cover"
    )


class ScheduleResponse(BaseModel):
    overview: str
    weekly_plan: List[str] = Field(default_factory=list)
    daily_slots: List[DailySlot] = Field(default_factory=list)


class BumperRequest(BaseModel):
    """Request to generate bumpers for a channel schedule."""

    channel: Channel
    schedule_overview: str
    duration_seconds: int
    focus_window: Optional[str] = Field(
        None, description="Temporal focus for the bumper, e.g. 'coming up this week'"
    )


class Bumper(BaseModel):
    title: str
    script: str
    duration_seconds: int


class BumperResponse(BaseModel):
    bumpers: List[Bumper]

