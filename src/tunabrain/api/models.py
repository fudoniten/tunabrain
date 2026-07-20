from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tunabrain.scheduling.grid import (
    CatalogProfile,
    DaypartBlock,
    DaypartCandidate,
    DaypartSkeleton,
    FeasibilityReport,
    Grid,
    GridStrip,
    Override,
    SelectionStrategy,
)


class MediaItem(BaseModel):
    """A piece of media in the Tunarr library.

    DEPRECATED FIELD: `genres` is a hardcoded first-class field. In the dimension
    model, genre should be just another dimension value (e.g. "genre:comedy"),
    not a top-level schema property. Use `categories` in CategorizationRequest
    to supply dimensions instead.
    """

    id: str = Field(..., description="Unique identifier for the media item")
    title: str = Field(..., description="Title of the media")
    imdb_id: str | None = Field(
        None, description="IMDB identifier for the media item, e.g. tt0149460"
    )
    description: str | None = None
    # DEPRECATED: Hardcoded genres field. Use dimensions instead.
    genres: list[str] = Field(default_factory=list)
    duration_minutes: int | None = Field(None, description="Runtime in minutes")
    rating: str | None = Field(None, description="Content rating, e.g. TV-14")
    critical_rating: float | None = Field(None, description="Critic rating, from 1 to 10")
    audience_rating: float | None = Field(None, description="Audience rating, from 1 to 10")
    current_tags: list[str] = Field(
        default_factory=list,
        description="Existing tags already assigned to the media that should be reviewed",
    )
    is_episode: bool = Field(
        False, description="True when this item is a TV episode rather than a standalone film"
    )
    season_number: int | None = Field(None, description="Season number for TV episodes")
    episode_number: int | None = Field(None, description="Episode number within the season")
    parent_id: str | None = Field(
        None, description="ID of the parent series when this item is a TV episode"
    )


class Channel(BaseModel):
    """A Tunarr channel definition."""

    name: str
    description: str | None = None


class MediaContext(BaseModel):
    """Reference information that grounds a tagging/categorization request.

    Tunabrain normally grounds tagging by auto-searching Wikipedia for the
    media's title. That search can land on the wrong article (e.g. an ambiguous
    title), and because the matched page was never surfaced, a bad match was
    invisible — producing bad tags/categories with no way to diagnose them.

    This model closes that loop in both directions:

    - **On the request**, a caller may supply ``text``, ``summary``, or
      ``links`` to override the auto-search. Any supplied grounding is used
      instead of searching Wikipedia, so operators can correct a bad match.
    - **On the response**, ``summary`` (the text actually fed to the model),
      ``source`` (where it came from), and ``links`` (e.g. the Wikipedia page
      the search matched) are always populated, so the grounding is visible.

    Store the returned context in Tunarr Scheduler and edit it in a UI; sending
    the corrected context back on the next request re-tags against the fix.
    """

    text: str | None = Field(
        None,
        description=(
            "Free-form operator-supplied description or notes about the media. "
            "When present (and no summary is given), it grounds the model "
            "directly and the Wikipedia auto-search is skipped."
        ),
    )
    links: list[str] = Field(
        default_factory=list,
        description=(
            "Reference URLs about the media. Wikipedia links are fetched and "
            "summarized in place of the auto-search; other links are echoed "
            "but not fetched. On a response this carries the page(s) actually "
            "used (e.g. the Wikipedia article the search matched)."
        ),
    )
    summary: str | None = Field(
        None,
        description=(
            "The resolved reference text used to ground the model. Always set "
            "on a response. If supplied on a request it is reused verbatim and "
            "no lookup runs — this is the field to store and correct for stable "
            "re-tagging."
        ),
    )
    source: str | None = Field(
        None,
        description=(
            "Provenance of the resolved summary: 'provided-summary', "
            "'provided-text', 'provided-link', 'wikipedia', or 'none'. Set on "
            "responses so a bad auto-match is diagnosable."
        ),
    )


class TaggingRequest(BaseModel):
    """Request to generate free-form tags for a media item.

    Tags are free-form metadata, separate from dimensions. Use
    CategorizationRequest for structured dimension-based categorization.
    Both are valid: tags for arbitrary keywords, dimensions for controlled
    vocabulary scheduling attributes.
    """

    media: MediaItem
    existing_tags: list[str] = Field(
        default_factory=list,
        description="Preferred tags to reuse when generating a final tag set",
    )
    context: MediaContext | None = Field(
        None,
        description=(
            "Optional grounding context to override the Wikipedia auto-search. "
            "Supply corrected info here to fix bad tagging from a wrong match."
        ),
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class TaggingResponse(BaseModel):
    """Free-form tag response."""

    tags: list[str]
    context: MediaContext = Field(
        default_factory=MediaContext,
        description=(
            "The grounding context actually used, echoed back so it can be "
            "stored and corrected. Reveals which reference (e.g. Wikipedia "
            "page) drove the tags."
        ),
    )


class TagSample(BaseModel):
    """Metadata about an existing tag for governance review.

    Tag governance helps keep the free-form tag namespace clean and useful.
    Dimensions use a controlled vocabulary, so they don't need governance.
    """

    tag: str = Field(..., description="The original tag value to review")
    usage_count: int = Field(
        0,
        description="Approximate usage count for prioritization; may be zero when unknown",
    )
    example_titles: list[str] = Field(
        default_factory=list, description="Representative titles that use this tag"
    )


# DEPRECATED: Hardcoded channel mapping request. Use CategorizationRequest with "channel" dimension.
class ChannelMappingRequest(BaseModel):
    """Request to map media to one or more channels.

    DEPRECATED: Channels are a dimension now. Use CategorizationRequest
    with a "channel" dimension instead.
    """

    media: MediaItem
    channels: list[Channel]
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


# DEPRECATED: Hardcoded channel mapping. Use DimensionSelection with "channel" dimension.
class ChannelMapping(BaseModel):
    """DEPRECATED: Hardcoded channel mapping. Use DimensionSelection instead."""

    channel_name: str
    reasons: list[str] = Field(default_factory=list)


# DEPRECATED: Hardcoded channel mapping response. Use CategorizationResponse instead.
class ChannelMappingResponse(BaseModel):
    """DEPRECATED: Hardcoded channel mapping response. Use CategorizationResponse instead."""

    mappings: list[ChannelMapping]


class DimensionSelection(BaseModel):
    """Selected scheduling-friendly attributes for a media item."""

    dimension: str = Field(..., description="Name of the scheduling dimension")
    values: list[str] = Field(default_factory=list, description="Chosen values")
    notes: list[str] = Field(
        default_factory=list,
        description="Short notes or reasons for the chosen values",
    )


class CategoryValue(BaseModel):
    """A single value within a categorization dimension with optional description."""

    value: str = Field(..., description="The value identifier/name")
    description: str | None = Field(
        None, description="Optional description of what this value represents"
    )


class CategoryDefinition(BaseModel):
    """Definition of a categorization dimension provided by the caller."""

    description: str = Field(..., description="What the dimension represents")
    values: list[str] | list[CategoryValue] = Field(
        default_factory=list,
        description="Candidate values (as strings or CategoryValue objects with descriptions)",
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
    context: MediaContext | None = Field(
        None,
        description=(
            "Optional grounding context to override the Wikipedia auto-search. "
            "Supply corrected info here to fix bad categorization from a wrong match."
        ),
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
    context: MediaContext = Field(
        default_factory=MediaContext,
        description=(
            "The grounding context actually used, echoed back so it can be "
            "stored and corrected. Reveals which reference (e.g. Wikipedia "
            "page) drove the categorization."
        ),
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


class BumperRequest(BaseModel):
    """Request to generate bumpers for a channel schedule."""

    channel: Channel
    schedule_overview: str
    duration_seconds: int
    focus_window: str | None = Field(
        None, description="Temporal focus for the bumper, e.g. 'coming up this week'"
    )
    theme: str | None = Field(
        None, description="Optional creative theme override"
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM and downstream service calls",
    )


class Bumper(BaseModel):
    title: str
    script: str
    duration_seconds: int
    image_base64: str | None = Field(
        None, description="Base64-encoded PNG image for the bumper visual"
    )


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

class EpisodeSpecialFlagRequest(BaseModel):
    """Request to generate constrained special flags for an episode."""
    
    media: MediaItem = Field(..., description="Episode metadata")
    parent_title: str | None = Field(
        None,
        description="Title of the parent show for context"
    )
    existing_flags: list[str] = Field(
        default_factory=list,
        description="Existing flags to preserve when generating new ones"
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for outgoing LLM calls"
    )


class EpisodeSpecialFlagResponse(BaseModel):
    """Response with constrained special flags for an episode."""
    
    flags: list[str] = Field(
        description="List of special flags (constrained vocabulary)"
    )


# Constrained vocabulary for episode special flags
EPISODE_SPECIAL_FLAGS = {
    "christmas",
    "halloween", 
    "crossover",
    "series-finale",
    "musical",
    "special-event",
    "clip-show",
    "flashback",
    "dream-sequence",
    "bottle-episode",
    "clip-compilation",
    "season-premiere",
    "season-finale",
    "two-part",
    "movie-special",
    "guest-star",
    "live-action",
    "animation",
}


# ============================================================================
# Quarterly Strategy Models
# ============================================================================


class CostEstimate(BaseModel):
    """Cost tracking for an operation."""
    
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")
    llm_calls_used: int = Field(..., description="Number of LLM calls made")
    estimated_tokens: str = Field(..., description="Token estimate (e.g., '~2,800')")
    provider: str = Field(default="openrouter", description="LLM provider name")
    model: str | None = Field(None, description="Model name used")


class ChannelContext(BaseModel):
    """Channel context for scheduling."""
    
    name: str = Field(..., description="Channel name")
    description: str = Field(..., description="Channel description/purpose")


class MediaCandidateSummary(BaseModel):
    """Summary of available media (avoids listing all items)."""
    
    available_count: int = Field(..., description="Total available episodes/movies")
    summary: str = Field(..., description="Breakdown by genre/type")
    preview_sample: list[MediaItem] = Field(
        default_factory=list,
        description="5-10 representative items for LLM context"
    )
    tag_availability: dict[str, int] = Field(
        default_factory=dict,
        description="Approximate counts by tag"
    )


class ChannelStrategyAdjustment(BaseModel):
    """Programming guidance for one channel."""
    
    channel: str = Field(..., description="Channel name")
    theme: str = Field(..., description="Channel's programming focus (1-2 sentences)")
    rationale: str = Field(..., description="Why this theme fits this channel")
    recommended_mix: dict[str, str] = Field(
        default_factory=dict,
        description="Content distribution (e.g., 40% drama, 30% comedy)"
    )
    special_focus: list[str] = Field(
        default_factory=list,
        description="Areas of emphasis"
    )


class SpecialEvent(BaseModel):
    """Calendar event impacting programming."""
    
    date: str = Field(..., description="Date or date range")
    event_name: str = Field(..., description="Event name")
    recommendation: str = Field(..., description="How to schedule around this")


class QuarterlyStrategy(BaseModel):
    """High-level quarterly programming strategy."""
    
    quarter: str = Field(..., description="Quarter identifier")
    overall_theme: str = Field(..., description="Main seasonal theme")
    reasoning: str = Field(..., description="Why this theme")
    key_decisions: list[str] = Field(
        default_factory=list,
        description="5-10 strategic decisions"
    )
    channel_strategies: list[ChannelStrategyAdjustment] = Field(
        ...,
        description="Per-channel strategy"
    )
    special_events: list[SpecialEvent] = Field(
        default_factory=list,
        description="Calendar events"
    )
    implied_monthly_themes: dict[str, str] = Field(
        default_factory=dict,
        description="Suggested monthly sub-themes"
    )


class QuarterlyStrategyRequest(BaseModel):
    """Request to generate quarterly strategy."""
    
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(
        ...,
        description="Quarter"
    )
    year: int = Field(
        ...,
        ge=2024,
        le=2030,
        description="Year"
    )
    channels: list[ChannelContext] = Field(
        ...,
        description="Channels to schedule"
    )
    media_candidates: MediaCandidateSummary = Field(
        ...,
        description="Available media"
    )
    strategic_guidance: str | None = Field(
        None,
        description="Optional strategic direction"
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field(
        "balanced",
        description="Cost vs quality"
    )


class QuarterlyStrategyResponse(BaseModel):
    """Response from quarterly strategy generation."""
    
    strategy_id: str = Field(..., description="Unique ID for auditing")
    status: Literal["success", "partial", "error"] = Field(...)
    strategy: QuarterlyStrategy = Field(..., description="The strategy")
    cost_estimate: CostEstimate = Field(..., description="Cost estimate")
    suggested_next_steps: list[str] = Field(
        default_factory=list,
        description="Recommended next actions"
    )



class ErrorResponse(BaseModel):
    """Structured error response."""
    
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    details: dict | None = Field(None, description="Additional context")
    suggested_action: str | None = Field(None, description="What to do")


# ============================================================================
# Monthly Strategy Models (Phase 2)
# ============================================================================


class TimeBlockRecommendation(BaseModel):
    """Recommended content for a time block."""
    
    time_block: Literal["early_morning", "morning", "afternoon", "prime", "late_night"] = Field(
        ..., description="Time block identifier"
    )
    time_range: str = Field(..., description="e.g., '06:00-09:00' or 'Mon-Fri 09:00-12:00'")
    recommended_content: str = Field(..., description="Type of content for this block")
    content_mix: dict[str, str] = Field(
        default_factory=dict,
        description="Genre/type breakdown (e.g., 60% comedy, 40% sitcom)"
    )
    rationale: str = Field(..., description="Why this mix works for this time block")


class MonthlyTheme(BaseModel):
    """Monthly programming strategy."""
    
    month: str = Field(..., description="Month identifier (YYYY-MM)")
    theme_name: str = Field(..., description="1-2 sentence theme name")
    theme_description: str = Field(..., description="3-5 sentences detailed description")
    key_focus_areas: list[str] = Field(
        default_factory=list,
        description="3-5 strategic focus areas for the month"
    )
    time_block_recommendations: list[TimeBlockRecommendation] = Field(
        default_factory=list,
        description="Content recommendations per time block"
    )
    opening_tagline: str = Field(..., description="Short promotional tagline for the month")
    special_notes: str = Field(
        default="",
        description="Any special considerations or events impacting programming"
    )


class MonthlyStrategyRequest(BaseModel):
    """Request to generate monthly strategy."""
    
    month: str = Field(
        ...,
        description="Month identifier (YYYY-MM format, e.g., '2026-10')"
    )
    channels: list[ChannelContext] = Field(
        ...,
        description="Channels to schedule (copied from quarterly if available)"
    )
    quarterly_context: QuarterlyStrategy | None = Field(
        None,
        description="Optional quarterly strategy for context (used to derive focus)"
    )
    media_candidates: MediaCandidateSummary = Field(
        ...,
        description="Available media for this month"
    )
    strategic_guidance: str | None = Field(
        None,
        description="Month-specific strategic direction"
    )
    max_iterations: int = Field(
        8,
        ge=3,
        le=15,
        description="Max agent iterations (default 8 for convergence)"
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field(
        "balanced",
        description="Cost vs quality"
    )


class MonthlyStrategyAgentIteration(BaseModel):
    """Record of a single agent iteration."""
    
    iteration_number: int = Field(..., description="1-indexed iteration number")
    strategy: MonthlyTheme = Field(..., description="Strategy at this iteration")
    validation_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Validation score (0.0-1.0, higher is better)"
    )
    feedback: str = Field(
        default="",
        description="LLM feedback on what to refine in next iteration"
    )
    is_converged: bool = Field(
        False,
        description="True if strategy meets convergence threshold"
    )


class MonthlyStrategyResponse(BaseModel):
    """Response from monthly strategy generation."""
    
    strategy_id: str = Field(..., description="Unique ID for auditing")
    status: Literal["success", "partial", "error"] = Field(...)
    strategy: MonthlyTheme = Field(..., description="Final converged monthly strategy")
    iteration_count: int = Field(..., description="Total iterations to converge")
    convergence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final validation score"
    )
    iterations_history: list[MonthlyStrategyAgentIteration] = Field(
        default_factory=list,
        description="Full history of all iterations (for debugging)"
    )
    cost_estimate: CostEstimate = Field(..., description="Cost for all LLM calls")
    suggested_next_steps: list[str] = Field(
        default_factory=list,
        description="Recommended next actions"
    )


# ============================================================================
# Quarterly Grid Proposal (Phase 4)
# ============================================================================


class QuarterlyGridRequest(BaseModel):
    """Request to propose one channel's frozen quarterly grid.

    Per-channel by design: Tunarr Scheduler loops channels and calls this once
    each, so every request stays small and bounded. Tunabrain runs two internal
    passes (dayparting skeleton, then strip-fill per daypart) against the
    ``catalog_profile`` — it never sees raw media.
    """

    channel: ChannelContext = Field(..., description="Channel to author a grid for")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(..., description="Quarter")
    year: int = Field(..., ge=2024, le=2030)
    catalog_profile: CatalogProfile = Field(
        ..., description="The shape of available media for this channel"
    )
    quarterly_theme: str | None = Field(
        None,
        description="Optional creative theme from the quarterly-strategy endpoint, for coherence",
    )
    strategic_guidance: str | None = Field(
        None, description="Optional channel-specific direction"
    )
    broadcast_day_start: str = Field(
        "06:00", description="Wall-clock start of the programmable day ('HH:MM')"
    )
    default_media_id: str | None = Field(
        None,
        description="Optional fallback media_id (e.g. 'random:sitcom') to fill uncovered time",
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class QuarterlyGridResponse(BaseModel):
    """Response carrying the proposed grid plus its dayparting skeleton."""

    grid_id: str = Field(..., description="Unique id for auditing")
    status: Literal["success", "partial", "error"] = Field(...)
    grid: Grid = Field(..., description="The proposed frozen grid for this channel")
    skeleton: DaypartSkeleton = Field(..., description="The Pass-A dayparting it was filled from")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues (e.g. a daypart returned no strips)",
    )
    cost_estimate: CostEstimate = Field(...)
    suggested_next_steps: list[str] = Field(default_factory=list)


# ============================================================================
# Split-round-trip grid proposal (DURATION_AWARE_SCHEDULING.md §4.3, Option A)
#
# propose-quarterly-grid above runs both passes in one call, so Tunarr
# Scheduler never sees daypart bounds until the whole grid comes back — too
# late to hand Pass B a duration-feasible candidate menu tiled to each
# block's REAL bounds. These two endpoints split proposal into two round
# trips: Pass A alone, then Pass B once per daypart with that daypart's
# candidates attached. propose-quarterly-grid is unchanged and still valid
# for callers that don't need the candidate menu.
# ============================================================================


class DaypartSkeletonRequest(BaseModel):
    """Pass A only: propose the coarse dayparting for a channel."""

    channel: ChannelContext = Field(..., description="Channel to author a grid for")
    catalog_profile: CatalogProfile = Field(
        ..., description="The shape of available media for this channel"
    )
    quarterly_theme: str | None = Field(
        None,
        description="Optional creative theme from the quarterly-strategy endpoint, for coherence",
    )
    strategic_guidance: str | None = Field(
        None, description="Optional channel-specific direction"
    )
    broadcast_day_start: str = Field(
        "06:00", description="Wall-clock start of the programmable day ('HH:MM')"
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class DaypartSkeletonResponse(BaseModel):
    """Response carrying just the Pass-A dayparting — no strips yet."""

    skeleton: DaypartSkeleton = Field(...)
    cost_estimate: CostEstimate = Field(...)


class StripFillRequest(BaseModel):
    """Pass B for ONE daypart block, against a precomputed candidate menu.

    `candidates` is built by Tunarr Scheduler's scheduling/candidates.clj from
    the catalog's per-tag runtime histogram and this exact block's bounds —
    never invented here. `prior_strips` carries every strip chosen for
    earlier blocks in the same grid, for cross-daypart coherence (same role
    `propose_quarterly_grid`'s internal loop already plays).
    """

    channel: ChannelContext = Field(...)
    catalog_profile: CatalogProfile = Field(...)
    block: DaypartBlock = Field(..., description="The daypart to fill, with its real bounds")
    candidates: list[DaypartCandidate] = Field(
        default_factory=list,
        description="Duration-feasible slot-tiling options for this block; empty is valid "
        "(falls back to unconstrained strip-fill, same as propose_quarterly_grid today)",
    )
    prior_strips: list[GridStrip] = Field(
        default_factory=list, description="Strips already chosen for earlier daypart blocks"
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class StripFillResponse(BaseModel):
    """Response carrying the strips filled for one daypart block."""

    strips: list[GridStrip] = Field(default_factory=list)
    cost_estimate: CostEstimate = Field(...)


class QuarterlyGridRepairRequest(BaseModel):
    """Request to repair an existing grid against feasibility findings.

    Drives the propose -> check -> repair loop: Tunarr runs the deterministic
    feasibility checker, and feeds any shortfalls back here for a targeted fix.
    Only the flagged strips should change; the rest of the grid stays put.
    """

    channel: ChannelContext = Field(...)
    catalog_profile: CatalogProfile = Field(...)
    current_grid: Grid = Field(..., description="The grid that failed feasibility")
    feasibility_report: FeasibilityReport = Field(
        ..., description="Deterministic findings to address"
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class QuarterlyGridRepairResponse(BaseModel):
    """Response carrying the revised grid."""

    grid_id: str = Field(...)
    status: Literal["success", "partial", "error"] = Field(...)
    grid: Grid = Field(..., description="The revised grid")
    changes: list[str] = Field(
        default_factory=list, description="Human-readable summary of what was changed"
    )
    cost_estimate: CostEstimate = Field(...)


# ============================================================================
# Schedule Review / critique loop (Phase 7)
#
# The feasibility checker is deterministic and only sees structure (capacity,
# overlaps, coverage). It can't judge *taste* — whether a concrete week is
# repetitive, ignores its own daypart intent, leans on generic random pools
# when named series are available, or just reads as bad TV. That judgement is
# exactly what a cheap LLM does well when it's handed a CONCRETE realized week
# (real show titles in real slots) rather than an abstract catalog summary.
#
# review-grid critiques; revise-grid-from-review applies the critique. Tunarr
# Scheduler drives the loop (expand a sample week -> review -> revise on fail,
# bounded) exactly as it already drives the propose -> feasibility -> repair
# loop, but for taste instead of capacity.
# ============================================================================


class ReviewSlot(BaseModel):
    """One concrete slot in the sample week handed to the reviewer.

    This is a realized view — a specific weekday and time with the actual show
    (or pool) that lands there — as opposed to the abstract ``GridStrip`` rule
    it came from. Tunarr Scheduler produces it by expanding one representative
    week of the frozen grid and labelling each slot with its show title, so the
    reviewer sees "Mon 17:00-17:30 Seinfeld" rather than "series:42".
    """

    day: Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"] = Field(
        ..., description="Weekday this slot airs on"
    )
    start: str = Field(..., description="Start time 'HH:MM' (24h)")
    end: str = Field(..., description="End time 'HH:MM'; end <= start wraps past midnight")
    label: str = Field(
        ..., description="Human-readable content, e.g. 'Seinfeld' or 'random: comedy pool'"
    )
    media_id: str = Field(..., description="The underlying media_id (series:/movie:/random:)")
    strategy: SelectionStrategy = Field(
        "sequential", description="How content rotates within the slot"
    )
    daypart: str | None = Field(
        None, description="Daypart this slot falls in, if known (links back to the skeleton)"
    )


class ReviewFinding(BaseModel):
    """One taste problem the reviewer found in the sample week."""

    aspect: Literal[
        "variety", "daypart-fit", "genericness", "series-usage", "pacing", "coherence", "other"
    ] = Field(..., description="Which quality dimension this finding is about")
    severity: Literal["minor", "major"] = Field(
        ..., description="'major' findings are what fail the review; 'minor' are nice-to-fix"
    )
    message: str = Field(..., description="What's wrong, concretely and actionably")
    target: str | None = Field(
        None, description="Daypart name or strip_id this finding is about, if specific"
    )


class ScheduleReviewRequest(BaseModel):
    """Ask the reviewer to critique a concrete realized week against its plan."""

    channel: ChannelContext = Field(...)
    skeleton: DaypartSkeleton | None = Field(
        None, description="The dayparting plan (roles/intent) the grid was filled from"
    )
    grid: Grid = Field(..., description="The frozen grid whose realization is under review")
    sample_week: list[ReviewSlot] = Field(
        default_factory=list,
        description="A representative week expanded from the grid, labelled with real titles",
    )
    catalog_profile: CatalogProfile | None = Field(
        None,
        description="Optional catalog shape, so the reviewer can flag under-used named series",
    )
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class ScheduleReview(BaseModel):
    """The reviewer's verdict on one realized week."""

    verdict: Literal["pass", "fail"] = Field(
        ..., description="'fail' when any 'major' finding is present"
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall taste quality, 0 (bad TV) to 1 (great)"
    )
    summary: str = Field(..., description="One-paragraph overall assessment")
    findings: list[ReviewFinding] = Field(default_factory=list)


class ScheduleReviewResponse(BaseModel):
    """Response carrying the review verdict."""

    review_id: str = Field(..., description="Unique id for auditing")
    status: Literal["success", "error"] = Field(...)
    review: ScheduleReview = Field(...)
    cost_estimate: CostEstimate = Field(...)


class ReviewReviseRequest(BaseModel):
    """Apply a failed review's findings to the grid (the revise half of the loop).

    Mirrors QuarterlyGridRepairRequest, but the feedback is *taste* findings
    from a ScheduleReview rather than deterministic feasibility findings. Only
    the strips implicated by the findings should change; the rest stays put.
    """

    channel: ChannelContext = Field(...)
    catalog_profile: CatalogProfile = Field(...)
    current_grid: Grid = Field(..., description="The grid whose realized week the review failed")
    review: ScheduleReview = Field(..., description="The critique to act on")
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class ReviewReviseResponse(BaseModel):
    """Response carrying the taste-revised grid (same shape as a repair)."""

    grid_id: str = Field(...)
    status: Literal["success", "partial", "error"] = Field(...)
    grid: Grid = Field(..., description="The revised grid")
    changes: list[str] = Field(
        default_factory=list, description="Human-readable summary of what was changed"
    )
    cost_estimate: CostEstimate = Field(...)


# ============================================================================
# Monthly Overrides (Phase 6)
# ============================================================================


class MonthlyOverridesRequest(BaseModel):
    """Request to propose sparse monthly overrides over a frozen grid.

    Per channel-month. The grid is supplied as *context* so the LLM proposes only
    deltas, never a re-authored schedule.
    """

    channel: ChannelContext = Field(...)
    month: str = Field(..., description="Month identifier, 'YYYY-MM'")
    grid: Grid = Field(..., description="The frozen weekly grid this month layers over")
    catalog_profile: CatalogProfile = Field(
        ..., description="Available media, for choosing special-event content"
    )
    monthly_theme: str | None = Field(
        None, description="Optional monthly theme for coherence"
    )
    planned_events: list[str] = Field(
        default_factory=list,
        description="Operator-supplied events/requests (e.g. 'Cheers marathon Sat the 10th')",
    )
    strategic_guidance: str | None = Field(None, description="Optional month-specific direction")
    cost_tier: Literal["economy", "balanced", "premium"] = Field("balanced")


class MonthlyOverridesResponse(BaseModel):
    """Response carrying the sparse override deltas for the month."""

    overrides_id: str = Field(..., description="Unique id for auditing")
    status: Literal["success", "partial", "error"] = Field(...)
    month: str = Field(...)
    overrides: list[Override] = Field(
        default_factory=list, description="Sparse exceptions (may be empty)"
    )
    warnings: list[str] = Field(default_factory=list)
    cost_estimate: CostEstimate = Field(...)
    suggested_next_steps: list[str] = Field(default_factory=list)


# ============================================================================
# Grout enrichment: /enrich/short-form and /enrich/long-form
# ============================================================================
#
# Two orchestrated endpoints layered on top of the existing /categorize + /tags
# building blocks. They exist so Grout (bulk, uncategorized media that doesn't
# fit Jellyfin's film/show paradigm) can get the same scheduling metadata in a
# single round trip. Short-form wraps categorize+tags directly; long-form first
# runs STT (and optional keyframe captioning) to synthesise grounding context
# for media that carries no reliable external metadata.
#
# These are additive: they reuse MediaItem, MediaContext, Channel,
# CategoryDefinition, DimensionSelection, and CostEstimate verbatim and never
# mutate the existing categorize/tag schemas.


class EnrichShortFormRequest(BaseModel):
    """Request to enrich short-form media (bumpers, fillers, idents, ads, music videos).

    This is an orchestration over the existing /categorize + /tags endpoints.
    Short-form media has no audio of consequence to transcribe; the only
    available signals are filename, duration, and (optionally) operator-supplied
    context. Use this when duration_seconds < 600 and the media has no
    substantial dialogue track.
    """

    media: MediaItem = Field(..., description="The media item to enrich")
    categories: dict[str, CategoryDefinition] = Field(
        default_factory=dict,
        description="Operator-supplied dimension catalog, forwarded verbatim to /categorize",
    )
    existing_tags: list[str] = Field(
        default_factory=list, description="Pre-existing free-form tags to reuse when tagging"
    )
    context: MediaContext | None = Field(
        None, description="Optional operator-supplied grounding, propagated to categorize and tags"
    )
    channels: list[Channel] = Field(
        default_factory=list, description="Optional channels passed through to /categorize"
    )
    debug: bool = Field(
        False, description="Enable debug logging for outgoing LLM and downstream service calls"
    )


class EnrichShortFormResponse(BaseModel):
    """Combined enrichment result for short-form media."""

    media: MediaItem = Field(..., description="The media item that was enriched (echoed back)")
    describe: DescribeMedia | None = Field(
        None,
        description=(
            "Refined display title and short description from the /enrich/describe "
            "step. Null only if that step failed (see warnings)."
        ),
    )
    dimensions: list[DimensionSelection] = Field(
        default_factory=list, description="Structured dimension selections from /categorize"
    )
    tags: list[str] = Field(default_factory=list, description="Free-form tags from /tags")
    context: MediaContext | None = Field(
        None, description="Resolved grounding context actually used, echoed back for storage"
    )
    cost_estimate: CostEstimate = Field(..., description="Estimated LLM cost for this enrichment")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues (e.g. categorize or tags degraded to a partial result)",
    )


class MediaSource(BaseModel):
    """How to obtain the media for processing. Exactly one of url/file_id must be set."""

    url: str | None = Field(
        None, description="HTTP(S) URL to fetch the media from (e.g. YouTube, S3, etc.)"
    )
    file_id: str | None = Field(
        None,
        description=(
            "ID of a media file already staged in the cluster's shared scratch space "
            "(path constructed via the TUNABRAIN_SCRATCH_DIR env var)"
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> MediaSource:
        if bool(self.url) == bool(self.file_id):
            raise ValueError("MediaSource requires exactly one of 'url' or 'file_id'")
        return self


class EnrichLongFormOptions(BaseModel):
    """Per-call knobs for the long-form enrichment pipeline."""

    stt_backend: Literal["whisper-http", "subgen", "auto"] = Field(
        "auto",
        description="Which STT service to use. 'auto' probes both and uses the one that responds first.",
    )
    enable_keyframe_analysis: bool = Field(
        True,
        description="Extract evenly-spaced keyframes and include their captions in the context",
    )
    keyframe_count: int = Field(
        5, ge=1, le=20, description="Number of evenly-spaced keyframes to extract and caption"
    )
    max_transcript_chars: int = Field(
        8000,
        description=(
            "Cap the transcript length sent to the LLM (the full transcript is always "
            "returned in the top-level 'transcript' field regardless of this cap)"
        ),
    )
    stt_timeout_seconds: int = Field(
        600, ge=10, le=3600, description="Per-request timeout for the STT backend call"
    )
    skip_stt_below_seconds: int = Field(
        30, ge=0, description="Skip STT entirely if the media duration is below this many seconds"
    )


class EnrichLongFormRequest(BaseModel):
    """Request to enrich long-form media (documentaries, video essays, YouTube series).

    Tunabrain owns the heavy lifting:
      1. Pull the media from the provided source (URL or pre-staged scratch path)
      2. Extract the audio track
      3. Run STT against the cluster's STT service (pluggable; defaults to auto)
      4. Optionally extract a small set of keyframes and caption them
      5. Combine transcript + keyframe captions as the grounding context
      6. Run /categorize + /tags with the resolved context

    Use this when duration_seconds >= 600 OR the media has a substantial dialogue
    track OR you need grounding from the actual audio/video content.
    """

    media: MediaItem = Field(..., description="The media item to enrich")
    source: MediaSource = Field(..., description="Where to obtain the media bytes")
    categories: dict[str, CategoryDefinition] = Field(
        default_factory=dict,
        description="Operator-supplied dimension catalog, forwarded verbatim to /categorize",
    )
    existing_tags: list[str] = Field(
        default_factory=list, description="Pre-existing free-form tags to reuse when tagging"
    )
    channels: list[Channel] = Field(
        default_factory=list, description="Optional channels passed through to /categorize"
    )
    options: EnrichLongFormOptions = Field(
        default_factory=EnrichLongFormOptions, description="Per-call pipeline knobs"
    )
    debug: bool = Field(
        False, description="Enable debug logging for outgoing LLM and downstream service calls"
    )


class PipelineStageResult(BaseModel):
    """Per-stage status for the long-form pipeline, with timing and diagnostics."""

    stage: Literal[
        "fetch", "extract_audio", "stt", "keyframes", "categorize", "tags", "describe"
    ] = Field(..., description="Which pipeline stage this result describes")
    status: Literal["success", "skipped", "warning", "failed"] = Field(
        ..., description="Outcome of the stage"
    )
    duration_seconds: float = Field(..., description="Wall-clock time spent in this stage")
    backend: str | None = Field(
        None, description="For the STT stage: the backend actually used ('whisper-http' or 'subgen')"
    )
    detail: str | None = Field(None, description="Optional human-readable detail or error message")


class EnrichLongFormResponse(BaseModel):
    """Combined enrichment result for long-form media."""

    media: MediaItem = Field(..., description="The media item that was enriched (echoed back)")
    describe: DescribeMedia | None = Field(
        None,
        description=(
            "Refined display title and short description from the /enrich/describe "
            "step. Null only if that step failed (see warnings)."
        ),
    )
    dimensions: list[DimensionSelection] = Field(
        default_factory=list, description="Structured dimension selections from /categorize"
    )
    tags: list[str] = Field(default_factory=list, description="Free-form tags from /tags")
    transcript: str = Field(
        "", description="Full STT transcript (may be empty if STT was skipped or failed)"
    )
    keyframe_captions: list[str] = Field(
        default_factory=list, description="Captions for extracted keyframes, in temporal order"
    )
    context: MediaContext | None = Field(
        None, description="Resolved grounding context (transcript + keyframe captions)"
    )
    pipeline_stages: list[PipelineStageResult] = Field(
        default_factory=list, description="Per-stage status with timing and warnings"
    )
    cost_estimate: CostEstimate = Field(..., description="Estimated LLM cost for this enrichment")
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal issues encountered across the pipeline"
    )


# ============================================================================
# Grout enrichment: /enrich/describe
# ============================================================================
#
# A small, single-purpose endpoint that derives a display-ready title and a
# short description from a media item that already carries a rough working
# title (typically a filename, an on-disk path, or the literal "Unknown"). It
# is the first step /enrich/short-form and /enrich/long-form orchestrate
# internally, exposed publicly so callers can request describe-only enrichment
# without paying for a full tag/dimension pass. Like the other enrichment
# endpoints it reuses MediaItem, MediaContext, and CostEstimate verbatim.


class EnrichDescribeRequest(BaseModel):
    """Request to derive a clean title and short description for a media item.

    The caller is responsible for providing a working ``title`` (typically
    derived from the filename, the on-disk path, or a human-set value). The
    model refines the title and produces a one-sentence description.

    This endpoint is the public version of what ``/enrich/short-form`` and
    ``/enrich/long-form`` orchestrate internally as the first step. It is
    exposed publicly so other callers (e.g. Marquee's bulk-import UI) can
    request describe-only enrichment without paying the cost of a full
    tag/dimension pass.
    """

    media: MediaItem = Field(
        ...,
        description=(
            "The media item to describe. ``media.title`` must be a non-empty "
            "string (use the filename, the on-disk path, or the literal "
            "'Unknown' when nothing else is available)."
        ),
    )
    context: MediaContext | None = Field(
        None,
        description=(
            "Optional grounding context. If supplied, the model uses it "
            "instead of the Wikipedia auto-search. See /tags and /categorize "
            "for the full MediaContext contract."
        ),
    )
    debug: bool = Field(
        False,
        description="Enable debug logging for the LLM call.",
    )

    @model_validator(mode="after")
    def _title_must_be_non_empty(self) -> EnrichDescribeRequest:
        # The endpoint refines whatever title it is given but never invents one
        # from nothing, so an empty/whitespace-only title is a request error
        # (422) rather than something the model is asked to paper over.
        if not self.media.title or not self.media.title.strip():
            raise ValueError("media.title must be a non-empty string")
        return self


class DescribeMedia(BaseModel):
    """The describe-only media result. Subset of MediaItem.

    The full ``MediaItem`` is overkill for a describe response — the caller
    already has the original item and is only interested in the two fields
    the model can fill. Keeping this small avoids the impression that the
    model rewrote the rest of the row.
    """

    id: str = Field(..., description="Echoed from the request.")
    title: str = Field(..., description="The refined title.")
    description: str | None = Field(
        None,
        description=(
            "A one-sentence description. May be null when the model judges "
            "a description would be noise (e.g. a 5-second bumper)."
        ),
    )


class EnrichDescribeResponse(BaseModel):
    """Describe-only enrichment response."""

    media: DescribeMedia = Field(
        ...,
        description=(
            "The refined media. ``id`` is echoed; ``title`` and ``description`` "
            "are the model output."
        ),
    )
    context: MediaContext = Field(
        default_factory=MediaContext,
        description=(
            "Resolved grounding context actually used. Echoed back so the "
            "caller can store and correct it (same pattern as /tags and "
            "/categorize)."
        ),
    )
    cost_estimate: CostEstimate = Field(
        ...,
        description="Cost estimate for the LLM call(s) made.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues (e.g. description was filtered).",
    )


# ============================================================================
# Grout enrichment: /enrich/profile (directory / tag-group profiling)
# ============================================================================
#
# A single-call endpoint that derives a shared scheduling profile (dimensions +
# tags) for a *group* of related media, from the group's name and a small
# sample of its filenames. It exists so a large, well-organised library (e.g.
# 200k YouTube pulls laid out one directory per channel) can be enriched at the
# group level — one LLM call per directory — instead of paying for a per-file
# /enrich/short-form call across the whole corpus. Every child item then
# inherits the group's profile downstream (in Grout).
#
# Unlike /categorize and /enrich/short-form, this endpoint sees NO single media
# item: filenames are the only content signal, and the prompt is explicit that
# the model must not hallucinate beyond what the names support. It reuses
# CostEstimate verbatim and never touches the existing enrichment schemas.


class EnrichProfileRequest(BaseModel):
    """Request to derive a shared metadata profile for a group of related media.

    The group is identified by a human-readable ``concept_name`` (typically a
    channel / creator / series / directory name) and grounded on a small
    ``sample_filenames`` list. There is no per-item ``MediaItem`` — the whole
    point is to classify the group once and let the caller fan the result out
    to every member.
    """

    concept_name: str = Field(
        ...,
        description=(
            "Human-readable name of the group, e.g. a channel/creator/series/"
            "directory name ('Adam Neely Music'). This is the primary signal."
        ),
    )
    sample_filenames: list[str] = Field(
        default_factory=list,
        description=(
            "A sample of filenames drawn from the group. The only per-item "
            "content signal the model gets; keep it small (5 is typical)."
        ),
    )
    sample_count: int = Field(
        5,
        ge=1,
        description=(
            "The number of filenames the caller intended to sample. Informational "
            "— the prompt grounds on whatever ``sample_filenames`` actually holds."
        ),
    )
    categories: dict[str, CategoryDefinition] = Field(
        default_factory=dict,
        description=(
            "Dimensions with descriptions and allowed values, same shape as "
            "CategorizationRequest.categories. When supplied, the model is "
            "constrained to these dimensions and their candidate values only "
            "— every listed dimension gets at least one value, and any value "
            "outside the candidates is dropped. When omitted, the model "
            "proposes values freely across the fixed dimension keys (channel, "
            "audience, freshness, season, time-slot), matching pre-v1.1 "
            "behavior."
        ),
    )
    debug: bool = Field(
        False, description="Enable debug logging for the outgoing LLM call."
    )

    @model_validator(mode="after")
    def _sample_filenames_non_empty(self) -> EnrichProfileRequest:
        # Filenames are the only content signal; with none, the model would be
        # guessing purely from the concept name. Reject rather than paper over
        # it (surfaces as a 422 at the route), matching /enrich/describe's
        # empty-title contract.
        if not self.sample_filenames:
            raise ValueError("sample_filenames must not be empty")
        return self


class EnrichProfileResponse(BaseModel):
    """Shared profile derived for a media group.

    ``dimensions`` is a plain ``{dimension: [values]}`` map (not the richer
    ``DimensionSelection`` list) because the caller fans it out verbatim to a
    whole directory of items — the per-value ``notes`` that ``/categorize``
    returns would be meaningless applied group-wide.
    """

    concept_name: str = Field(..., description="Echoed from the request.")
    dimensions: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Derived dimension selections, e.g. {'channel': ['muse'], "
            "'audience': ['adult']}. Only confident dimensions are set; unclear "
            "ones are omitted."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="3-7 short lowercase-hyphenated free-form tags describing the group.",
    )
    grounding_source: str = Field(
        "filename-pattern",
        description="Where the profile was grounded. Always 'filename-pattern' in v1.",
    )
    cost_estimate: CostEstimate = Field(
        ..., description="Estimated LLM cost for the single profiling call."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues (e.g. the model output degraded to a partial result).",
    )
