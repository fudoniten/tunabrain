"""Grid-based scheduling contracts (Phase 0).

These models define the integration boundaries between the three systems:

- **Pseudovision** (media server / playout) produces the raw aggregates behind
  ``CatalogProfile`` and ultimately consumes the expanded ``DailySlot`` stream.
- **Tunarr Scheduler** (Clojure control plane) stores the frozen ``Grid`` and the
  sparse ``Override`` list, runs the deterministic expander
  (:mod:`tunabrain.scheduling.expander`), and runs the feasibility checker that
  emits ``FeasibilityReport``.
- **Tunabrain** (this service, stateless) only ever *proposes*: it receives a
  ``CatalogProfile`` (and, for repairs, a ``FeasibilityReport``) and returns a
  ``Grid`` or ``Override`` list. It never persists anything and never sees raw
  media.

The core invariant the whole design rests on:

    A week's structure is a pure, deterministic function of
    (frozen grid, sparse overrides, dates). The same inputs always produce the
    same slots. Nothing re-authors the schedule week to week.

See ``docs/scheduling-grid-spec.md`` for the precedence/expansion algorithm.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

# ---------------------------------------------------------------------------
# Wire-contract base model
# ---------------------------------------------------------------------------


class _WireModel(BaseModel):
    """Base for every model that crosses the Tunabrain -> Tunarr boundary.

    Serialization omits any key whose value is ``None`` so the JSON carries only
    the fields that actually apply. This matters most for ``OverrideScope``,
    which the scheduler models as a *closed* discriminated union: a single-date
    scope that also serialized ``days``/``effective_*`` as explicit ``null``
    would be rejected as carrying disallowed keys. Dropping nulls keeps each
    payload matching exactly one branch of that union, and the same discipline
    keeps every other optional-heavy contract (``Content.label``, grid strips,
    ...) free of stray ``null`` siblings.

    We drop ``None`` specifically (equivalent to ``exclude_none=True``), not
    unset fields: a field that is legitimately *set to* ``None`` (e.g. an
    optional ``effective_end``) should still vanish from the wire rather than
    reappear as ``null``.
    """

    @model_serializer(mode="wrap")
    def _drop_none(self, handler: Any) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
"""Lower-case three-letter weekday codes. Monday is the canonical week start."""

DayGroup = Literal["daily", "weekdays", "weekends"]
"""Named convenience groups. ``weekdays`` = mon-fri, ``weekends`` = sat-sun."""

# A day pattern is either an explicit list of weekdays or a named group.
DayPattern = list[DayOfWeek] | DayGroup

SelectionStrategy = Literal["random", "sequential", "specific"]
"""How Pseudovision picks concrete content *within* a slot. Mirrors
``DailySlot.media_selection_strategy`` — structural placement is frozen by the
grid, but the fill (which episode) may still rotate at air time."""


class Content(_WireModel):
    """What airs in a slot. Maps directly onto fields of ``DailySlot``.

    ``media_id`` follows the existing convention: ``"series:<id>"``,
    ``"movie:<id>"``, or ``"random:<category>"``. Episode resolution happens
    downstream in Pseudovision; the grid only references the *intent*.
    """

    media_id: str = Field(
        ...,
        description="Media identifier: 'random:category', 'series:show-id', or 'movie:movie-id'",
    )
    strategy: SelectionStrategy = Field(
        "sequential",
        description="How Pseudovision selects content within the slot",
    )
    marathon: bool = Field(
        False,
        description="Hint that this block is a single long run of one show (affects fill, not structure)",
    )
    category_filters: list[str] = Field(
        default_factory=list,
        description="Category tags to constrain content (e.g., ['comedy', 'sitcom'])",
    )
    label: str | None = Field(
        None, description="Human-readable label for GUI display (e.g., 'Seinfeld at Five')"
    )
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. CatalogProfile  (Pseudovision -> Tunarr -> Tunabrain)
# ---------------------------------------------------------------------------


class ShowProfile(_WireModel):
    """Per-show rollup. The unit the LLM authors strips against."""

    media_id: str = Field(..., description="e.g. 'series:seinfeld'")
    title: str
    genres: list[str] = Field(default_factory=list)
    episode_count: int = Field(..., ge=0, description="Total episodes in library")
    available_episode_count: int = Field(
        ..., ge=0, description="Episodes eligible to air (e.g. unwatched / in-window)"
    )
    avg_runtime_minutes: float | None = Field(
        None, description="Average episode runtime, for capacity math"
    )
    tags: list[str] = Field(default_factory=list)


class GenreProfile(_WireModel):
    """Per-genre rollup, for high-level dayparting decisions."""

    genre: str
    show_count: int = Field(..., ge=0)
    episode_count: int = Field(..., ge=0)


class RuntimeBucket(_WireModel):
    """One bar of the runtime histogram (helps fit content to slot lengths)."""

    label: str = Field(..., description="e.g. '20-30min'")
    min_minutes: int = Field(..., ge=0)
    max_minutes: int | None = Field(
        None, description="None for the open-ended top bucket (e.g. '210+min')"
    )
    item_count: int = Field(..., ge=0)


class TagAggregate(_WireModel):
    """Per-tag rollup in the dimension model (e.g. 'genre:comedy',
    'channel:goldenreels') — the generalization of the deprecated ``genres``
    field to any dimension, not just genre."""

    tag: str
    show_count: int = Field(..., ge=0)
    episode_count: int = Field(..., ge=0)


class TagRuntimeHistogram(_WireModel):
    """Runtime distribution for one tag (e.g. 'genre:movie', 'genre:sitcom'),
    for slot-fit reasoning within a specific ``random:<category>`` pool rather
    than the catalog as a whole. A movie pool and a sitcom pool can have wildly
    different runtime distributions — ``CatalogProfile.runtime_histogram``
    alone can't answer "does *this* category have content at *this* length"."""

    tag: str
    buckets: list[RuntimeBucket] = Field(default_factory=list)


class CatalogProfile(_WireModel):
    """The *shape* of the library — never the library itself.

    Built deterministically by Tunarr Scheduler from Pseudovision aggregates.
    Sized the same whether the library has 5,000 items or 5,000,000. This is the
    only view of the media catalog Tunabrain's LLM ever sees.
    """

    channel_scope: str | None = Field(
        None, description="Channel this profile is sliced for, if any"
    )
    total_items: int = Field(..., ge=0)
    total_episodes: int = Field(..., ge=0)
    movie_count: int = Field(0, ge=0)
    shows: list[ShowProfile] = Field(default_factory=list)
    genres: list[GenreProfile] = Field(default_factory=list)
    tag_aggregates: list[TagAggregate] = Field(default_factory=list)
    runtime_histogram: list[RuntimeBucket] = Field(default_factory=list)
    tag_runtime_histograms: list[TagRuntimeHistogram] = Field(default_factory=list)
    generated_at: datetime | None = Field(
        None, description="When Pseudovision produced the underlying aggregate"
    )


# ---------------------------------------------------------------------------
# 2. GridStrip / DaypartSkeleton / Grid  (Tunabrain -> Tunarr, stored)
# ---------------------------------------------------------------------------


class GridStrip(_WireModel):
    """One recurring strip in the frozen weekly grid.

    A strip is a *rule*, not a dated slot: "weekdays 17:00-18:00 -> Seinfeld"
    is a single strip that covers every matching weekday all quarter. Times are
    24h ``"HH:MM"`` wall-clock; when ``end <= start`` the strip crosses midnight
    into the next calendar day (e.g. ``22:00`` -> ``10:00``).
    """

    strip_id: str = Field(..., description="Stable id, unique within the grid")
    days: DayPattern = Field(..., description="Which weekdays this strip applies to")
    start: str = Field(..., description="Start time, 'HH:MM' (24h)")
    end: str = Field(..., description="End time, 'HH:MM' (24h); end <= start wraps past midnight")
    content: Content
    priority: int = Field(
        0, description="Tiebreaker among equally-specific base strips; higher wins"
    )
    daypart: str | None = Field(
        None, description="Optional daypart this strip belongs to (links back to the skeleton)"
    )


class DaypartBlock(_WireModel):
    """A coarse block of the broadcast day with an assigned programming role.

    Output of Pass A (dayparting). Strips (Pass B) are filled *within* a block's
    bounds and inherit its role. Small enough to review on one screen.
    """

    name: str = Field(..., description="e.g. 'prime', 'late_night'")
    start: str = Field(..., description="Block start 'HH:MM'")
    end: str = Field(..., description="Block end 'HH:MM' (end <= start wraps past midnight)")
    role: str = Field(..., description="Programming intent, e.g. 'marquee sitcoms'")
    genre_focus: list[str] = Field(default_factory=list)
    rationale: str | None = None


class DaypartSkeleton(_WireModel):
    """The coherence-bearing top-level frame for one channel's grid (Pass A)."""

    channel: str
    blocks: list[DaypartBlock] = Field(default_factory=list)


class Grid(_WireModel):
    """A channel's complete frozen weekly grid (the base layer).

    Authored once (per quarter) and then immutable. Everything that varies
    week-to-week comes from ``Override``s layered on top, never from re-authoring
    the grid.
    """

    channel: str
    broadcast_day_start: str = Field(
        "06:00",
        description="Wall-clock start of the programmable day; used only for default-fill bounds",
    )
    skeleton: DaypartSkeleton | None = Field(
        None, description="The Pass-A dayparting this grid was filled from (for audit/GUI)"
    )
    strips: list[GridStrip] = Field(default_factory=list)
    default_content: Content | None = Field(
        None,
        description="Fallback for any time not covered by a strip (e.g. 'random:sitcom')",
    )


# ---------------------------------------------------------------------------
# 3. Override  (Tunabrain -> Tunarr, stored)
# ---------------------------------------------------------------------------


class OverrideScope(_WireModel):
    """When an override applies. Exactly one of ``date`` or ``days`` must be set.

    - ``date``: a single calendar day ("2026-01-10") — most specific.
    - ``days`` (+ optional ``effective_start``/``effective_end``): a recurring
      pattern bounded to a window, e.g. "all Fridays this month".
    """

    date: str | None = Field(None, description="Specific calendar date, 'YYYY-MM-DD'")
    days: DayPattern | None = Field(None, description="Recurring weekday pattern")
    effective_start: str | None = Field(
        None, description="Lower bound (inclusive) for recurring scope, 'YYYY-MM-DD'"
    )
    effective_end: str | None = Field(
        None, description="Upper bound (inclusive) for recurring scope, 'YYYY-MM-DD'"
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> OverrideScope:
        has_date = self.date is not None
        has_days = self.days is not None
        if has_date == has_days:
            raise ValueError("OverrideScope requires exactly one of 'date' or 'days'")
        if has_date and (self.effective_start or self.effective_end):
            raise ValueError("effective_start/end only apply to recurring 'days' scopes")
        return self


class Override(_WireModel):
    """A sparse, higher-precedence exception layered over the frozen grid.

    The monthly layer emits these as *deltas* — "Sat the 10th: Cheers marathon",
    "Fridays this month: evening movie". A month with no special plans produces
    zero overrides, and every week then materializes identically to the grid.
    """

    override_id: str = Field(..., description="Stable id, unique within its set")
    scope: OverrideScope
    start: str = Field(..., description="Start time 'HH:MM'")
    end: str = Field(..., description="End time 'HH:MM' (end <= start wraps past midnight)")
    content: Content
    mode: Literal["replace"] = Field(
        "replace",
        description="v1 supports 'replace' (carve out the base grid in this window only)",
    )
    priority: int = Field(0, description="Tiebreaker among equally-specific overrides; higher wins")
    note: str | None = None


# ---------------------------------------------------------------------------
# 4. FeasibilityReport  (Tunarr -> Tunabrain, the *true* repair feedback)
# ---------------------------------------------------------------------------


class StripFeasibility(_WireModel):
    """Capacity finding for one strip/override over the planning horizon."""

    rule_id: str = Field(..., description="strip_id or override_id this finding refers to")
    media_id: str
    slots_required: int = Field(
        ..., ge=0, description="Airings this rule produces over the horizon"
    )
    episodes_available: int = Field(..., ge=0)
    headroom_ratio: float | None = Field(
        None, description="episodes_available / slots_required (None when slots_required == 0)"
    )
    status: Literal["ok", "tight", "shortfall"] = Field(
        ..., description="'tight' = under a configured comfort margin; 'shortfall' = insufficient"
    )
    message: str = Field(default="")


class FeasibilityReport(_WireModel):
    """Deterministic validation feedback. Replaces the heuristic length-checks of
    the old monthly loop with real arithmetic the LLM can act on."""

    horizon_start: str = Field(..., description="'YYYY-MM-DD' inclusive")
    horizon_end: str = Field(..., description="'YYYY-MM-DD' exclusive")
    overall_status: Literal["ok", "warnings", "blocked"] = Field(...)
    strip_findings: list[StripFeasibility] = Field(default_factory=list)
    overlaps: list[str] = Field(
        default_factory=list, description="Human-readable overlap conflicts among same-layer rules"
    )
    uncovered_intervals: list[str] = Field(
        default_factory=list,
        description="Time ranges with no strip and no default_content",
    )
    notes: list[str] = Field(default_factory=list)
