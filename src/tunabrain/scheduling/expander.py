"""Reference deterministic grid expander (Phase 0).

This is the canonical implementation of the precedence/expansion algorithm
described in ``docs/scheduling-grid-spec.md``. In production this logic lives in
**Tunarr Scheduler (Clojure)** — this Python version exists as an executable
specification and a portability reference. Tunabrain itself never calls it at
request time; it is here so the algorithm has one unambiguous source of truth
and a test that pins its behavior.

Key invariant: ``expand(...)`` is a pure function. Identical
(grid, overrides, date range) inputs always yield byte-identical output. There
is no randomness and no LLM. "Generating the weekly schedule" is really just
projecting the frozen grid onto concrete dates.

The algorithm is interval painting with a precedence cascade:

1. Materialize every strip and override into absolute ``[start, end)`` datetime
   intervals for the requested date range (a wrap past midnight is just an
   interval whose end is the next day — no special case).
2. Sweep the day's boundary points; in each elementary interval the
   highest-precedence covering rule wins. Overrides outrank the base grid; among
   equally-layered rules the more specific scope wins, then explicit priority,
   then definition order.
3. Merge adjacent elementary intervals won by the same rule.
4. Fill any remaining gaps with the grid's ``default_content``.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from tunabrain.api.models import DailySlot
from tunabrain.scheduling.grid import (
    Content,
    DayPattern,
    Grid,
    Override,
)

# Monday=0 .. Sunday=6  ->  three-letter codes used in DayPattern
_WEEKDAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _matches_pattern(pattern: DayPattern, d: date) -> bool:
    """True if calendar date ``d``'s weekday is covered by ``pattern``."""
    code = _WEEKDAY_CODES[d.weekday()]
    if pattern == "daily":
        return True
    if pattern == "weekdays":
        return d.weekday() < 5
    if pattern == "weekends":
        return d.weekday() >= 5
    # explicit list
    return code in pattern


def _pattern_specificity(pattern: DayPattern) -> int:
    """Specificity rank for a recurring day pattern (higher = more specific).

    A specific calendar date (handled separately, rank 3) is the most specific.
    """
    if isinstance(pattern, list):
        return 2  # explicit weekday list
    if pattern in ("weekdays", "weekends"):
        return 1  # named group
    return 0  # "daily"


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(hour=int(hh), minute=int(mm))


def _interval_for(day: date, start: str, end: str) -> tuple[datetime, datetime]:
    """Absolute [start, end) for a strip on ``day``. Wraps to next day if end<=start."""
    start_t = _parse_hhmm(start)
    end_t = _parse_hhmm(end)
    abs_start = datetime.combine(day, start_t)
    abs_end = datetime.combine(day, end_t)
    if abs_end <= abs_start:
        abs_end += timedelta(days=1)
    return abs_start, abs_end


class _Candidate:
    """A materialized, dated interval competing to fill time."""

    __slots__ = ("start", "end", "content", "precedence", "rule_id")

    def __init__(
        self,
        start: datetime,
        end: datetime,
        content: Content,
        precedence: tuple,
        rule_id: str,
    ) -> None:
        self.start = start
        self.end = end
        self.content = content
        self.precedence = precedence
        self.rule_id = rule_id


def _materialize(
    grid: Grid,
    overrides: list[Override],
    range_start: date,
    range_end: date,
) -> list[_Candidate]:
    """Expand recurring rules into concrete dated candidates over [start, end).

    Precedence tuple (compared lexicographically, higher wins):
        (layer_rank, scope_specificity, priority, order_index)
    where layer_rank: base grid = 0, override = 1.
    """
    candidates: list[_Candidate] = []
    order = 0

    # We materialize one extra day before the range so a strip that *started*
    # the prior day and wraps past midnight can still cover the early hours of
    # range_start. Output is clipped to the requested window at the end.
    scan_start = range_start - timedelta(days=1)

    # Base grid strips (layer_rank = 0)
    d = scan_start
    while d < range_end:
        for strip in grid.strips:
            if _matches_pattern(strip.days, d):
                abs_start, abs_end = _interval_for(d, strip.start, strip.end)
                precedence = (0, _pattern_specificity(strip.days), strip.priority, order)
                candidates.append(
                    _Candidate(abs_start, abs_end, strip.content, precedence, strip.strip_id)
                )
                order += 1
        d += timedelta(days=1)

    # Overrides (layer_rank = 1)
    for ov in overrides:
        scope = ov.scope
        if scope.date is not None:
            target = date.fromisoformat(scope.date)
            if scan_start <= target < range_end:
                abs_start, abs_end = _interval_for(target, ov.start, ov.end)
                precedence = (1, 3, ov.priority, order)  # specific date = specificity 3
                candidates.append(
                    _Candidate(abs_start, abs_end, ov.content, precedence, ov.override_id)
                )
                order += 1
        else:
            eff_start = (
                date.fromisoformat(scope.effective_start)
                if scope.effective_start
                else scan_start
            )
            eff_end = (
                date.fromisoformat(scope.effective_end) if scope.effective_end else range_end
            )
            d = scan_start
            while d < range_end:
                if eff_start <= d <= eff_end and _matches_pattern(scope.days, d):
                    abs_start, abs_end = _interval_for(d, ov.start, ov.end)
                    precedence = (1, _pattern_specificity(scope.days), ov.priority, order)
                    candidates.append(
                        _Candidate(abs_start, abs_end, ov.content, precedence, ov.override_id)
                    )
                    order += 1
                d += timedelta(days=1)

    return candidates


def expand(
    grid: Grid,
    overrides: list[Override],
    range_start: date,
    range_end: date,
) -> list[DailySlot]:
    """Project the frozen grid + overrides onto [range_start, range_end).

    Args:
        grid: The frozen base grid for one channel.
        overrides: Sparse exceptions layered on top (may be empty).
        range_start: First date to materialize (inclusive).
        range_end: One past the last date (exclusive).

    Returns:
        Concrete ``DailySlot`` list, sorted by start time, with overlaps resolved
        by precedence and gaps filled by ``grid.default_content`` when present.
    """
    candidates = _materialize(grid, overrides, range_start, range_end)

    window_start = datetime.combine(range_start, time())
    window_end = datetime.combine(range_end, time())

    if not candidates and grid.default_content is None:
        return []

    # Collect boundary points within the window.
    points: set[datetime] = {window_start, window_end}
    for c in candidates:
        if window_start < c.start < window_end:
            points.add(c.start)
        if window_start < c.end < window_end:
            points.add(c.end)
    ordered_points = sorted(points)

    # Paint each elementary interval with its highest-precedence covering rule.
    painted: list[tuple[datetime, datetime, Content, str]] = []
    for left, right in zip(ordered_points, ordered_points[1:]):
        winner: _Candidate | None = None
        for c in candidates:
            if c.start <= left and c.end >= right:  # fully covers the elementary interval
                if winner is None or c.precedence > winner.precedence:
                    winner = c
        if winner is not None:
            painted.append((left, right, winner.content, winner.rule_id))
        elif grid.default_content is not None:
            painted.append((left, right, grid.default_content, "__default__"))
        # else: genuine gap, left unfilled

    # Merge adjacent elementary intervals won by the same rule, then build slots.
    merged: list[tuple[datetime, datetime, Content, str]] = []
    for left, right, content, rule_id in painted:
        if merged and merged[-1][1] == left and merged[-1][3] == rule_id:
            prev = merged[-1]
            merged[-1] = (prev[0], right, prev[2], rule_id)
        else:
            merged.append((left, right, content, rule_id))

    slots = [
        DailySlot(
            start_time=left,
            end_time=right,
            media_id=content.media_id,
            media_selection_strategy=content.strategy,
            category_filters=list(content.category_filters),
            notes=list(content.notes),
        )
        for left, right, content, _rule_id in merged
    ]
    slots.sort(key=lambda s: s.start_time)
    return slots
