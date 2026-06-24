"""Behavioural tests for the deterministic grid expander (Phase 0).

These pin the contract the Clojure port must also satisfy: determinism,
precedence cascade, partial overrides, cross-midnight strips, and default fill.
"""

from __future__ import annotations

from datetime import date

from tunabrain.scheduling.expander import expand
from tunabrain.scheduling.grid import (
    Content,
    Grid,
    GridStrip,
    Override,
    OverrideScope,
)


def _seinfeld_grid() -> Grid:
    """A small channel grid:
    - Seinfeld weekdays 17:00-18:00
    - Random sitcoms overnight 22:00-10:00 daily (wraps midnight)
    - default_content fills the rest of the day with random sitcoms
    """
    return Grid(
        channel="Classic Comedy",
        strips=[
            GridStrip(
                strip_id="seinfeld-prime",
                days="weekdays",
                start="17:00",
                end="18:00",
                content=Content(media_id="series:seinfeld", strategy="sequential"),
            ),
            GridStrip(
                strip_id="overnight-sitcoms",
                days="daily",
                start="22:00",
                end="10:00",
                content=Content(media_id="random:sitcom", strategy="random"),
            ),
        ],
        default_content=Content(media_id="random:sitcom", strategy="random"),
    )


def test_expansion_is_deterministic():
    grid = _seinfeld_grid()
    start, end = date(2026, 1, 5), date(2026, 1, 12)  # a full Mon-Sun week

    first = expand(grid, [], start, end)
    second = expand(grid, [], start, end)

    assert [s.model_dump() for s in first] == [s.model_dump() for s in second]


def test_weeks_are_identical_without_overrides():
    """The whole point: with no overrides, every week materializes the same."""
    grid = _seinfeld_grid()
    week1 = expand(grid, [], date(2026, 1, 5), date(2026, 1, 12))
    week2 = expand(grid, [], date(2026, 1, 12), date(2026, 1, 19))

    def shape(slots):
        # Compare structure (weekday + time-of-day + media), not absolute dates.
        return [
            (
                s.start_time.weekday(),
                s.start_time.strftime("%H:%M"),
                s.end_time.strftime("%H:%M"),
                s.media_id,
            )
            for s in slots
        ]

    assert shape(week1) == shape(week2)


def test_seinfeld_airs_on_weekdays_not_weekends():
    grid = _seinfeld_grid()
    slots = expand(grid, [], date(2026, 1, 5), date(2026, 1, 12))

    seinfeld = [s for s in slots if s.media_id == "series:seinfeld"]
    # Mon-Fri only => 5 airings
    assert len(seinfeld) == 5
    assert all(s.start_time.weekday() < 5 for s in seinfeld)
    assert all(s.start_time.strftime("%H:%M") == "17:00" for s in seinfeld)


def test_override_replaces_only_its_window():
    """A Saturday marathon carves out the daytime grid but leaves the overnight
    strip on either side intact (partial override)."""
    grid = _seinfeld_grid()
    override = Override(
        override_id="cheers-marathon",
        scope=OverrideScope(date="2026-01-10"),  # the Saturday
        start="10:00",
        end="22:00",
        content=Content(media_id="series:cheers", strategy="sequential", marathon=True),
    )

    slots = expand(grid, [override], date(2026, 1, 10), date(2026, 1, 11))

    cheers = [s for s in slots if s.media_id == "series:cheers"]
    assert len(cheers) == 1
    assert cheers[0].start_time.strftime("%H:%M") == "10:00"
    assert cheers[0].end_time.strftime("%H:%M") == "22:00"

    # The overnight sitcom strip (22:00 onward) survives after the marathon.
    after = [s for s in slots if s.start_time.strftime("%H:%M") == "22:00"]
    assert after and after[0].media_id == "random:sitcom"


def test_specific_date_override_outranks_recurring_override():
    """Specificity cascade: a dated override beats a same-window recurring one."""
    grid = _seinfeld_grid()
    recurring = Override(
        override_id="friday-movie",
        scope=OverrideScope(days=["fri"]),
        start="19:00",
        end="21:00",
        content=Content(media_id="movie:generic-comedy"),
    )
    dated = Override(
        override_id="special-premiere",
        scope=OverrideScope(date="2026-01-09"),  # a Friday
        start="19:00",
        end="21:00",
        content=Content(media_id="movie:special-premiere"),
    )

    slots = expand(grid, [recurring, dated], date(2026, 1, 9), date(2026, 1, 10))
    evening = [s for s in slots if s.start_time.strftime("%H:%M") == "19:00"]
    assert evening and evening[0].media_id == "movie:special-premiere"


def test_no_gaps_when_default_content_present():
    grid = _seinfeld_grid()
    slots = expand(grid, [], date(2026, 1, 6), date(2026, 1, 7))  # a Tuesday

    slots.sort(key=lambda s: s.start_time)
    # Wall-to-wall coverage: each slot's end meets the next slot's start.
    for a, b in zip(slots, slots[1:]):
        assert a.end_time == b.start_time


def test_empty_grid_without_default_yields_nothing():
    grid = Grid(channel="Empty", strips=[])
    assert expand(grid, [], date(2026, 1, 5), date(2026, 1, 12)) == []
