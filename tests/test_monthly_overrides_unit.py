"""Unit tests for monthly override proposal (Phase 6)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from tunabrain.api.models import ChannelContext, MonthlyOverridesRequest
from tunabrain.scheduling import monthly_overrides as mo
from tunabrain.scheduling.expander import expand
from tunabrain.scheduling.grid import (
    CatalogProfile,
    Content,
    Grid,
    GridStrip,
    OverrideScope,
    ShowProfile,
)


def _profile() -> CatalogProfile:
    return CatalogProfile(
        total_items=300,
        total_episodes=300,
        shows=[
            ShowProfile(
                media_id="series:cheers",
                title="Cheers",
                genres=["comedy"],
                episode_count=270,
                available_episode_count=270,
                avg_runtime_minutes=24,
            )
        ],
    )


def _grid() -> Grid:
    return Grid(
        channel="Classic Comedy",
        strips=[
            GridStrip(
                strip_id="overnight",
                days="daily",
                start="22:00",
                end="10:00",
                content=Content(media_id="random:sitcom", strategy="random"),
            ),
            GridStrip(
                strip_id="daytime",
                days="daily",
                start="10:00",
                end="22:00",
                content=Content(media_id="series:seinfeld", strategy="sequential"),
            ),
        ],
        default_content=Content(media_id="random:sitcom"),
    )


def _request() -> MonthlyOverridesRequest:
    return MonthlyOverridesRequest(
        channel=ChannelContext(name="Classic Comedy", description="vintage sitcoms"),
        month="2026-01",
        grid=_grid(),
        catalog_profile=_profile(),
        planned_events=["Cheers marathon Saturday the 10th"],
    )


# --- pure helpers -----------------------------------------------------------


def test_month_bounds():
    first, last = mo.month_bounds("2026-02")
    assert first == date(2026, 2, 1)
    assert last == date(2026, 2, 28)  # 2026 is not a leap year


def test_grid_summary_lists_strips_and_default():
    text = mo.summarize_grid_for_prompt(_grid())
    assert "series:seinfeld" in text
    assert "random:sitcom" in text
    assert "default fill" in text


def test_prompt_includes_grid_and_events():
    messages = mo.build_monthly_overrides_prompt(_request())
    system, user = messages[0]["content"], messages[1]["content"]
    assert "OVERRIDES" in system
    assert "empty list" in system.lower()  # sparsity is encouraged
    assert "Cheers marathon" in user
    assert "series:seinfeld" in user  # frozen grid shown for delta-only proposals


def test_parse_dated_override():
    payload = {
        "overrides": [
            {
                "scope": {"date": "2026-01-10"},
                "start": "10:00",
                "end": "22:00",
                "media_id": "series:cheers",
                "strategy": "sequential",
                "marathon": True,
                "label": "Cheers Marathon",
            }
        ]
    }
    overrides, warnings = mo._parse_overrides("Classic Comedy", "2026-01", payload)
    assert not warnings
    assert len(overrides) == 1
    ov = overrides[0]
    assert ov.scope.date == "2026-01-10"
    assert ov.content.marathon is True
    assert ov.override_id == "classic_comedy-2026-01-ovr-0"


def test_recurring_override_bounded_to_month():
    payload = {
        "overrides": [
            {
                "scope": {"days": ["fri"]},
                "start": "19:00",
                "end": "21:00",
                "media_id": "movie:comedy-night",
            }
        ]
    }
    overrides, _ = mo._parse_overrides("Classic Comedy", "2026-01", payload)
    scope = overrides[0].scope
    assert scope.days == ["fri"]
    # Recurrence is clamped to the month deterministically.
    assert scope.effective_start == "2026-01-01"
    assert scope.effective_end == "2026-01-31"


def test_out_of_month_date_warns_but_keeps():
    payload = {
        "overrides": [
            {"scope": {"date": "2026-03-05"}, "start": "10:00", "end": "12:00", "media_id": "x"}
        ]
    }
    overrides, warnings = mo._parse_overrides("C", "2026-01", payload)
    assert len(overrides) == 1
    assert any("outside 2026-01" in w for w in warnings)


def test_invalid_scope_skipped():
    payload = {"overrides": [{"scope": {}, "start": "10:00", "end": "12:00", "media_id": "x"}]}
    overrides, warnings = mo._parse_overrides("C", "2026-01", payload)
    assert overrides == []
    assert warnings


# --- wire serialization: closed-union scope must carry no null siblings ------


def test_dated_scope_serializes_without_null_siblings():
    """A single-date scope must wire as exactly {"date": ...} — the inapplicable
    days/effective_* keys must not appear at all (the scheduler's OverrideScope
    is a closed union and rejects stray nulls)."""
    scope = OverrideScope(date="2026-06-05")
    assert scope.model_dump() == {"date": "2026-06-05"}
    assert json.loads(scope.model_dump_json()) == {"date": "2026-06-05"}


def test_recurring_scope_serializes_only_set_effective_bounds():
    """A days-scope keeps its discriminator plus only whichever effective_*
    bounds are actually set; an unset effective_end must drop out, not null."""
    scope = OverrideScope(days=["fri"], effective_start="2026-06-01")
    assert scope.model_dump() == {"days": ["fri"], "effective_start": "2026-06-01"}
    both = OverrideScope(
        days="weekends", effective_start="2026-06-01", effective_end="2026-06-30"
    )
    assert both.model_dump() == {
        "days": "weekends",
        "effective_start": "2026-06-01",
        "effective_end": "2026-06-30",
    }


def test_scope_with_both_date_and_days_is_invalid():
    """Genuine ambiguity stays a hard error — dropping nulls must not paper it over."""
    with pytest.raises(ValueError):
        OverrideScope(date="2026-06-05", days=["fri"])


def test_proposed_overrides_carry_no_null_scope_keys():
    """End-to-end: every scope object in proposed overrides has no null values."""
    first_day, last_day = mo.month_bounds("2026-01")
    payload = {
        "overrides": [
            {"scope": {"date": "2026-01-10"}, "start": "10:00", "end": "22:00", "media_id": "x"},
            {"scope": {"days": ["fri"]}, "start": "19:00", "end": "21:00", "media_id": "y"},
        ]
    }
    overrides, _ = mo._parse_overrides("Classic Comedy", "2026-01", payload)
    dumped_scopes = [ov.model_dump()["scope"] for ov in overrides]
    assert dumped_scopes[0] == {"date": "2026-01-10"}
    assert dumped_scopes[1] == {
        "days": ["fri"],
        "effective_start": first_day.isoformat(),
        "effective_end": last_day.isoformat(),
    }
    for scope in dumped_scopes:
        assert None not in scope.values()


# --- orchestration + end-to-end through the expander ------------------------


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


async def test_propose_and_apply_through_expander(monkeypatch):
    """A proposed dated marathon should actually carve out the daytime grid when
    run through the deterministic expander (Phase 0 + Phase 6 together)."""
    payload = {
        "overrides": [
            {
                "scope": {"date": "2026-01-10"},
                "start": "10:00",
                "end": "22:00",
                "media_id": "series:cheers",
                "strategy": "sequential",
                "marathon": True,
                "label": "Cheers Marathon",
                "note": "Operator request",
            }
        ]
    }
    monkeypatch.setattr(
        mo, "get_chat_model", lambda *a, **k: type(
            "L", (), {"invoke": lambda self, m, **k: _FakeResponse(json.dumps(payload))}
        )()
    )

    overrides, warnings, llm_calls = await mo.propose_monthly_overrides(_request())
    assert llm_calls == 1 and not warnings and len(overrides) == 1

    # Expand the Saturday with the proposed override.
    slots = expand(_grid(), overrides, date(2026, 1, 10), date(2026, 1, 11))
    cheers = [s for s in slots if s.media_id == "series:cheers"]
    assert len(cheers) == 1
    assert cheers[0].start_time.strftime("%H:%M") == "10:00"
    assert cheers[0].end_time.strftime("%H:%M") == "22:00"
    # Overnight grid survives on either side of the marathon window.
    assert any(s.media_id == "random:sitcom" for s in slots)
    # The normal daytime Seinfeld strip is fully displaced that day.
    assert not any(s.media_id == "series:seinfeld" for s in slots)


@pytest.mark.parametrize("payload", [{"overrides": []}, {}])
async def test_empty_overrides_is_valid(monkeypatch, payload):
    monkeypatch.setattr(
        mo, "get_chat_model", lambda *a, **k: type(
            "L", (), {"invoke": lambda self, m, **k: _FakeResponse(json.dumps(payload))}
        )()
    )
    overrides, warnings, _ = await mo.propose_monthly_overrides(_request())
    assert overrides == [] and not warnings
