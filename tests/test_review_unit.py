"""Unit tests for the schedule review / critique loop (Phase 7)."""

from __future__ import annotations

import json

import pytest

from tunabrain.api.models import (
    ChannelContext,
    ReviewReviseRequest,
    ScheduleReview,
    ScheduleReviewRequest,
    ReviewFinding,
    ReviewSlot,
)
from tunabrain.scheduling import quarterly_grid as qg
from tunabrain.scheduling import review as rv
from tunabrain.scheduling.grid import (
    CatalogProfile,
    Content,
    DaypartBlock,
    DaypartSkeleton,
    GenreProfile,
    Grid,
    GridStrip,
    ShowProfile,
)


def _profile() -> CatalogProfile:
    return CatalogProfile(
        total_items=900,
        total_episodes=880,
        movie_count=20,
        shows=[
            ShowProfile(
                media_id="series:seinfeld",
                title="Seinfeld",
                genres=["comedy", "sitcom"],
                episode_count=180,
                available_episode_count=180,
                avg_runtime_minutes=22,
            ),
            ShowProfile(
                media_id="series:cheers",
                title="Cheers",
                genres=["comedy", "sitcom"],
                episode_count=270,
                available_episode_count=270,
                avg_runtime_minutes=24,
            ),
        ],
        genres=[GenreProfile(genre="comedy", show_count=2, episode_count=450)],
    )


def _skeleton() -> DaypartSkeleton:
    return DaypartSkeleton(
        channel="Classic Comedy",
        blocks=[
            DaypartBlock(name="daytime", start="06:00", end="17:00", role="rerun sitcoms"),
            DaypartBlock(
                name="prime", start="17:00", end="22:00", role="marquee sitcoms",
                genre_focus=["sitcom"],
            ),
        ],
    )


def _grid() -> Grid:
    return Grid(
        channel="Classic Comedy",
        strips=[
            GridStrip(
                strip_id="prime-0", days="weekdays", start="17:00", end="18:00",
                content=Content(media_id="random:comedy", strategy="random"), daypart="prime",
            ),
        ],
        default_content=Content(media_id="random:sitcom", strategy="random"),
    )


def _sample_week() -> list[ReviewSlot]:
    return [
        ReviewSlot(day="mon", start="17:00", end="18:00", label="random: comedy pool",
                   media_id="random:comedy", strategy="random", daypart="prime"),
        ReviewSlot(day="tue", start="17:00", end="18:00", label="random: comedy pool",
                   media_id="random:comedy", strategy="random", daypart="prime"),
    ]


def _review_request() -> ScheduleReviewRequest:
    return ScheduleReviewRequest(
        channel=ChannelContext(name="Classic Comedy", description="24/7 vintage sitcoms"),
        skeleton=_skeleton(),
        grid=_grid(),
        sample_week=_sample_week(),
        catalog_profile=_profile(),
    )


# --- pure prompt rendering --------------------------------------------------


def test_render_daypart_plan_lists_roles_and_focus():
    text = rv.render_daypart_plan(_review_request())
    assert "prime 17:00-22:00: marquee sitcoms" in text
    assert "focus: sitcom" in text


def test_render_daypart_plan_tolerates_missing_skeleton():
    req = _review_request()
    req.skeleton = None
    assert "no explicit daypart plan" in rv.render_daypart_plan(req)


def test_render_sample_week_groups_by_day_in_time_order():
    slots = [
        ReviewSlot(day="mon", start="20:00", end="22:00", label="Movie Night", media_id="movie:1"),
        ReviewSlot(day="mon", start="17:00", end="17:30", label="Seinfeld", media_id="series:1"),
    ]
    req = _review_request()
    req.sample_week = slots
    text = rv.render_sample_week(req)
    # Monday header present, and the 17:00 slot renders before the 20:00 slot.
    assert "MON:" in text
    assert text.index("Seinfeld") < text.index("Movie Night")


def test_build_review_prompt_includes_plan_week_and_profile():
    messages = rv.build_review_prompt(_review_request())
    user = messages[1]["content"]
    assert "marquee sitcoms" in user       # plan
    assert "comedy pool" in user           # sample week
    assert "Seinfeld" in user              # catalog profile (under-used series)


# --- verdict derivation -----------------------------------------------------


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


def _fake_llm_returning(payload: dict):
    class _LLM:
        def invoke(self, messages, **kwargs):
            return _FakeResponse(json.dumps(payload))
    return _LLM()


@pytest.fixture
def _mock_review_llm(monkeypatch):
    """review_grid calls quarterly_grid._invoke_json, which resolves
    get_chat_model in the quarterly_grid namespace — so that's the patch target."""
    def _install(payload):
        monkeypatch.setattr(qg, "get_chat_model", lambda *a, **k: _fake_llm_returning(payload))
    return _install


async def test_review_fails_when_a_major_finding_exists(_mock_review_llm):
    _mock_review_llm({
        "verdict": "pass",  # model is wrong; a major finding is present
        "score": 0.4,
        "summary": "Prime is all generic pools.",
        "findings": [
            {"aspect": "series-usage", "severity": "major",
             "message": "prime is random:comedy though Seinfeld/Cheers exist", "target": "prime"},
        ],
    })
    review, calls = await rv.review_grid(_review_request())
    assert calls == 1
    assert review.verdict == "fail", "a major finding must force a fail regardless of model verdict"
    assert review.findings[0].aspect == "series-usage"


async def test_review_passes_on_minor_only(_mock_review_llm):
    _mock_review_llm({
        "verdict": "fail",  # model is wrong; only a minor finding is present
        "score": 0.85,
        "summary": "Strong week, small nit.",
        "findings": [
            {"aspect": "pacing", "severity": "minor", "message": "late block a touch long"},
        ],
    })
    review, _ = await rv.review_grid(_review_request())
    assert review.verdict == "pass"


async def test_review_passes_with_no_findings(_mock_review_llm):
    _mock_review_llm({"verdict": "pass", "score": 0.95, "summary": "Great.", "findings": []})
    review, _ = await rv.review_grid(_review_request())
    assert review.verdict == "pass"
    assert review.findings == []


# --- revise -----------------------------------------------------------------


def _revise_request() -> ReviewReviseRequest:
    review = ScheduleReview(
        verdict="fail", score=0.4, summary="Prime too generic.",
        findings=[ReviewFinding(aspect="series-usage", severity="major",
                                message="use a named series in prime", target="prime-0")],
    )
    return ReviewReviseRequest(
        channel=ChannelContext(name="Classic Comedy", description="24/7 vintage sitcoms"),
        catalog_profile=_profile(),
        current_grid=_grid(),
        review=review,
    )


def test_build_revise_prompt_carries_findings_and_current_strips():
    messages = rv.build_revise_prompt(_revise_request())
    user = messages[1]["content"]
    assert "use a named series in prime" in user   # finding
    assert "prime-0" in user                         # current strip id
    assert "Seinfeld" in user                        # catalog for the swap


async def test_revise_grid_from_review_returns_revised_grid(_mock_review_llm):
    _mock_review_llm({
        "strips": [
            {"strip_id": "prime-0", "days": "weekdays", "start": "17:00", "end": "18:00",
             "media_id": "series:seinfeld", "strategy": "sequential"},
        ],
        "changes": ["prime-0: random:comedy -> series:seinfeld (addresses series-usage)"],
    })
    grid, changes, calls = await rv.revise_grid_from_review(_revise_request())
    assert calls == 1
    assert isinstance(grid, Grid)
    assert grid.strips[0].content.media_id == "series:seinfeld"
    assert grid.strips[0].strip_id == "prime-0", "existing strip id preserved"
    assert len(changes) == 1
