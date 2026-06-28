"""Unit tests for the quarterly grid proposal (Phase 4)."""

from __future__ import annotations

import json

import pytest

from tunabrain.api.models import (
    ChannelContext,
    QuarterlyGridRepairRequest,
    QuarterlyGridRequest,
)
from tunabrain.scheduling import quarterly_grid as qg
from tunabrain.scheduling.grid import (
    CatalogProfile,
    Content,
    FeasibilityReport,
    GenreProfile,
    Grid,
    GridStrip,
    RuntimeBucket,
    ShowProfile,
    StripFeasibility,
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
                tags=["classic"],
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
        runtime_histogram=[RuntimeBucket(label="20-30min", min_minutes=20, max_minutes=30, item_count=450)],
    )


def _request() -> QuarterlyGridRequest:
    return QuarterlyGridRequest(
        channel=ChannelContext(name="Classic Comedy", description="24/7 vintage sitcoms"),
        quarter="Q1",
        year=2026,
        catalog_profile=_profile(),
        quarterly_theme="New year, classic laughs",
        default_media_id="random:sitcom",
    )


# --- pure helpers -----------------------------------------------------------


def test_summarize_profile_includes_shows_and_genres():
    text = qg.summarize_catalog_profile(_profile())
    assert "series:seinfeld" in text
    assert "Seinfeld" in text
    assert "comedy" in text
    assert "22min" in text  # avg runtime rounded


def test_summarize_profile_omits_unschedulable_shows():
    profile = _profile()
    profile.shows.append(
        ShowProfile(
            media_id="series:noeps",
            title="Cancelled Pilot",
            genres=["drama"],
            episode_count=12,
            available_episode_count=0,  # nothing to air -> must be pruned
            avg_runtime_minutes=45,
        )
    )
    text = qg.summarize_catalog_profile(profile)
    assert "series:noeps" not in text  # dropped from the per-show list
    assert "1 further shows have no available episodes" in text  # but acknowledged
    assert "series:seinfeld" in text  # schedulable shows still listed


def test_summarize_profile_respects_max_shows_over_schedulable_only():
    text = qg.summarize_catalog_profile(_profile(), max_shows=1)
    # Cheers has more available eps than Seinfeld, so it leads; the tail count
    # reflects schedulable shows, not the raw catalog size.
    assert "series:cheers" in text
    assert "and 1 more schedulable shows" in text


def test_summarize_profile_rotates_tail_so_no_show_is_permanently_dead():
    import random

    # A catalog larger than the budget: shows s0..s9, all schedulable, descending
    # episode counts so s0/s1 are the deterministic anchors and s2..s9 the tail.
    shows = [
        ShowProfile(
            media_id=f"series:s{i}",
            title=f"Show {i}",
            episode_count=100,
            available_episode_count=100 - i,
            avg_runtime_minutes=20,
        )
        for i in range(10)
    ]
    profile = CatalogProfile(total_items=10, total_episodes=955, movie_count=0, shows=shows)

    seen: set[int] = set()
    for seed in range(30):
        text = qg.summarize_catalog_profile(profile, max_shows=4, rng=random.Random(seed))
        for i in range(10):
            if f"series:s{i}\n" in text or f"series:s{i})" in text:
                seen.add(i)

    # Every schedulable show surfaces across runs -> nothing is permanently dead.
    assert seen == set(range(10))
    # Anchors (highest-volume) are always present; a single run is still capped.
    one_run = qg.summarize_catalog_profile(profile, max_shows=4, rng=random.Random(0))
    assert "series:s0)" in one_run and "series:s1)" in one_run
    assert "and 6 more schedulable shows" in one_run


def test_skeleton_prompt_construction():
    messages = qg.build_daypart_skeleton_prompt(_request())
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    system, user = messages[0]["content"], messages[1]["content"]
    assert "JSON" in system
    assert "Classic Comedy" in user
    assert "06:00" in user  # broadcast_day_start
    assert "New year, classic laughs" in user  # theme carried for coherence


def test_strip_fill_prompt_includes_block_bounds_and_prior_strips():
    from tunabrain.scheduling.grid import DaypartBlock

    block = DaypartBlock(name="prime", start="17:00", end="22:00", role="marquee sitcoms")
    prior = [
        GridStrip(
            strip_id="classic-comedy-daytime-0",
            days="weekdays",
            start="10:00",
            end="12:00",
            content=Content(media_id="series:cheers"),
        )
    ]
    messages = qg.build_strip_fill_prompt(_request(), block, prior)
    user = messages[1]["content"]
    assert "prime" in user
    assert "17:00-22:00" in user
    assert "series:cheers" in user  # prior strip shown for consistency


def test_parse_strips_generates_stable_ids():
    payload = {
        "strips": [
            {"days": "weekdays", "start": "17:00", "end": "18:00", "media_id": "series:seinfeld"},
            {"days": ["sat", "sun"], "start": "18:00", "end": "19:00", "media_id": "series:cheers"},
        ]
    }
    strips = qg._parse_strips("Classic Comedy", "prime", payload, start_index=0)
    assert [s.strip_id for s in strips] == ["classic_comedy-prime-0", "classic_comedy-prime-1"]
    assert strips[0].content.media_id == "series:seinfeld"
    assert strips[1].days == ["sat", "sun"]


# --- orchestration with a mocked LLM ---------------------------------------


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Returns a daypart skeleton on the first call, then strips per daypart."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages, **kwargs):
        system = messages[0]["content"]
        if "DAYPARTING" in system:
            return _FakeResponse(
                json.dumps(
                    {
                        "blocks": [
                            {"name": "daytime", "start": "06:00", "end": "17:00", "role": "rerun sitcoms"},
                            {"name": "prime", "start": "17:00", "end": "22:00", "role": "marquee sitcoms"},
                            {"name": "overnight", "start": "22:00", "end": "06:00", "role": "filler"},
                        ]
                    }
                )
            )
        # strip fill - one strip keyed off the daypart in the user prompt
        user = messages[1]["content"]
        if "prime" in user:
            media = "series:seinfeld"
        elif "daytime" in user:
            media = "series:cheers"
        else:
            media = "random:sitcom"
        return _FakeResponse(
            json.dumps(
                {
                    "strips": [
                        {
                            "days": "daily",
                            "start": "00:00",
                            "end": "01:00",
                            "media_id": media,
                            "strategy": "sequential",
                        }
                    ]
                }
            )
        )


@pytest.fixture
def _mock_llm(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(qg, "get_chat_model", lambda *a, **k: fake)
    return fake


async def test_propose_quarterly_grid_runs_two_passes(_mock_llm):
    grid, skeleton, warnings, llm_calls = await qg.propose_quarterly_grid(_request())

    assert isinstance(grid, Grid)
    assert len(skeleton.blocks) == 3
    # 1 skeleton call + 1 per daypart
    assert llm_calls == 4
    assert len(grid.strips) == 3
    assert grid.default_content is not None and grid.default_content.media_id == "random:sitcom"
    assert not warnings
    # strips carry their daypart linkage
    assert {s.daypart for s in grid.strips} == {"daytime", "prime", "overnight"}


def test_invoke_json_truncation_raises_actionable_error(monkeypatch):
    """A length-limit truncation surfaces a clear ValueError, not a raw 500."""
    from openai import LengthFinishReasonError
    from openai.types.chat import ChatCompletion

    truncated = ChatCompletion(
        id="x",
        model="m",
        object="chat.completion",
        created=0,
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": '{"blo'},
            }
        ],
    )

    class _TruncatingLLM:
        def invoke(self, messages, **kwargs):
            raise LengthFinishReasonError(completion=truncated)

    monkeypatch.setattr(qg, "get_chat_model", lambda *a, **k: _TruncatingLLM())

    with pytest.raises(ValueError, match="completion budget"):
        qg._invoke_json([{"role": "user", "content": "x"}], max_tokens=4096, temperature=0.3)


async def test_repair_preserves_unflagged_strips(_mock_llm, monkeypatch):
    current = Grid(
        channel="Classic Comedy",
        strips=[
            GridStrip(
                strip_id="keep-1",
                days="weekdays",
                start="17:00",
                end="18:00",
                content=Content(media_id="series:cheers"),
            )
        ],
    )
    report = FeasibilityReport(
        horizon_start="2026-01-01",
        horizon_end="2026-04-01",
        overall_status="blocked",
        strip_findings=[
            StripFeasibility(
                rule_id="keep-1",
                media_id="series:cheers",
                slots_required=65,
                episodes_available=270,
                status="ok",
            )
        ],
    )

    # Repair LLM returns a corrected list keeping the id.
    def fake_invoke(messages, **kwargs):
        return _FakeResponse(
            json.dumps(
                {
                    "strips": [
                        {
                            "strip_id": "keep-1",
                            "days": "weekdays",
                            "start": "17:00",
                            "end": "18:00",
                            "media_id": "series:cheers",
                            "strategy": "sequential",
                        }
                    ],
                    "changes": ["No change needed"],
                }
            )
        )

    monkeypatch.setattr(_mock_llm, "invoke", fake_invoke)

    req = QuarterlyGridRepairRequest(
        channel=ChannelContext(name="Classic Comedy", description="x"),
        catalog_profile=_profile(),
        current_grid=current,
        feasibility_report=report,
    )
    revised, changes, llm_calls = await qg.repair_quarterly_grid(req)

    assert llm_calls == 1
    assert revised.strips[0].strip_id == "keep-1"
    assert changes == ["No change needed"]
