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


def test_strip_fill_prompt_instructs_series_first_authoring():
    """The system prompt must steer the model toward naming specific series
    for anchor/marquee dayparts, not defaulting to random:<genre> pools —
    the fix for schedules that were all generic genre rotation and almost
    no recurring named shows."""
    from tunabrain.scheduling.grid import DaypartBlock

    block = DaypartBlock(name="prime", start="17:00", end="22:00", role="marquee sitcoms")
    messages = qg.build_strip_fill_prompt(_request(), block, [])
    system = messages[0]["content"]
    assert "SERIES-FIRST" in system
    assert "series:<media_id>" in system
    assert "random:<genre>" in system


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


# --- split round-trip (DURATION_AWARE_SCHEDULING.md §4.3, Option A) --------


def test_render_candidate_menu_empty_is_blank():
    assert qg.render_candidate_menu([]) == ""
    assert qg.render_candidate_menu(None) == ""  # type: ignore[arg-type]


def test_render_candidate_menu_renders_layouts_and_inventory():
    from tunabrain.scheduling.grid import CandidateSlot, DaypartCandidate

    candidates = [
        DaypartCandidate(
            layout_id="genre-movie-90-105min",
            weight=12.0,
            slots=[
                CandidateSlot(duration_minutes=90, category="movie", available_count=12),
            ],
        ),
        DaypartCandidate(
            layout_id="genre-sitcom-15-30min",
            weight=200.0,
            slots=[
                CandidateSlot(duration_minutes=30, category="sitcom", available_count=200),
                CandidateSlot(duration_minutes=30, category="sitcom", available_count=200),
            ],
        ),
    ]
    menu = qg.render_candidate_menu(candidates)
    assert "genre-movie-90-105min" in menu
    assert "90min movie" in menu
    assert "x12 available" in menu
    assert "genre-sitcom-15-30min" in menu
    assert "30min sitcom" in menu


def test_strip_fill_prompt_includes_candidate_menu_when_supplied():
    from tunabrain.scheduling.grid import CandidateSlot, DaypartBlock, DaypartCandidate

    block = DaypartBlock(name="prime", start="20:00", end="22:00", role="movie night")
    candidates = [
        DaypartCandidate(
            layout_id="movie-90min",
            weight=12.0,
            slots=[CandidateSlot(duration_minutes=90, category="movie", available_count=12)],
        )
    ]
    with_menu = qg.build_strip_fill_prompt(_request(), block, [], candidates=candidates)
    without_menu = qg.build_strip_fill_prompt(_request(), block, [], candidates=None)

    assert "DURATION-FEASIBLE SLOT MENU" in with_menu[1]["content"]
    assert "movie-90min" in with_menu[1]["content"]
    assert "prefer strip lengths" in with_menu[0]["content"]  # system-prompt rule added

    assert "DURATION-FEASIBLE SLOT MENU" not in without_menu[1]["content"]
    assert "prefer strip lengths" not in without_menu[0]["content"]


async def test_propose_daypart_skeleton_and_strip_fill_compose_like_propose_quarterly_grid(
    _mock_llm,
):
    """The split functions, called directly across "two round trips", should
    produce the identical result propose_quarterly_grid gets from calling them
    internally in one call — the whole point of the refactor being additive."""
    skeleton, skeleton_calls = await qg.propose_daypart_skeleton(_request())
    assert len(skeleton.blocks) == 3
    assert skeleton_calls == 1

    all_strips: list[GridStrip] = []
    total_calls = skeleton_calls
    for block in skeleton.blocks:
        strips, calls = await qg.propose_strip_fill(_request(), block, all_strips)
        total_calls += calls
        all_strips.extend(strips)

    assert total_calls == 4
    assert len(all_strips) == 3
    assert {s.daypart for s in all_strips} == {"daytime", "prime", "overnight"}


async def test_propose_strip_fill_accepts_narrower_strip_fill_request(_mock_llm):
    """The real split-round-trip caller (tunarr-scheduler) sends a
    StripFillRequest, not a full QuarterlyGridRequest — build_strip_fill_prompt
    only needs the fields the two share, so this must work identically."""
    from tunabrain.api.models import StripFillRequest
    from tunabrain.scheduling.grid import DaypartBlock

    block = DaypartBlock(name="prime", start="17:00", end="22:00", role="marquee sitcoms")
    request = StripFillRequest(
        channel=ChannelContext(name="Classic Comedy", description="24/7 vintage sitcoms"),
        catalog_profile=_profile(),
        block=block,
        candidates=[],
        prior_strips=[],
    )
    strips, calls = await qg.propose_strip_fill(request, block, [])
    assert calls == 1
    assert len(strips) == 1
    assert strips[0].content.media_id == "series:seinfeld"


async def test_propose_daypart_skeleton_accepts_narrower_skeleton_request(_mock_llm):
    from tunabrain.api.models import DaypartSkeletonRequest

    request = DaypartSkeletonRequest(
        channel=ChannelContext(name="Classic Comedy", description="24/7 vintage sitcoms"),
        catalog_profile=_profile(),
    )
    skeleton, calls = await qg.propose_daypart_skeleton(request)
    assert calls == 1
    assert len(skeleton.blocks) == 3


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


class _Reply:
    def __init__(self, content: str):
        self.content = content


class _ScriptedLLM:
    """Returns a queued reply per ``invoke`` call, tracking how many were made."""

    def __init__(self, *contents: str):
        self._replies = list(contents)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return _Reply(self._replies.pop(0))


def test_strip_code_fences_unwraps_json_block():
    fenced = '```json\n{"strips": []}\n```'
    assert json.loads(qg._strip_code_fences(fenced)) == {"strips": []}
    # Bare JSON passes through untouched.
    assert qg._strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_invoke_json_strips_markdown_fence(monkeypatch):
    """A model that ignores json_object and fences its output still parses."""
    llm = _ScriptedLLM('```json\n{"blocks": []}\n```')
    monkeypatch.setattr(qg, "get_chat_model", lambda *a, **k: llm)

    assert qg._invoke_json([], max_tokens=4096, temperature=0.3) == {"blocks": []}
    assert llm.calls == 1


def test_invoke_json_retries_then_succeeds(monkeypatch):
    """A transient bad-JSON response is re-rolled rather than failing the request."""
    llm = _ScriptedLLM("not json at all", '{"blocks": [1]}')
    monkeypatch.setattr(qg, "get_chat_model", lambda *a, **k: llm)

    assert qg._invoke_json([], max_tokens=4096, temperature=0.3) == {"blocks": [1]}
    assert llm.calls == 2


def test_invoke_json_raises_after_exhausting_retries(monkeypatch):
    """Persistent bad JSON surfaces a clear error after the attempt cap."""
    llm = _ScriptedLLM(*(["garbage"] * qg._MAX_JSON_ATTEMPTS))
    monkeypatch.setattr(qg, "get_chat_model", lambda *a, **k: llm)

    with pytest.raises(ValueError, match="invalid JSON after"):
        qg._invoke_json([], max_tokens=4096, temperature=0.3)
    assert llm.calls == qg._MAX_JSON_ATTEMPTS


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


def test_repair_prompt_makes_nonexistent_and_repetition_findings_actionable():
    """The deterministic checker now emits two shortfall kinds the old repair
    guidance ('reduce frequency / swap to a show with more episodes / pool into
    random') couldn't resolve: a `random:<genre>` naming a category that isn't in
    the catalog (hallucinated — reducing its frequency fixes nothing), and a
    short slot whose right-length pool is too thin for how often it airs. The
    repair prompt must tell the model the actual fix for each, and must still
    render the checker's verbatim finding messages so it knows which strip."""
    current = Grid(
        channel="Classic Comedy",
        strips=[
            GridStrip(
                strip_id="ghost",
                days="weekdays",
                start="20:00",
                end="21:00",
                content=Content(media_id="random:sci-fi-and-fantasy", strategy="random"),
            ),
            GridStrip(
                strip_id="shorts",
                days="weekdays",
                start="12:00",
                end="12:30",
                content=Content(media_id="random:comedy", strategy="random"),
            ),
        ],
    )
    report = FeasibilityReport(
        horizon_start="2026-01-01",
        horizon_end="2026-04-01",
        overall_status="blocked",
        strip_findings=[
            StripFeasibility(
                rule_id="ghost",
                media_id="random:sci-fi-and-fantasy",
                slots_required=65,
                episodes_available=0,
                status="shortfall",
                message="category 'sci-fi-and-fantasy' does not exist in the catalog profile",
            ),
            StripFeasibility(
                rule_id="shorts",
                media_id="random:comedy",
                slots_required=65,
                episodes_available=450,
                status="shortfall",
                message="only 5 'comedy' item(s) within 15min of the strip's 30min length "
                "for 15 airing(s)/week — each would repeat ~3.0×/week",
            ),
        ],
    )
    req = QuarterlyGridRepairRequest(
        channel=ChannelContext(name="Classic Comedy", description="x"),
        catalog_profile=_profile(),
        current_grid=current,
        feasibility_report=report,
    )
    messages = qg.build_grid_repair_prompt(req)
    system = messages[0]["content"]
    user = messages[1]["content"]

    # Hallucinated-category guidance: swap to a real genre / series, don't just
    # cut frequency.
    assert "does not exist" in system
    assert "AVAILABLE MEDIA" in system
    # Length/repetition guidance: reduce frequency / lengthen / drop duplicates,
    # not swap to another thin short category.
    assert "repeats N" in system or "repetition" in system.lower()

    # The checker's verbatim messages reach the model so it knows which strip
    # and why (the existing plumbing this guidance relies on).
    assert "does not exist in the catalog profile" in user
    assert "repeat ~3.0" in user
