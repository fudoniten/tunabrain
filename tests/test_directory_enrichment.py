"""Unit tests for the /enrich/profile chain (directory / tag-group profiling).

These cover the single-call group-profiling building block: request validation
(empty filename sample -> 422), the happy path (dimensions + tags), tag/dimension
sanitisation, and graceful degradation to an empty profile when the LLM call
fails. The LLM is stubbed so the tests never make a network call, mirroring the
pattern in test_describe.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tunabrain.api.models import (
    CategoryDefinition,
    CategoryValue,
    EnrichProfileRequest,
    GroupContext,
)
from tunabrain.chains import directory_enrichment as prof_mod
from tunabrain.chains.directory_enrichment import ProfileResult, enrich_profile


def _request(**overrides) -> EnrichProfileRequest:
    base = {
        "concept_name": "Adam Neely Music",
        "sample_filenames": [
            "2021-08-03 Adam Neely - Mechanation (2021).mp4",
            "2022-01-11 The most UNHINGED chord progression.mp4",
            "2023-05-02 Why does this song sound SO GOOD.mp4",
        ],
    }
    base.update(overrides)
    return EnrichProfileRequest(**base)


@pytest.fixture
def stub_profile(monkeypatch):
    """Stub the LLM + parser so the chain runs offline.

    The returned dict records what the chain saw and lets each test set the
    structured ``ProfileResult`` the fake LLM should produce.
    """
    calls: dict[str, object] = {}
    calls["result"] = ProfileResult(
        dimensions={"channel": ["muse"], "audience": ["adult"]},
        tags=["music", "music-theory", "educational", "jazz"],
    )

    class FakeParser:
        def __init__(self, *args, **kwargs):
            pass

        def get_format_instructions(self) -> str:
            return "FORMAT"

        async def ainvoke(self, _response):
            return calls["result"]

    class FakeLLM:
        async def ainvoke(self, messages):
            calls["invoked"] = True
            calls["messages"] = messages
            return "raw-llm-response"

    monkeypatch.setattr(prof_mod, "PydanticOutputParser", FakeParser)
    monkeypatch.setattr(prof_mod, "get_chat_model", lambda *a, **k: FakeLLM())
    return calls


def test_empty_sample_filenames_is_rejected():
    """An empty filename sample fails validation (surfaces as 422 at the route)."""
    with pytest.raises(ValidationError):
        EnrichProfileRequest(concept_name="Adam Neely Music", sample_filenames=[])


@pytest.mark.anyio
async def test_returns_dimensions_and_tags(stub_profile):
    resp = await enrich_profile(_request())

    assert resp.concept_name == "Adam Neely Music"
    assert resp.dimensions["channel"] == ["muse"]
    assert 3 <= len(resp.tags) <= 7
    assert resp.grounding_source == "filename-pattern"
    assert resp.cost_estimate.llm_calls_used == 1
    assert resp.warnings == []
    assert stub_profile["invoked"] is True


@pytest.mark.anyio
async def test_tags_are_sanitized(stub_profile):
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["muse"]},
        tags=["Music Theory", "JAZZ!!", "  ", "music-theory"],
    )
    resp = await enrich_profile(_request())

    # Lowercased, hyphenated, punctuation stripped, blanks dropped, deduped.
    assert resp.tags == ["music-theory", "jazz"]
    assert all(t == t.lower() for t in resp.tags)
    assert all(" " not in t for t in resp.tags)


@pytest.mark.anyio
async def test_unknown_dimension_keys_are_dropped(stub_profile):
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["muse"], "made-up": ["nonsense"], "audience": []},
        tags=["music"],
    )
    resp = await enrich_profile(_request())

    # Only known, non-empty dimensions survive.
    assert set(resp.dimensions.keys()) == {"channel"}


@pytest.mark.anyio
async def test_degrades_to_empty_profile_on_llm_failure(monkeypatch):
    class BoomLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("llm exploded")

    monkeypatch.setattr(prof_mod, "get_chat_model", lambda *a, **k: BoomLLM())

    resp = await enrich_profile(_request())

    # A failed LLM call is a warning, not a 500: empty profile is returned.
    assert resp.dimensions == {}
    assert resp.tags == []
    assert any("profile failed" in w for w in resp.warnings)
    assert resp.cost_estimate.llm_calls_used == 1


# --- request.categories: controlled vocabulary ------------------------------
#
# When Grout supplies `categories` (fetched from Tunarr Scheduler's
# /api/dimensions/descriptions), the model is constrained to the given
# dimensions and their candidate values: hallucinated values are dropped and
# every listed dimension is guaranteed at least one value.

_CATEGORIES = {
    "channel": CategoryDefinition(
        description="Which channel this content airs on",
        values=[
            CategoryValue(value="toontown", description="Animated content"),
            CategoryValue(value="infobytes", description="Science & technology"),
        ],
    ),
    "audience": CategoryDefinition(
        description="Who this is appropriate for",
        values=["kids", "teen", "adult"],
    ),
}


@pytest.mark.anyio
async def test_categories_prompt_includes_candidate_values(stub_profile):
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["toontown"], "audience": ["kids"]},
        tags=["retro", "commercials", "gaming"],
    )
    await enrich_profile(_request(categories=_CATEGORIES))

    messages = stub_profile["messages"]
    rendered = "\n".join(str(m) for m in messages)
    assert "toontown: Animated content" in rendered
    assert "infobytes: Science & technology" in rendered
    assert "kids" in rendered and "teen" in rendered and "adult" in rendered


@pytest.mark.anyio
async def test_categories_hallucinated_value_is_dropped_and_replaced(stub_profile):
    # The model invents "educational" for channel, which is not a candidate.
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["educational"], "audience": ["kids"]},
        tags=["retro"],
    )
    resp = await enrich_profile(_request(categories=_CATEGORIES))

    # Hallucinated value dropped; dimension still gets a fallback value
    # rather than coming back empty/nil.
    assert resp.dimensions["channel"] == ["toontown"]
    assert resp.dimensions["audience"] == ["kids"]


@pytest.mark.anyio
async def test_categories_omitted_dimension_is_filled_with_fallback(stub_profile):
    # The model omits "audience" entirely.
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["infobytes"]},
        tags=["science"],
    )
    resp = await enrich_profile(_request(categories=_CATEGORIES))

    assert resp.dimensions["channel"] == ["infobytes"]
    # Every requested dimension must have at least one value.
    assert resp.dimensions["audience"] == ["kids"]


@pytest.mark.anyio
async def test_categories_unrequested_dimension_is_dropped(stub_profile):
    # The model returns a dimension outside the requested categories.
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["toontown"], "made-up": ["nonsense"]},
        tags=["retro"],
    )
    resp = await enrich_profile(_request(categories=_CATEGORIES))

    assert set(resp.dimensions.keys()) == {"channel", "audience"}


@pytest.mark.anyio
async def test_no_categories_preserves_freeform_behavior(stub_profile):
    # Without categories, the model proposes freely and omitted dimensions
    # stay omitted (no forced fallback) — pre-v1.1 behavior.
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["muse"]},
        tags=["music"],
    )
    resp = await enrich_profile(_request())

    assert resp.dimensions == {"channel": ["muse"]}
    assert "audience" not in resp.dimensions


# --- request.context: operator-supplied grounding notes ---------------------
#
# Grout threads a directory profile's manually-set `context` (text/links) into
# this field so an operator can correct a misclassification, e.g. "these are
# retro VIDEO GAME ads, not vintage film content." Unlike per-item
# MediaContext, links here are never fetched/summarized -- they're rendered
# into the prompt as plain text alongside the free-form notes.


@pytest.mark.anyio
async def test_context_text_and_links_appear_in_prompt(stub_profile):
    await enrich_profile(
        _request(
            context=GroupContext(
                text="these are retro VIDEO GAME ads, not vintage film content",
                links=["https://example.com/about-these-ads"],
            )
        )
    )

    rendered = "\n".join(str(m) for m in stub_profile["messages"])
    assert "these are retro VIDEO GAME ads, not vintage film content" in rendered
    assert "https://example.com/about-these-ads" in rendered


@pytest.mark.anyio
async def test_context_links_are_not_fetched(stub_profile, monkeypatch):
    # Guard against ever silently adding a fetch step to this chain: no
    # httpx/requests-style client should be touched just because a link was
    # supplied. (There's no fetch machinery imported into this module at all
    # today; this test documents the intent so a future change doesn't
    # reintroduce it without a conscious decision.)
    import tunabrain.chains.directory_enrichment as mod

    assert not hasattr(mod, "httpx")
    assert not hasattr(mod, "requests")

    await enrich_profile(_request(context=GroupContext(links=["https://example.com/x"])))
    # No exception, no network call attempted -- the stubbed LLM is the only
    # thing invoked (see the `stub_profile` fixture).


@pytest.mark.anyio
async def test_no_context_omits_operator_context_section(stub_profile):
    await enrich_profile(_request())

    rendered = "\n".join(str(m) for m in stub_profile["messages"])
    assert "Operator-provided context" not in rendered


@pytest.mark.anyio
async def test_blank_context_omits_operator_context_section(stub_profile):
    # A GroupContext with no text and no links (e.g. Grout normalizes an
    # empty edit to None before ever sending it, but defend here too) renders
    # nothing -- no dangling "Operator-provided context:" header with no body.
    await enrich_profile(_request(context=GroupContext()))

    rendered = "\n".join(str(m) for m in stub_profile["messages"])
    assert "Operator-provided context" not in rendered


@pytest.mark.anyio
async def test_context_combines_with_categories(stub_profile):
    stub_profile["result"] = ProfileResult(
        dimensions={"channel": ["toontown"], "audience": ["kids"]},
        tags=["retro"],
    )
    resp = await enrich_profile(
        _request(
            categories=_CATEGORIES,
            context=GroupContext(text="these are retro video game ads"),
        )
    )

    rendered = "\n".join(str(m) for m in stub_profile["messages"])
    assert "these are retro video game ads" in rendered
    assert "toontown: Animated content" in rendered
    assert resp.dimensions["channel"] == ["toontown"]
