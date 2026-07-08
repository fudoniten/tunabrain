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

from tunabrain.api.models import EnrichProfileRequest
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
