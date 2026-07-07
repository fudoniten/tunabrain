"""Unit tests for the /enrich/describe chain.

These cover the describe-only building block: title validation, grounding
context resolution, the happy path (refined title + one-sentence description),
the null-description case, and graceful degradation to the working title when
the LLM call fails. The LLM and the Wikipedia auto-search are stubbed so the
tests never make a network call.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tunabrain.api.models import (
    EnrichDescribeRequest,
    MediaContext,
    MediaItem,
)
from tunabrain.chains import describe as describe_mod
from tunabrain.chains.context import ResolvedContext
from tunabrain.chains.describe import DescribeResult, describe_media


def _media(**overrides) -> MediaItem:
    base = {
        "id": "grout-1",
        "title": "2025-09-28 Animation School - Keeping motivated - (2025) (2025)",
        "duration_minutes": 9,
    }
    base.update(overrides)
    return MediaItem(**base)


@pytest.fixture
def stub_describe(monkeypatch):
    """Stub resolve_media_context + the LLM so describe runs offline.

    The returned dict records what the chain saw and lets each test set the
    structured result the fake LLM should produce.
    """
    calls: dict[str, object] = {}
    calls["result"] = DescribeResult(
        title="Animation School: Keeping Motivated",
        description="A 2025 animated short on staying motivated through a long project.",
    )

    async def fake_resolve(media, context=None, *, llm=None, debug=False):
        calls["resolve_context"] = context
        return ResolvedContext(
            "Grounding summary fed to the model",
            MediaContext(summary="resolved summary", source="wikipedia", links=["http://x"]),
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
            return "raw-llm-response"

    monkeypatch.setattr(describe_mod, "resolve_media_context", fake_resolve)
    monkeypatch.setattr(describe_mod, "PydanticOutputParser", FakeParser)
    monkeypatch.setattr(describe_mod, "get_chat_model", lambda *a, **k: FakeLLM())
    return calls


def test_empty_title_is_rejected():
    """An empty/whitespace title fails validation (surfaces as 422 at the route)."""
    with pytest.raises(ValidationError):
        EnrichDescribeRequest(media=MediaItem(id="grout-1", title="   "))


@pytest.mark.anyio
async def test_describe_returns_refined_title_and_description(stub_describe):
    resp = await describe_media(_media())

    assert resp.media.id == "grout-1"
    assert resp.media.title == "Animation School: Keeping Motivated"
    assert resp.media.description.startswith("A 2025 animated short")
    assert resp.warnings == []
    assert resp.cost_estimate.llm_calls_used == 1
    # Resolved grounding context is echoed back for storage/correction.
    assert resp.context.summary == "resolved summary"
    assert resp.context.source == "wikipedia"


@pytest.mark.anyio
async def test_describe_forwards_context_override(stub_describe):
    override = MediaContext(text="operator note about the video")
    await describe_media(_media(), override)

    # The caller's context is handed to resolution to skip the auto-search.
    assert stub_describe["resolve_context"] is override


@pytest.mark.anyio
async def test_describe_allows_null_description(stub_describe):
    stub_describe["result"] = DescribeResult(title="Channel Ident", description=None)
    resp = await describe_media(_media(title="ident-5s.mp4", duration_minutes=None))

    assert resp.media.title == "Channel Ident"
    assert resp.media.description is None
    assert resp.warnings == []


@pytest.mark.anyio
async def test_describe_keeps_working_title_when_model_returns_empty(stub_describe):
    stub_describe["result"] = DescribeResult(title="   ", description=None)
    resp = await describe_media(_media())

    # Never returns an empty title: falls back to the working title with a warning.
    assert resp.media.title == _media().title
    assert any("empty title" in w for w in resp.warnings)


@pytest.mark.anyio
async def test_describe_degrades_to_working_title_on_llm_failure(monkeypatch):
    async def fake_resolve(media, context=None, *, llm=None, debug=False):
        return ResolvedContext("grounding", MediaContext(source="none"))

    class BoomLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("llm exploded")

    monkeypatch.setattr(describe_mod, "resolve_media_context", fake_resolve)
    monkeypatch.setattr(describe_mod, "get_chat_model", lambda *a, **k: BoomLLM())

    media = _media()
    resp = await describe_media(media)

    # A failed LLM call is a warning, not a 500: the working title is returned.
    assert resp.media.title == media.title
    assert any("describe failed" in w for w in resp.warnings)
    assert resp.cost_estimate.llm_calls_used == 1
