"""Unit tests for the /enrich/short-form orchestration chain.

These cover the wiring between the endpoint and the existing /categorize + /tags
building blocks: the categories catalog is forwarded verbatim, the categorize
context is propagated into tags, and each sub-call degrades gracefully to a
warning rather than a hard failure.
"""

from __future__ import annotations

import pytest

from tunabrain.api.models import (
    CategoryDefinition,
    Channel,
    DimensionSelection,
    EnrichShortFormRequest,
    MediaContext,
    MediaItem,
)
from tunabrain.chains import enrich_short
from tunabrain.chains.categorization import CategorizationResult
from tunabrain.chains.enrich_short import run_enrich_short_form


def _media() -> MediaItem:
    return MediaItem(id="grout-1", title="mystery-bumper-5e0ff26a", duration_minutes=1)


def _categories() -> dict[str, CategoryDefinition]:
    return {
        "audience": CategoryDefinition(description="Time-of-day suitability", values=["daytime"]),
        "channel": CategoryDefinition(description="Target channel", values=["goldenreels"]),
    }


@pytest.fixture
def stub_chain(monkeypatch):
    """Stub categorize_media + generate_tags with recording fakes."""
    calls: dict[str, object] = {}

    async def fake_categorize(*, media, categories, channels, debug, context):
        calls["categorize"] = {
            "categories": categories,
            "channels": channels,
            "context": context,
        }
        return CategorizationResult(
            dimensions=[DimensionSelection(dimension="audience", values=["daytime"])],
            channel_mappings=[],
            context=MediaContext(summary="resolved by categorize", source="provided-summary"),
        )

    async def fake_generate_tags(media, existing_tags=None, *, debug=False, context=None):
        calls["tags"] = {"existing_tags": existing_tags, "context": context}
        return ["filler", "short"], MediaContext(
            summary="resolved by tags", source="provided-summary"
        )

    monkeypatch.setattr(enrich_short, "categorize_media", fake_categorize)
    monkeypatch.setattr(enrich_short, "generate_tags", fake_generate_tags)
    return calls


@pytest.mark.anyio
async def test_enrich_short_form_passes_categories_through_to_categorize(stub_chain):
    categories = _categories()
    req = EnrichShortFormRequest(
        media=_media(), categories=categories, channels=[Channel(name="goldenreels")]
    )
    await run_enrich_short_form(req)

    # The categories dict is forwarded verbatim.
    assert stub_chain["categorize"]["categories"] == categories
    assert [c.name for c in stub_chain["categorize"]["channels"]] == ["goldenreels"]


@pytest.mark.anyio
async def test_enrich_short_form_propagates_context_from_categorize_to_tags(stub_chain):
    req = EnrichShortFormRequest(media=_media(), categories=_categories())
    await run_enrich_short_form(req)

    # The context resolved by categorize is fed into the tags call.
    tags_context = stub_chain["tags"]["context"]
    assert isinstance(tags_context, MediaContext)
    assert tags_context.summary == "resolved by categorize"


@pytest.mark.anyio
async def test_enrich_short_form_returns_combined_response(stub_chain):
    req = EnrichShortFormRequest(media=_media(), categories=_categories())
    resp = await run_enrich_short_form(req)

    assert [d.dimension for d in resp.dimensions] == ["audience"]
    assert resp.tags == ["filler", "short"]
    # The last context echoed (tags ran second) is what's returned for storage.
    assert resp.context.summary == "resolved by tags"
    assert resp.warnings == []
    assert resp.cost_estimate.llm_calls_used >= 1


@pytest.mark.anyio
async def test_enrich_short_form_handles_categorize_failure(monkeypatch):
    async def boom_categorize(**kwargs):
        raise RuntimeError("categorize exploded")

    captured: dict[str, object] = {}

    async def fake_generate_tags(media, existing_tags=None, *, debug=False, context=None):
        captured["context"] = context
        return ["still-tagged"], MediaContext(summary="tags ran", source="none")

    monkeypatch.setattr(enrich_short, "categorize_media", boom_categorize)
    monkeypatch.setattr(enrich_short, "generate_tags", fake_generate_tags)

    req = EnrichShortFormRequest(media=_media(), categories=_categories())
    resp = await run_enrich_short_form(req)

    # Categorize failed: warning present, dimensions empty, tags still attempted.
    assert any("categorize failed" in w for w in resp.warnings)
    assert resp.dimensions == []
    assert resp.tags == ["still-tagged"]
    # With no categorize context, the request's context (None here) is propagated.
    assert captured["context"] is None


@pytest.mark.anyio
async def test_enrich_short_form_handles_tags_failure(monkeypatch):
    async def fake_categorize(**kwargs):
        return CategorizationResult(
            dimensions=[DimensionSelection(dimension="audience", values=["daytime"])],
            channel_mappings=[],
            context=MediaContext(summary="cat ctx", source="provided-summary"),
        )

    async def boom_tags(*args, **kwargs):
        raise RuntimeError("tags exploded")

    monkeypatch.setattr(enrich_short, "categorize_media", fake_categorize)
    monkeypatch.setattr(enrich_short, "generate_tags", boom_tags)

    req = EnrichShortFormRequest(media=_media(), categories=_categories())
    resp = await run_enrich_short_form(req)

    # Tags failed: warning present, dimensions still returned.
    assert any("tags failed" in w for w in resp.warnings)
    assert [d.dimension for d in resp.dimensions] == ["audience"]
    assert resp.tags == []
    # Falls back to the categorize context for storage.
    assert resp.context.summary == "cat ctx"
