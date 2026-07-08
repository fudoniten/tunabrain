"""Unit tests for grounding-context resolution and its round-trip through the
tagging chain.

These cover the fix for the "wrong Wikipedia page" problem: callers can now
override the auto-search with their own context, and the context actually used
is always echoed back so a bad match is diagnosable.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tunabrain.api.models import MediaContext, MediaItem
from tunabrain.chains import context as context_module
from tunabrain.chains.context import NO_CONTEXT_TEXT, resolve_media_context
from tunabrain.chains.tagging import generate_tags
from tunabrain.tools.wikipedia import page_title_from_url, page_url


def _media(title: str = "Juice") -> MediaItem:
    return MediaItem(id="1", title=title, description="A film.")


class StubWikipediaLookup:
    """Stub for ``WikipediaLookup`` that records calls and returns canned data.

    ``resolve_async`` mimics the auto-search; ``summarize_title_async`` mimics
    fetching a specific page. Both return ``(summary, page_url)``.
    """

    instances: list["StubWikipediaLookup"] = []

    def __init__(self, *, debug: bool = False, llm=None) -> None:
        self.resolve_calls: list[dict] = []
        self.title_calls: list[str] = []
        StubWikipediaLookup.instances.append(self)

    async def resolve_async(self, *, name, year=None, imdb_id=None, llm=None):
        self.resolve_calls.append({"name": name, "year": year, "imdb_id": imdb_id})
        return f"Auto summary for {name}.", page_url(f"{name} (film)")

    async def summarize_title_async(self, title, *, llm=None):
        self.title_calls.append(title)
        return f"Summary of {title}.", page_url(title)


@pytest.fixture(autouse=True)
def _stub_wikipedia(monkeypatch):
    StubWikipediaLookup.instances = []
    monkeypatch.setattr(context_module, "WikipediaLookup", StubWikipediaLookup)
    return StubWikipediaLookup


# --- URL helpers ----------------------------------------------------------------


def test_page_title_from_url_extracts_wikipedia_title():
    assert (
        page_title_from_url("https://en.wikipedia.org/wiki/Juice_(1992_film)")
        == "Juice (1992 film)"
    )


def test_page_title_from_url_rejects_non_wikipedia():
    assert page_title_from_url("https://www.imdb.com/title/tt0104573/") is None
    assert page_title_from_url("not a url at all") is None


def test_page_url_roundtrips_spaces_and_underscores():
    assert page_url("Juice (1992 film)") == page_url("Juice_(1992 film)")
    assert page_url("Juice (1992 film)").endswith("Juice_%281992_film%29")


# --- resolution precedence ------------------------------------------------------


@pytest.mark.anyio
async def test_resolve_prefers_verbatim_summary():
    ctx = MediaContext(summary="Corrected synopsis.")
    resolved = await resolve_media_context(_media(), ctx)

    assert resolved.grounding_text == "Corrected synopsis."
    assert resolved.output.summary == "Corrected synopsis."
    assert resolved.output.source == "provided-summary"
    # No Wikipedia work was done — the override wins.
    assert not StubWikipediaLookup.instances


@pytest.mark.anyio
async def test_resolve_uses_free_form_text():
    ctx = MediaContext(text="  A 1992 crime drama set in Harlem.  ")
    resolved = await resolve_media_context(_media(), ctx)

    assert resolved.grounding_text == "A 1992 crime drama set in Harlem."
    assert resolved.output.source == "provided-text"
    assert not StubWikipediaLookup.instances


@pytest.mark.anyio
async def test_resolve_fetches_wikipedia_link_and_skips_search():
    ctx = MediaContext(links=["https://en.wikipedia.org/wiki/Juice_(1992_film)"])
    resolved = await resolve_media_context(_media(), ctx)

    assert resolved.output.source == "provided-link"
    assert "Summary of Juice (1992 film)." in resolved.grounding_text
    assert resolved.output.links == [page_url("Juice (1992 film)")]
    # The specific page was fetched; the auto-search was not used.
    stub = StubWikipediaLookup.instances[0]
    assert stub.title_calls == ["Juice (1992 film)"]
    assert stub.resolve_calls == []


@pytest.mark.anyio
async def test_resolve_non_wikipedia_link_falls_back_to_search():
    ctx = MediaContext(links=["https://www.imdb.com/title/tt0104573/"])
    resolved = await resolve_media_context(_media(), ctx)

    # No Wikipedia link to fetch, so the auto-search runs as a fallback.
    assert resolved.output.source == "wikipedia"
    # The auto-search ran; no specific page was ever fetched.
    all_resolve_calls = [c for s in StubWikipediaLookup.instances for c in s.resolve_calls]
    all_title_calls = [c for s in StubWikipediaLookup.instances for c in s.title_calls]
    assert all_resolve_calls and not all_title_calls


@pytest.mark.anyio
async def test_resolve_auto_search_returns_matched_page():
    resolved = await resolve_media_context(_media("Juice"), None)

    assert resolved.output.source == "wikipedia"
    assert resolved.grounding_text == "Auto summary for Juice."
    # The matched page is surfaced so a wrong match is diagnosable.
    assert resolved.output.links == [page_url("Juice (film)")]


@pytest.mark.anyio
async def test_resolve_skips_search_for_placeholder_title():
    # A title that reduces to a placeholder must never drive a search — this is
    # the "<unnamed>" -> "Unnamed Memory" bug.
    for junk in ("Unknown", "<unnamed>", "Untitled", "12345"):
        StubWikipediaLookup.instances = []
        resolved = await resolve_media_context(_media(junk), None)
        assert resolved.output.source == "none"
        assert resolved.grounding_text == NO_CONTEXT_TEXT
        # No WikipediaLookup was even constructed.
        assert not StubWikipediaLookup.instances


@pytest.mark.anyio
async def test_resolve_skips_search_when_disabled(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        context_module,
        "get_settings",
        lambda: SimpleNamespace(enable_wikipedia_search=False),
    )
    # A real (non-placeholder) title, but the auto-search is disabled globally.
    resolved = await resolve_media_context(_media("Juice"), None)
    assert resolved.output.source == "none"
    assert resolved.grounding_text == NO_CONTEXT_TEXT
    assert not StubWikipediaLookup.instances


@pytest.mark.anyio
async def test_resolve_handles_no_match(monkeypatch):
    class NoMatchLookup(StubWikipediaLookup):
        async def resolve_async(self, *, name, year=None, imdb_id=None, llm=None):
            return None

    monkeypatch.setattr(context_module, "WikipediaLookup", NoMatchLookup)
    resolved = await resolve_media_context(_media(), None)

    assert resolved.grounding_text == NO_CONTEXT_TEXT
    assert resolved.output.source == "none"


# --- round-trip through generate_tags ------------------------------------------


class RecordingLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self._responses.pop(0))


@pytest.mark.anyio
async def test_generate_tags_echoes_context_and_uses_override(monkeypatch):
    llm = RecordingLLM(['{"tags": ["crime-drama", "gritty"]}'])
    monkeypatch.setattr("tunabrain.chains.tagging.get_chat_model", lambda task=None: llm)

    ctx = MediaContext(summary="Juice (1992) is a violent crime drama set in Harlem.")
    tags, out_ctx = await generate_tags(_media(), context=ctx)

    assert tags == ["crime-drama", "gritty"]
    # The corrected context was echoed back for storage.
    assert out_ctx.source == "provided-summary"
    assert out_ctx.summary == "Juice (1992) is a violent crime drama set in Harlem."
    # The override text reached the prompt.
    human_message = llm.calls[0][-1]
    assert "violent crime drama" in human_message.content
