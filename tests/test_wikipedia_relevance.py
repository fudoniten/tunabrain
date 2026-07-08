"""Tests for the Wikipedia auto-search relevance gate.

The gate is the fix for blindly trusting Wikipedia's top hit: the search returns
a result for almost any string, so the auto-search now fetches several
candidates and lets an LLM reject them all when none is a genuine match. These
tests stub the search + LLM so no network or real model is used.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tunabrain.tools import wikipedia as wiki
from tunabrain.tools.wikipedia import WikiCandidate, WikipediaLookup


class FakeLLM:
    """Returns a fixed content string for every ainvoke (the gate verdict)."""

    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._content)


def _candidates():
    return [
        WikiCandidate("Unnamed Memory", "Japanese light novel series", "a fantasy anime"),
        WikiCandidate("Nameless", "2010 film", None),
    ]


@pytest.mark.anyio
async def test_gate_picks_confident_match():
    llm = FakeLLM('{"best_match_index": 2, "reason": "same film"}')
    lookup = WikipediaLookup(llm=llm)
    chosen = await lookup._select_relevant_candidate("Nameless", _candidates())
    assert chosen is not None
    assert chosen.title == "Nameless"


@pytest.mark.anyio
async def test_gate_returns_none_when_no_match():
    llm = FakeLLM('{"best_match_index": null, "reason": "obscure clip, not on wiki"}')
    lookup = WikipediaLookup(llm=llm)
    chosen = await lookup._select_relevant_candidate("mystery-bumper-5e0ff", _candidates())
    assert chosen is None


@pytest.mark.anyio
async def test_gate_returns_none_on_out_of_range_index():
    llm = FakeLLM('{"best_match_index": 9, "reason": "hallucinated index"}')
    lookup = WikipediaLookup(llm=llm)
    chosen = await lookup._select_relevant_candidate("whatever", _candidates())
    assert chosen is None


@pytest.mark.anyio
async def test_gate_returns_none_on_unparseable_verdict():
    llm = FakeLLM("not json at all")
    lookup = WikipediaLookup(llm=llm)
    chosen = await lookup._select_relevant_candidate("whatever", _candidates())
    assert chosen is None


@pytest.mark.anyio
async def test_resolve_async_returns_none_when_gate_rejects(monkeypatch):
    async def fake_candidates(query, *, limit=5, debug=False):
        return _candidates()

    monkeypatch.setattr(wiki, "_search_wikipedia_candidates", fake_candidates)
    # Gate rejects everything.
    llm = FakeLLM('{"best_match_index": null, "reason": "none match"}')
    lookup = WikipediaLookup(llm=llm)

    result = await lookup.resolve_async(name="mystery-bumper-5e0ff")
    assert result is None


@pytest.mark.anyio
async def test_resolve_async_summarizes_the_gated_match(monkeypatch):
    async def fake_candidates(query, *, limit=5, debug=False):
        return _candidates()

    captured = {}

    async def fake_summarize(self, title, *, llm=None):
        captured["title"] = title
        return f"Summary of {title}.", f"http://en.wikipedia.org/wiki/{title}"

    monkeypatch.setattr(wiki, "_search_wikipedia_candidates", fake_candidates)
    monkeypatch.setattr(WikipediaLookup, "summarize_title_async", fake_summarize)
    llm = FakeLLM('{"best_match_index": 1, "reason": "match"}')
    lookup = WikipediaLookup(llm=llm)

    result = await lookup.resolve_async(name="Unnamed Memory")
    assert result is not None
    assert captured["title"] == "Unnamed Memory"


@pytest.mark.anyio
async def test_resolve_async_trusts_top_hit_for_imdb_id(monkeypatch):
    async def fake_search(query, *, debug=False):
        return "Some Movie (2019 film)"

    async def fake_candidates(*a, **k):  # should NOT be called on the imdb path
        raise AssertionError("candidate search should be skipped for imdb_id")

    async def fake_summarize(self, title, *, llm=None):
        return f"Summary of {title}.", f"http://en.wikipedia.org/wiki/{title}"

    monkeypatch.setattr(wiki, "_search_wikipedia", fake_search)
    monkeypatch.setattr(wiki, "_search_wikipedia_candidates", fake_candidates)
    monkeypatch.setattr(WikipediaLookup, "summarize_title_async", fake_summarize)
    # No gate LLM should be consulted on the imdb path.
    llm = FakeLLM("should not be used")
    lookup = WikipediaLookup(llm=llm)

    result = await lookup.resolve_async(name="whatever", imdb_id="tt1234567")
    assert result is not None
    assert llm.calls == 0
