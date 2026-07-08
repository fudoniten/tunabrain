from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tunabrain.llm import get_chat_model
from tunabrain.tools.titles import clean_search_query

logger = logging.getLogger(__name__)

# How many search hits to consider for the relevance gate. Wikipedia's search
# returns *something* for almost any query, so the auto-search fetches a handful
# of candidates and lets an LLM decide whether any is a genuine match rather than
# blindly trusting the top hit.
_CANDIDATE_LIMIT = 5


WIKIPEDIA_API = "https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
WIKIPEDIA_PAGE_EXTRACT_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "TunaBrain/0.1 (+https://github.com/tunarr-labs/tunabrain)"
REQUEST_HEADERS = {"User-Agent": WIKIPEDIA_USER_AGENT}


def page_url(title: str) -> str:
    """Build the canonical en.wikipedia.org URL for a page title.

    Accepts either a spaced ("Juice (1992 film)") or underscored
    ("Juice_(1992_film)") title; the result is always the underscored,
    percent-encoded article URL.
    """
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def page_title_from_url(url: str) -> str | None:
    """Extract a Wikipedia page title from a URL, or None if it isn't one.

    Only recognises ``*.wikipedia.org/wiki/<title>`` links; every other URL
    returns ``None`` so the caller can decide how to treat non-Wikipedia
    references. Underscores are converted back to spaces and percent-encoding
    is decoded so the title can be passed straight to the extract API.
    """
    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return None
    if "wikipedia.org" not in (parsed.netloc or ""):
        return None
    prefix = "/wiki/"
    path = parsed.path or ""
    if not path.startswith(prefix):
        return None
    title = unquote(path[len(prefix):]).replace("_", " ").strip()
    return title or None


def _schedule_summary_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize Wikipedia articles for content scheduling. Focus on the plot,"
                " main events, tone, themes, and viewer considerations. Explicitly state the"
                " release date, time period, setting, and any adult content (violence, sex,"
                " language). Ignore cast lists, production notes, gossip, or reception.",
            ),
            (
                "human",
                "Title: {title}\n"
                "Full article text:\n{article}\n\n"
                "Write a concise paragraph (4-6 sentences) that highlights the narrative,"
                " tone, themes, era/setting, notable adult content, and release date for"
                " scheduling use.",
            ),
        ]
    )


def _build_search_query(name: str, year: int | None, imdb_id: str | None) -> str:
    if imdb_id:
        return imdb_id
    if year:
        return f"{name} ({year})"
    return name


def _extract_article_text(data: dict) -> str | None:
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    extract = page.get("extract")
    if extract:
        return str(extract)
    return None


def _fetch_full_article_sync(title: str, *, debug: bool = False) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
    }
    if debug:
        logger.debug(
            "Wikipedia full article request (sync): %s params=%s", WIKIPEDIA_PAGE_EXTRACT_API, params
        )
    with httpx.Client(headers=REQUEST_HEADERS) as client:
        resp = client.get(WIKIPEDIA_PAGE_EXTRACT_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    if debug:
        logger.debug("Wikipedia full article response (sync) [%s]", resp.status_code)
    article = _extract_article_text(data)
    if not article:
        raise ValueError(f"Failed to retrieve Wikipedia article text for: {title}")
    return article


async def _fetch_full_article(title: str, *, debug: bool = False) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
    }
    if debug:
        logger.debug(
            "Wikipedia full article request (async): %s params=%s", WIKIPEDIA_PAGE_EXTRACT_API, params
        )
    async with httpx.AsyncClient(headers=REQUEST_HEADERS) as client:
        resp = await client.get(WIKIPEDIA_PAGE_EXTRACT_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    if debug:
        logger.debug("Wikipedia full article response (async) [%s]", resp.status_code)
    article = _extract_article_text(data)
    if not article:
        raise ValueError(f"Failed to retrieve Wikipedia article text for: {title}")
    return article


def _summarize_article_sync(
    llm: BaseChatModel, *, title: str, article: str, debug: bool = False
) -> str:
    prompt = _schedule_summary_prompt()
    messages = prompt.format_messages(title=title, article=article)
    if debug:
        logger.debug(
            "Wikipedia scheduling summary request (sync): title=%s length=%s",
            title,
            len(article),
        )
    response = llm.invoke(messages)
    content = getattr(response, "content", str(response))
    return content.strip()


async def _summarize_article_async(
    llm: BaseChatModel, *, title: str, article: str, debug: bool = False
) -> str:
    prompt = _schedule_summary_prompt()
    messages = prompt.format_messages(title=title, article=article)
    if debug:
        logger.debug(
            "Wikipedia scheduling summary request (async): title=%s length=%s",
            title,
            len(article),
        )
    response = await llm.ainvoke(messages)
    content = getattr(response, "content", str(response))
    return content.strip()


def _search_wikipedia_sync(query: str, *, debug: bool = False) -> str | None:
    params = {"q": query, "limit": 1}
    if debug:
        logger.debug("Wikipedia search request (sync): %s params=%s", WIKIPEDIA_API, params)
    with httpx.Client(headers=REQUEST_HEADERS) as client:
        resp = client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    if debug:
        logger.debug(
            "Wikipedia search response (sync) [%s]: %s", resp.status_code, data
        )
    search_results = data.get("pages", [])
    if not search_results:
        return None
    top_result = search_results[0]
    return top_result.get("key") or top_result.get("title")


async def _search_wikipedia(query: str, *, debug: bool = False) -> str | None:
    params = {"q": query, "limit": 1}
    if debug:
        logger.debug("Wikipedia search request (async): %s params=%s", WIKIPEDIA_API, params)
    async with httpx.AsyncClient(headers=REQUEST_HEADERS) as client:
        resp = await client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    if debug:
        logger.debug(
            "Wikipedia search response (async) [%s]: %s", resp.status_code, data
        )
    search_results = data.get("pages", [])
    if not search_results:
        return None
    top_result = search_results[0]
    return top_result.get("key") or top_result.get("title")


@dataclass
class WikiCandidate:
    """One Wikipedia search hit, with the snippets used by the relevance gate."""

    title: str
    description: str | None = None
    excerpt: str | None = None


def _clean_excerpt(excerpt: str | None) -> str | None:
    """Strip the ``<span class="searchmatch">`` markup Wikipedia returns."""
    if not excerpt:
        return None
    return re.sub(r"<[^>]+>", "", excerpt).strip() or None


async def _search_wikipedia_candidates(
    query: str, *, limit: int = _CANDIDATE_LIMIT, debug: bool = False
) -> list[WikiCandidate]:
    """Return up to ``limit`` search candidates (title + snippets) for ``query``.

    Unlike :func:`_search_wikipedia`, which returns only the top hit's title,
    this preserves each hit's ``description``/``excerpt`` so the relevance gate
    can judge whether any candidate genuinely matches the media.
    """
    params = {"q": query, "limit": limit}
    if debug:
        logger.debug("Wikipedia candidate search: %s params=%s", WIKIPEDIA_API, params)
    async with httpx.AsyncClient(headers=REQUEST_HEADERS) as client:
        resp = await client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    candidates: list[WikiCandidate] = []
    for page in data.get("pages", []):
        title = page.get("title") or page.get("key")
        if not title:
            continue
        candidates.append(
            WikiCandidate(
                title=title,
                description=page.get("description"),
                excerpt=_clean_excerpt(page.get("excerpt")),
            )
        )
    if debug:
        logger.debug(
            "Wikipedia candidates for %r: %s", query, [c.title for c in candidates]
        )
    return candidates


class _RelevanceVerdict(BaseModel):
    """The relevance gate's decision about which candidate (if any) matches."""

    best_match_index: int | None = Field(
        None,
        description=(
            "1-based index of the candidate that is clearly the same work as the "
            "media, or null if none is a confident match."
        ),
    )
    reason: str = Field("", description="Brief justification for the choice.")


def _relevance_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You match a media item to a Wikipedia article. You are given the "
                "media's working title (which may be a filename or a generic "
                "placeholder) and a numbered list of Wikipedia search candidates. "
                "Choose the candidate that is unmistakably the SAME work — same "
                "title and same kind of thing. Be skeptical: Wikipedia search "
                "returns a hit for almost any string, and most of this media is "
                "obscure and NOT on Wikipedia. If no candidate is clearly the same "
                "work, return null. A loose thematic or keyword overlap is NOT a "
                "match. Prefer null over a doubtful guess.",
            ),
            (
                "human",
                "Media working title: {title}\n"
                "Extra hints (may be blank): {hints}\n\n"
                "Wikipedia candidates:\n{candidates}\n\n"
                "Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )


class WikipediaLookup:
    """Retrieve and cache Wikipedia summaries for media items.

    This class always prefers IMDB IDs for disambiguation, falling back to the
    provided title (and optional year). Results are cached per query so
    repeated calls do not trigger new HTTP requests.
    """

    _cache: dict[str, str] = {}

    def __init__(self, *, debug: bool = False, llm: BaseChatModel | None = None) -> None:
        self.debug = debug
        self._llm = llm

    def _cache_key(self, name: str, year: int | None, imdb_id: str | None) -> str:
        if imdb_id:
            return imdb_id.lower()
        if year:
            return f"{name.lower()} ({year})"
        return name.lower()

    def lookup(
        self,
        *,
        name: str,
        year: int | None = None,
        imdb_id: str | None = None,
        llm: BaseChatModel | None = None,
    ) -> str:
        """Synchronously fetch and summarize a Wikipedia article for scheduling."""

        cache_key = self._cache_key(name, year, imdb_id)
        if cache_key in self._cache:
            if self.debug:
                logger.debug("Wikipedia cache hit for %s", cache_key)
            return self._cache[cache_key]

        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = _search_wikipedia_sync(query, debug=self.debug)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")

        article = _fetch_full_article_sync(title, debug=self.debug)
        summary = _summarize_article_sync(
            llm or self._llm or get_chat_model(),
            title=title,
            article=article,
            debug=self.debug,
        )

        self._cache[cache_key] = summary
        return summary

    async def lookup_async(
        self,
        *,
        name: str,
        year: int | None = None,
        imdb_id: str | None = None,
        llm: BaseChatModel | None = None,
    ) -> str:
        """Asynchronously fetch and summarize a Wikipedia article for scheduling."""

        cache_key = self._cache_key(name, year, imdb_id)
        if cache_key in self._cache:
            if self.debug:
                logger.debug("Wikipedia cache hit for %s", cache_key)
            return self._cache[cache_key]

        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = await _search_wikipedia(query, debug=self.debug)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")

        article = await _fetch_full_article(title, debug=self.debug)
        summary = await _summarize_article_async(
            llm or self._llm or get_chat_model(),
            title=title,
            article=article,
            debug=self.debug,
        )

        self._cache[cache_key] = summary
        return summary

    async def summarize_title_async(
        self, title: str, *, llm: BaseChatModel | None = None
    ) -> tuple[str, str]:
        """Fetch and summarize a specific Wikipedia page by title.

        Unlike :meth:`lookup_async`, this skips the search step — the caller
        already knows the exact page (e.g. from an operator-supplied link).
        Returns ``(summary, page_url)`` so the resolved article is visible to
        the caller.
        """
        article = await _fetch_full_article(title, debug=self.debug)
        summary = await _summarize_article_async(
            llm or self._llm or get_chat_model(),
            title=title,
            article=article,
            debug=self.debug,
        )
        return summary, page_url(title)

    async def _select_relevant_candidate(
        self,
        name: str,
        candidates: list[WikiCandidate],
        *,
        hints: str = "",
        llm: BaseChatModel | None = None,
    ) -> WikiCandidate | None:
        """Ask the LLM which candidate (if any) is a genuine match for ``name``.

        Returns the chosen :class:`WikiCandidate`, or ``None`` when no candidate
        is a confident match. Any failure to parse a decision is treated as "no
        match" so a bad guess is never forced downstream.
        """
        if not candidates:
            return None

        parser = PydanticOutputParser(pydantic_object=_RelevanceVerdict)
        lines = []
        for i, cand in enumerate(candidates, start=1):
            detail = " — ".join(p for p in (cand.description, cand.excerpt) if p)
            lines.append(f"{i}. {cand.title}" + (f" — {detail}" if detail else ""))
        messages = _relevance_prompt().format_messages(
            title=name,
            hints=hints or "none",
            candidates="\n".join(lines),
            format_instructions=f"\n\n{parser.get_format_instructions()}",
        )
        model = llm or self._llm or get_chat_model()
        try:
            response = await model.ainvoke(messages)
            verdict = await parser.ainvoke(response)
        except Exception as exc:  # pragma: no cover - defensive; treat as no match
            logger.warning("Wikipedia relevance gate failed for %r: %s", name, exc)
            return None

        idx = verdict.best_match_index
        if idx is None or not (1 <= idx <= len(candidates)):
            logger.info(
                "Wikipedia relevance gate found no confident match for %r (reason: %s)",
                name,
                verdict.reason or "n/a",
            )
            return None
        chosen = candidates[idx - 1]
        if self.debug:
            logger.debug("Wikipedia relevance gate chose %r for %r", chosen.title, name)
        return chosen

    async def resolve_async(
        self,
        *,
        name: str,
        year: int | None = None,
        imdb_id: str | None = None,
        llm: BaseChatModel | None = None,
    ) -> tuple[str, str] | None:
        """Search, gate, fetch, and summarize — returning ``(summary, page_url)``.

        This is the visibility-preserving variant of :meth:`lookup_async`: it
        exposes *which* page was used so a bad match can be diagnosed downstream.
        Returns ``None`` when the search finds nothing, or when the relevance
        gate rejects every candidate, rather than forcing a wrong match.

        An ``imdb_id`` is a precise identifier, so when one is supplied the top
        hit is trusted directly and the LLM gate is skipped.
        """
        if imdb_id:
            query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
            title = await _search_wikipedia(query, debug=self.debug)
            if not title:
                return None
            return await self.summarize_title_async(title, llm=llm)

        query = clean_search_query(name)
        candidates = await _search_wikipedia_candidates(query, debug=self.debug)
        if not candidates:
            return None
        hints = f"released {year}" if year else ""
        chosen = await self._select_relevant_candidate(
            name, candidates, hints=hints, llm=llm
        )
        if chosen is None:
            return None
        return await self.summarize_title_async(chosen.title, llm=llm)
