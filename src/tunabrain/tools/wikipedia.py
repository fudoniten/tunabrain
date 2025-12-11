from __future__ import annotations

import logging

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from tunabrain.llm import get_chat_model


logger = logging.getLogger(__name__)


WIKIPEDIA_API = "https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
WIKIPEDIA_PAGE_EXTRACT_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "TunaBrain/0.1 (+https://github.com/tunarr-labs/tunabrain)"
REQUEST_HEADERS = {"User-Agent": WIKIPEDIA_USER_AGENT}


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
