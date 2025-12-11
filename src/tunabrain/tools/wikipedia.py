from __future__ import annotations

import logging
from urllib.parse import quote

import httpx


logger = logging.getLogger(__name__)


WIKIPEDIA_API = "https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
WIKIPEDIA_SUMMARY_API = "https://api.wikimedia.org/core/v1/wikipedia/en/page/summary/"
WIKIPEDIA_USER_AGENT = "TunaBrain/0.1 (+https://github.com/tunarr-labs/tunabrain)"
REQUEST_HEADERS = {"User-Agent": WIKIPEDIA_USER_AGENT}


def _build_search_query(name: str, year: int | None, imdb_id: str | None) -> str:
    if imdb_id:
        return imdb_id
    if year:
        return f"{name} ({year})"
    return name


def _fetch_summary_sync(title: str, *, debug: bool = False) -> str:
    url = f"{WIKIPEDIA_SUMMARY_API}{quote(title)}"
    if debug:
        logger.debug("Wikipedia summary request (sync): %s", url)
    with httpx.Client(headers=REQUEST_HEADERS) as client:
        summary_resp = client.get(url)
        summary_resp.raise_for_status()
        data = summary_resp.json()
    if debug:
        logger.debug(
            "Wikipedia summary response (sync) [%s]: %s",
            summary_resp.status_code,
            data,
        )
    description = data.get("description")
    extract = data.get("extract")
    lines = [f"Title: {data.get('title', title)}"]
    if description:
        lines.append(f"Description: {description}")
    if extract:
        lines.append(extract)
    return "\n".join(lines)


async def _fetch_summary(title: str, *, debug: bool = False) -> str:
    url = f"{WIKIPEDIA_SUMMARY_API}{quote(title)}"
    if debug:
        logger.debug("Wikipedia summary request (async): %s", url)
    async with httpx.AsyncClient(headers=REQUEST_HEADERS) as client:
        summary_resp = await client.get(url)
        summary_resp.raise_for_status()
        data = summary_resp.json()
    if debug:
        logger.debug(
            "Wikipedia summary response (async) [%s]: %s",
            summary_resp.status_code,
            data,
        )
    description = data.get("description")
    extract = data.get("extract")
    lines = [f"Title: {data.get('title', title)}"]
    if description:
        lines.append(f"Description: {description}")
    if extract:
        lines.append(extract)
    return "\n".join(lines)


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

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def _cache_key(self, name: str, year: int | None, imdb_id: str | None) -> str:
        if imdb_id:
            return imdb_id.lower()
        if year:
            return f"{name.lower()} ({year})"
        return name.lower()

    def lookup(self, *, name: str, year: int | None = None, imdb_id: str | None = None) -> str:
        """Synchronously fetch a Wikipedia summary, using cached results when available."""

        cache_key = self._cache_key(name, year, imdb_id)
        if cache_key in self._cache:
            if self.debug:
                logger.debug("Wikipedia cache hit for %s", cache_key)
            return self._cache[cache_key]

        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = _search_wikipedia_sync(query, debug=self.debug)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")
        summary = _fetch_summary_sync(title, debug=self.debug)
        self._cache[cache_key] = summary
        return summary

    async def lookup_async(
        self, *, name: str, year: int | None = None, imdb_id: str | None = None
    ) -> str:
        """Asynchronously fetch a Wikipedia summary, using cached results when available."""

        cache_key = self._cache_key(name, year, imdb_id)
        if cache_key in self._cache:
            if self.debug:
                logger.debug("Wikipedia cache hit for %s", cache_key)
            return self._cache[cache_key]

        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = await _search_wikipedia(query, debug=self.debug)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")
        summary = await _fetch_summary(title, debug=self.debug)
        self._cache[cache_key] = summary
        return summary
