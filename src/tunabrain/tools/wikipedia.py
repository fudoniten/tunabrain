from __future__ import annotations

import logging
from urllib.parse import quote

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"


class WikipediaMediaLookupInput(BaseModel):
    """Input schema for looking up a media item on Wikipedia."""

    name: str = Field(..., description="Primary title of the media item")
    year: int | None = Field(
        None, description="Release year used to disambiguate titles when IMDB ID is absent"
    )
    imdb_id: str | None = Field(
        None, description="IMDB identifier for the media item, e.g. tt0149460"
    )


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
    with httpx.Client() as client:
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
    async with httpx.AsyncClient() as client:
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
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    }
    if debug:
        logger.debug("Wikipedia search request (sync): %s params=%s", WIKIPEDIA_API, params)
    with httpx.Client() as client:
        resp = client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    if debug:
        logger.debug(
            "Wikipedia search response (sync) [%s]: %s", resp.status_code, data
        )
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return None
    return search_results[0].get("title")


async def _search_wikipedia(query: str, *, debug: bool = False) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    }
    if debug:
        logger.debug("Wikipedia search request (async): %s params=%s", WIKIPEDIA_API, params)
    async with httpx.AsyncClient() as client:
        resp = await client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    if debug:
        logger.debug(
            "Wikipedia search response (async) [%s]: %s", resp.status_code, data
        )
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return None
    return search_results[0].get("title")


class WikipediaLookupTool(BaseTool):
    """Retrieve a scheduling-oriented summary of a media item from Wikipedia."""

    name: str = "wikipedia_media_lookup"
    description: str = (
        "Look up a media item on Wikipedia using an IMDB ID when available or the title and "
        "release year, returning a concise synopsis for scheduling decisions."
    )
    args_schema: type[WikipediaMediaLookupInput] = WikipediaMediaLookupInput
    debug: bool = False

    def _run(self, name: str, year: int | None = None, imdb_id: str | None = None) -> str:  # type: ignore[override]
        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = _search_wikipedia_sync(query, debug=self.debug)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")
        return _fetch_summary_sync(title, debug=self.debug)

    async def _arun(
        self, name: str, year: int | None = None, imdb_id: str | None = None
    ) -> str:  # type: ignore[override]
        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = await _search_wikipedia(query, debug=self.debug)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")
        return await _fetch_summary(title, debug=self.debug)
