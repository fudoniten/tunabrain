from __future__ import annotations

from typing import Optional
from urllib.parse import quote

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"


class WikipediaMediaLookupInput(BaseModel):
    """Input schema for looking up a media item on Wikipedia."""

    name: str = Field(..., description="Primary title of the media item")
    year: Optional[int] = Field(
        None, description="Release year used to disambiguate titles when IMDB ID is absent"
    )
    imdb_id: Optional[str] = Field(
        None, description="IMDB identifier for the media item, e.g. tt0149460"
    )


def _build_search_query(name: str, year: Optional[int], imdb_id: Optional[str]) -> str:
    if imdb_id:
        return imdb_id
    if year:
        return f"{name} ({year})"
    return name


def _fetch_summary_sync(title: str) -> str:
    with httpx.Client() as client:
        summary_resp = client.get(f"{WIKIPEDIA_SUMMARY_API}{quote(title)}")
        summary_resp.raise_for_status()
        data = summary_resp.json()
    description = data.get("description")
    extract = data.get("extract")
    lines = [f"Title: {data.get('title', title)}"]
    if description:
        lines.append(f"Description: {description}")
    if extract:
        lines.append(extract)
    return "\n".join(lines)


async def _fetch_summary(title: str) -> str:
    async with httpx.AsyncClient() as client:
        summary_resp = await client.get(f"{WIKIPEDIA_SUMMARY_API}{quote(title)}")
        summary_resp.raise_for_status()
        data = summary_resp.json()
    description = data.get("description")
    extract = data.get("extract")
    lines = [f"Title: {data.get('title', title)}"]
    if description:
        lines.append(f"Description: {description}")
    if extract:
        lines.append(extract)
    return "\n".join(lines)


def _search_wikipedia_sync(query: str) -> Optional[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    }
    with httpx.Client() as client:
        resp = client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return None
    return search_results[0].get("title")


async def _search_wikipedia(query: str) -> Optional[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return None
    return search_results[0].get("title")


class WikipediaLookupTool(BaseTool):
    """Retrieve a scheduling-oriented summary of a media item from Wikipedia."""

    name = "wikipedia_media_lookup"
    description = (
        "Look up a media item on Wikipedia using an IMDB ID when available or the title and "
        "release year, returning a concise synopsis for scheduling decisions."
    )
    args_schema = WikipediaMediaLookupInput

    def _run(self, name: str, year: Optional[int] = None, imdb_id: Optional[str] = None) -> str:  # type: ignore[override]
        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = _search_wikipedia_sync(query)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")
        return _fetch_summary_sync(title)

    async def _arun(
        self, name: str, year: Optional[int] = None, imdb_id: Optional[str] = None
    ) -> str:  # type: ignore[override]
        query = _build_search_query(name=name, year=year, imdb_id=imdb_id)
        title = await _search_wikipedia(query)
        if not title:
            raise ValueError(f"No Wikipedia entry found for query: {query}")
        return await _fetch_summary(title)
