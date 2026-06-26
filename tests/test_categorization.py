import pytest
from langchain_core.messages import AIMessage

from tunabrain.api.models import CategoryDefinition, DimensionSelection, MediaItem
from tunabrain.chains import categorization
from tunabrain.chains.categorization import (
    _channel_mappings_from_dimensions,
    categorize_media,
)


class StubLLM:
    def __init__(self, responses: dict[str, str]):
        # Map a substring found in the prompt to the canned response.
        self._responses = responses

    async def ainvoke(self, messages):
        text = "".join(getattr(m, "content", "") for m in messages)
        for needle, response in self._responses.items():
            if needle in text:
                return AIMessage(content=response)
        raise RuntimeError(f"No stub response matched prompt: {text[:200]}")


def test_channel_mappings_from_dimensions_bridges_channel_dimension():
    dimensions = [
        DimensionSelection(dimension="audience", values=["adult"], notes=["mature"]),
        DimensionSelection(
            dimension="channel",
            values=["toontown", "nippon", "galaxy"],
            notes=["anime with sci-fi elements"],
        ),
    ]

    mappings = _channel_mappings_from_dimensions(dimensions)

    assert [m.channel_name for m in mappings] == ["toontown", "nippon", "galaxy"]
    # Notes are carried over as reasons for each chosen channel.
    for mapping in mappings:
        assert mapping.reasons == ["anime with sci-fi elements"]


def test_channel_mappings_from_dimensions_ignores_non_channel_dimensions():
    dimensions = [
        DimensionSelection(dimension="audience", values=["adult"]),
        DimensionSelection(dimension="freshness", values=["modern"]),
    ]

    assert _channel_mappings_from_dimensions(dimensions) == []


@pytest.mark.anyio
async def test_categorize_media_surfaces_channel_dimension_as_mappings(monkeypatch):
    # Avoid any network access from the Wikipedia enrichment step.
    async def _no_wiki(self, *args, **kwargs):
        return ""

    monkeypatch.setattr(categorization.WikipediaLookup, "lookup_async", _no_wiki)

    media = MediaItem(id="1", title="FLCL", genres=["Animation", "Sci-Fi"])
    categories = {
        "channel": CategoryDefinition(
            description="Which IPTV channel this content is suitable for",
            values=["toontown", "nippon", "galaxy", "spotlight"],
        ),
    }

    llm = StubLLM(
        {
            "channel": (
                '{"dimension": {"dimension": "channel", '
                '"values": ["toontown", "nippon"], '
                '"notes": ["Anime with surreal animation"]}}'
            ),
        }
    )

    result = await categorize_media(media, categories, channels=None, llm=llm)

    # The channel dimension still appears in dimensions...
    channel_dim = next(d for d in result.dimensions if d.dimension == "channel")
    assert channel_dim.values == ["toontown", "nippon"]

    # ...and is now also surfaced as channel mappings.
    assert [m.channel_name for m in result.channel_mappings] == ["toontown", "nippon"]
