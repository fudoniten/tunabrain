import pytest
from langchain_core.messages import AIMessage

from tunabrain.api.models import Channel, MediaItem
from tunabrain.chains.channel_mapping import map_media_to_channels


class StubLLM:
    def __init__(self, responses: list[str]):
        self._responses = responses

    async def ainvoke(self, _messages):
        if not self._responses:
            raise RuntimeError("No stub responses remaining")
        return AIMessage(content=self._responses.pop(0))


class FailingLLM:
    async def ainvoke(self, _messages):  # pragma: no cover - fallback path
        raise RuntimeError("LLM unavailable")


@pytest.mark.anyio
async def test_channel_mapping_assigns_expected_channels():
    channels = [
        Channel(name="Toon", description="Animated and cartoon series"),
        Channel(name="Sitcom", description="Comedy and sitcom reruns"),
        Channel(name="SciFi", description="Science fiction and space adventures"),
        Channel(name="Classics", description="Retro and classic anthology TV"),
    ]

    simpsons = MediaItem(
        id="1",
        title="The Simpsons",
        genres=["Animation", "Comedy", "Sitcom"],
        description="An animated sitcom about the Simpsons family.",
    )

    futurama = MediaItem(
        id="2",
        title="Futurama",
        genres=["Animation", "Sci-Fi", "Comedy"],
        description="Animated sci-fi adventures in the 31st century.",
    )

    twilight_zone = MediaItem(
        id="3",
        title="The Twilight Zone",
        genres=["Sci-Fi", "Horror"],
        description="Classic anthology series exploring speculative fiction stories.",
    )

    llm = StubLLM(
        [
            '{"mappings": [{"channel_name": "Toon", "reasons": ["Animation focus"]}, {"channel_name": "Sitcom", "reasons": ["Comedy focus"]}]}',
            '{"mappings": [{"channel_name": "Toon", "reasons": ["Animated ensemble"]}, {"channel_name": "SciFi", "reasons": ["Futuristic setting"]}]}',
            '{"mappings": [{"channel_name": "SciFi", "reasons": ["Speculative anthology"]}, {"channel_name": "Classics", "reasons": ["Vintage television"]}]}',
        ]
    )

    simpsons_mapping = await map_media_to_channels(simpsons, channels, llm=llm)
    futurama_mapping = await map_media_to_channels(futurama, channels, llm=llm)
    twilight_mapping = await map_media_to_channels(twilight_zone, channels, llm=llm)

    assert {m.channel_name for m in simpsons_mapping} == {"Toon", "Sitcom"}
    assert {m.channel_name for m in futurama_mapping} == {"Toon", "SciFi"}
    assert {m.channel_name for m in twilight_mapping} == {"SciFi", "Classics"}

    for mapping in simpsons_mapping + futurama_mapping + twilight_mapping:
        assert mapping.reasons, f"Missing reasons for {mapping.channel_name}"


@pytest.mark.anyio
async def test_channel_mapping_limits_selection_but_returns_at_least_one():
    channels = [
        Channel(name="General", description="A little bit of everything"),
        Channel(name="Documentary", description="Nonfiction and documentary"),
        Channel(name="Sports", description="Live sports"),
        Channel(name="Reality", description="Unscripted competitions"),
    ]

    media = MediaItem(
        id="99",
        title="Unknown Show",
        genres=[],
        description="No metadata available",
    )

    mapping = await map_media_to_channels(media, channels, llm=FailingLLM())

    assert 1 <= len(mapping) <= 3
    assert mapping[0].channel_name == "General"
