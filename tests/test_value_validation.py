import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tunabrain.api.models import (
    CategoryDefinition,
    CategoryValue,
    Channel,
    MediaItem,
)
from tunabrain.chains.categorization import _categorize_single, _categorize_single_safe
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.chains.validation import format_invalid_feedback, partition_values


class RecordingLLM:
    """Stub LLM that records the messages it receives and replays responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("No stub responses remaining")
        return AIMessage(content=self._responses.pop(0))


def _media() -> MediaItem:
    return MediaItem(
        id="1",
        title="Some Show",
        genres=["Drama"],
        description="A dramatic show.",
    )


# --- partition_values / format_invalid_feedback ---------------------------------


def test_partition_values_splits_and_dedupes():
    valid, invalid = partition_values(
        ["spectrum", "thriller", "spectrum", "prime"],
        ["spectrum", "prime", "galaxy"],
    )
    assert valid == ["spectrum", "prime"]
    assert invalid == ["thriller"]


def test_format_invalid_feedback_mentions_both_sets():
    msg = format_invalid_feedback(["thriller"], ["spectrum", "prime"])
    assert "thriller" in msg
    assert "spectrum" in msg
    assert "prime" in msg


# --- categorization dimension validation ----------------------------------------


@pytest.mark.anyio
async def test_categorize_single_reprompts_then_accepts_valid():
    definition = CategoryDefinition(
        description="The channel for the media",
        values=[
            CategoryValue(value="spectrum", description="Sci-fi channel"),
            CategoryValue(value="prime", description="Premium drama"),
        ],
    )
    llm = RecordingLLM(
        [
            '{"dimension": {"dimension": "channel", "values": ["spectum"], "notes": []}}',
            '{"dimension": {"dimension": "channel", "values": ["spectrum"], "notes": []}}',
        ]
    )

    dim = await _categorize_single(
        llm=llm,
        media=_media(),
        category_name="channel",
        category_definition=definition,
        wikipedia_summary="n/a",
        debug=False,
    )

    assert dim.values == ["spectrum"]
    # The LLM was re-prompted with feedback about the invalid choice.
    assert len(llm.calls) == 2
    feedback = llm.calls[1][-1]
    assert isinstance(feedback, HumanMessage)
    assert "spectum" in feedback.content


@pytest.mark.anyio
async def test_categorize_single_filters_invalid_when_uncorrected():
    definition = CategoryDefinition(
        description="The channel for the media",
        values=["spectrum", "prime"],
    )
    # The LLM keeps returning a mix of valid and invalid values.
    llm = RecordingLLM(
        ['{"dimension": {"dimension": "channel", "values": ["prime", "thriller"], "notes": []}}'] * 3
    )

    dim = await _categorize_single(
        llm=llm,
        media=_media(),
        category_name="channel",
        category_definition=definition,
        wikipedia_summary="n/a",
        debug=False,
    )

    # Invalid value is dropped; valid value retained.
    assert dim.values == ["prime"]


@pytest.mark.anyio
async def test_categorize_single_safe_falls_back_when_all_invalid():
    definition = CategoryDefinition(
        description="The channel for the media",
        values=["spectrum", "prime"],
    )
    llm = RecordingLLM(
        ['{"dimension": {"dimension": "channel", "values": ["thriller"], "notes": []}}'] * 3
    )

    dim = await _categorize_single_safe(
        llm=llm,
        media=_media(),
        category_name="channel",
        category_definition=definition,
        wikipedia_summary="n/a",
        debug=False,
    )

    # Everything was invalid, so the fallback (first allowed value) is applied.
    assert dim.values == ["spectrum"]
    assert all(value in {"spectrum", "prime"} for value in dim.values)


# --- channel mapping validation -------------------------------------------------


@pytest.mark.anyio
async def test_channel_mapping_reprompts_then_accepts_valid():
    channels = [
        Channel(name="spectrum", description="Sci-fi channel"),
        Channel(name="prime", description="Premium drama"),
    ]
    llm = RecordingLLM(
        [
            '{"mappings": [{"channel_name": "spectum", "reasons": ["typo"]}]}',
            '{"mappings": [{"channel_name": "spectrum", "reasons": ["Sci-fi fit"]}]}',
        ]
    )

    mappings = await map_media_to_channels(_media(), channels, llm=llm)

    assert [m.channel_name for m in mappings] == ["spectrum"]
    assert len(llm.calls) == 2


@pytest.mark.anyio
async def test_channel_mapping_drops_invalid_channels():
    channels = [
        Channel(name="spectrum", description="Sci-fi channel"),
        Channel(name="prime", description="Premium drama"),
    ]
    llm = RecordingLLM(
        ['{"mappings": [{"channel_name": "prime", "reasons": ["fit"]}, {"channel_name": "action", "reasons": ["hallucinated"]}]}']
        * 3
    )

    mappings = await map_media_to_channels(_media(), channels, llm=llm)

    names = {m.channel_name for m in mappings}
    assert "action" not in names
    assert names <= {"spectrum", "prime"}
    assert names == {"prime"}
