import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tunabrain.api.models import (
    CategoryDefinition,
    CategoryValue,
    Channel,
    MediaContext,
    MediaItem,
)
from tunabrain.chains.categorization import _categorize_single, _categorize_single_safe
from tunabrain.chains.channel_mapping import map_media_to_channels
from tunabrain.chains.context import ResolvedContext
from tunabrain.chains.tagging import generate_tags
from tunabrain.chains.validation import (
    format_invalid_feedback,
    format_kebab_feedback,
    is_kebab_case,
    partition_kebab_case,
    partition_values,
)


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


async def _stub_resolve(media, context=None, *, llm=None, debug=False):
    """Stub for ``resolve_media_context`` used by ``generate_tags`` tests.

    The real resolver hits Wikipedia + the LLM, which we don't want in unit
    tests.  A short fixed summary keeps the human-prompt template populated and
    matches the production happy-path shape.
    """
    return ResolvedContext(
        "Test Wikipedia summary.",
        MediaContext(summary="Test Wikipedia summary.", source="wikipedia"),
    )


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


# --- kebab-case helpers --------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "documentary",
        "action-and-adventure",
        "sci-fi",
        "two-words",
        "1980s",
        "pre-2000",
    ],
)
def test_is_kebab_case_accepts_valid(value):
    assert is_kebab_case(value)


@pytest.mark.parametrize(
    "value",
    [
        "Documentary",               # capital
        "Action & Adventure",        # spaces, ampersand
        "action_and_adventure",      # underscores
        "ActionAndAdventure",        # camel/pascal
        "-leading-hyphen",           # leading hyphen
        "trailing-hyphen-",          # trailing hyphen
        "double--hyphen",            # consecutive hyphens
        "",                          # empty
        " action-and-adventure",     # leading space
        "action-and-adventure ",     # trailing space
        "café",                      # non-ASCII
    ],
)
def test_is_kebab_case_rejects_invalid(value):
    assert not is_kebab_case(value)


def test_partition_kebab_case_splits_and_dedupes():
    valid, invalid = partition_kebab_case(
        [
            "action-and-adventure",
            "Documentary",
            "action-and-adventure",  # duplicate of the first
            "sci-fi",
            "Two Words",
        ]
    )
    assert valid == ["action-and-adventure", "sci-fi"]
    assert invalid == ["Documentary", "Two Words"]


def test_format_kebab_feedback_mentions_invalid_values():
    msg = format_kebab_feedback(["Documentary", "Action & Adventure"])
    assert "Documentary" in msg
    assert "Action & Adventure" in msg
    # Sanity: the message also names the format requirement.
    assert "kebab-case" in msg


# --- generate_tags kebab-case validation ---------------------------------------


@pytest.mark.anyio
async def test_generate_tags_reprompts_then_accepts_kebab_case(monkeypatch):
    media = _media()
    llm = RecordingLLM(
        [
            # First response: raw Jellyfin genre strings (the bug we're fixing).
            '{"tags": ["Action & Adventure", "Documentary", "sci-fi"]}',
            # Second response (after feedback): kebab-case.
            '{"tags": ["action-and-adventure", "documentary", "sci-fi"]}',
        ]
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.get_chat_model", lambda task=None: llm
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.resolve_media_context", _stub_resolve
    )

    result, _ctx = await generate_tags(media, existing_tags=None)

    assert result == ["action-and-adventure", "documentary", "sci-fi"]
    # The LLM was re-prompted with feedback about the non-kebab-case tags.
    assert len(llm.calls) == 2
    feedback = llm.calls[1][-1]
    assert isinstance(feedback, HumanMessage)
    assert "Action & Adventure" in feedback.content
    assert "Documentary" in feedback.content


@pytest.mark.anyio
async def test_generate_tags_drops_non_kebab_case_when_uncorrected(monkeypatch):
    media = _media()
    # The LLM keeps returning a mix of valid and invalid tags.
    llm = RecordingLLM(
        [
            '{"tags": ["sci-fi", "Action & Adventure", "Documentary"]}',
            '{"tags": ["sci-fi", "Action & Adventure", "Documentary"]}',
            '{"tags": ["sci-fi", "Action & Adventure", "Documentary"]}',
        ]
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.get_chat_model", lambda task=None: llm
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.resolve_media_context", _stub_resolve
    )

    result, _ctx = await generate_tags(media, existing_tags=None)

    # Invalid tags are dropped; the one valid tag is retained.
    assert result == ["sci-fi"]
    # One initial call + two retries = three LLM calls.
    assert len(llm.calls) == 3


@pytest.mark.anyio
async def test_generate_tags_keeps_kebab_case_on_first_try(monkeypatch):
    """A well-behaved LLM that returns kebab-case tags gets through in one call."""
    media = _media()
    llm = RecordingLLM(
        ['{"tags": ["action-and-adventure", "documentary", "sci-fi"]}']
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.get_chat_model", lambda task=None: llm
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.resolve_media_context", _stub_resolve
    )

    result, _ctx = await generate_tags(media, existing_tags=None)

    assert result == ["action-and-adventure", "documentary", "sci-fi"]
    # No retries needed.
    assert len(llm.calls) == 1


@pytest.mark.anyio
async def test_generate_tags_prompt_includes_kebab_case_instruction(monkeypatch):
    """The system prompt sent to the LLM must include the kebab-case rule."""
    media = _media()
    llm = RecordingLLM(
        ['{"tags": ["action-and-adventure", "documentary", "sci-fi"]}']
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.get_chat_model", lambda task=None: llm
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.resolve_media_context", _stub_resolve
    )

    await generate_tags(media, existing_tags=None)

    messages = llm.calls[0]
    system_message = messages[0]
    assert "kebab-case" in system_message.content
    assert "lowercase" in system_message.content


@pytest.mark.anyio
async def test_generate_tags_drops_non_kebab_case_via_existing_tags(monkeypatch):
    """Non-kebab-case existing tags echoed back by the batch LLM are dropped.

    The catalog may still contain legacy non-kebab-case tags from before this
    fix landed.  When ``evaluate_tag_batches`` runs the LLM over those
    candidates the model can legitimately select one of them, so the safety net
    in the batch step must drop the result before it reaches the final prompt
    as a "vetted existing tag".
    """
    media = _media()
    llm = RecordingLLM(
        [
            # Batch call: echoes all three existing tags back as selected.
            '{"tags": ["Action & Adventure", "Documentary", "sci-fi"]}',
            # Final call: produces kebab-case tags.
            '{"tags": ["action-and-adventure", "documentary", "sci-fi"]}',
        ]
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.get_chat_model", lambda task=None: llm
    )
    monkeypatch.setattr(
        "tunabrain.chains.tagging.resolve_media_context", _stub_resolve
    )

    result, _ctx = await generate_tags(
        media, existing_tags=["Action & Adventure", "Documentary", "sci-fi"]
    )

    # The non-kebab-case existing tags were filtered by the batch safety net,
    # and the final generation produced clean kebab-case output.
    assert result == ["action-and-adventure", "documentary", "sci-fi"]
