"""Unit tests for title heuristics: placeholder detection + query cleaning."""

from __future__ import annotations

import pytest

from tunabrain.tools.titles import clean_search_query, is_placeholder_title


@pytest.mark.parametrize(
    "title",
    [
        "Unknown",
        "unknown",
        "<unnamed>",
        "  Untitled  ",
        "UNTITLED",
        "no-name",
        "none",
        "null",
        "n/a",
        "Untitled Video",
        "untitled_video.mp4",
        "video",
        "12345",  # reduces to empty (no letters)
        "____",  # reduces to empty
        "",
    ],
)
def test_is_placeholder_title_flags_placeholders(title):
    assert is_placeholder_title(title) is True


@pytest.mark.parametrize(
    "title",
    [
        "The Unknown Assassin",
        "Untitled Goose Game",
        "Unknown Pleasures",
        "Keeping Motivated",
        "keeping_motivated.2025.mp4",
        "A Video Essay on Rome",
        "Clipper Ship Documentary",  # contains 'clip' but not a strict match
    ],
)
def test_is_placeholder_title_keeps_real_titles(title):
    assert is_placeholder_title(title) is False


def test_clean_search_query_strips_extension_and_separators():
    assert clean_search_query("keeping_motivated.2025.mp4") == "keeping motivated 2025"


def test_clean_search_query_drops_release_cruft():
    got = clean_search_query("The.Big.Lecture.1080p.WEBRip.x265.mkv")
    assert got == "The Big Lecture"


def test_clean_search_query_strips_bracketed_groups():
    got = clean_search_query("Some Talk [YIFY] (720p)")
    assert got == "Some Talk"


def test_clean_search_query_preserves_a_clean_title():
    assert clean_search_query("Keeping Motivated") == "Keeping Motivated"


def test_clean_search_query_falls_back_when_cleaning_empties():
    # A title that is nothing but cruft should not reduce to "" — keep original.
    assert clean_search_query("1080p.x265") == "1080p.x265"
