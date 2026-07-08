"""Title heuristics: placeholder detection and search-query cleaning.

Grout sends free-form media with weak titles — usually a filename, an on-disk
path, or a literal placeholder like ``"Unknown"``/``"<unnamed>"`` when nothing
better exists. Feeding those straight into a Wikipedia search yields
confident-but-wrong matches (a title of ``"<unnamed>"`` happily matches an anime
called "Unnamed Memory"). These helpers let the context resolver (a) refuse to
search on a placeholder title, and (b) strip filename cruft off a title before
it is used as a search query.
"""

from __future__ import annotations

import re

# Whole-title placeholders. Compared after reducing the title to bare letters
# (lowercased, non-alphabetic characters stripped), so only a *strict* match is
# filtered: "Unknown" -> "unknown" (placeholder) but "The Unknown Assassin" ->
# "theunknownassassin" (kept). Keep entries lowercase and letters-only so they
# can match the reduced form.
_PLACEHOLDER_TITLES = {
    "unnamed",
    "unnamedvideo",
    "unknown",
    "unknownvideo",
    "untitled",
    "untitledvideo",
    "notitle",
    "noname",
    "none",
    "null",
    "nil",
    "na",
    "nan",
    "video",
    "clip",
    "movie",
    "media",
    "file",
    "download",
    "tmp",
    "temp",
    "newvideo",
    "newproject",
}

_ALPHA_ONLY_RE = re.compile(r"[^a-z]+")

# Container extensions attached to filename-derived titles. Stripped before
# both placeholder detection and query cleaning so "untitled_video.mp4" is
# recognised as a placeholder and "clip.mkv" searches as "clip".
_EXTENSION_RE = re.compile(
    r"\.(mp4|mkv|avi|mov|webm|m4v|flv|wmv|mpg|mpeg|ts|m2ts|ogv|3gp)$", re.IGNORECASE
)


def _reduce_to_letters(title: str) -> str:
    """Lowercase ``title`` and strip everything that isn't an ASCII letter."""
    return _ALPHA_ONLY_RE.sub("", (title or "").lower())


def is_placeholder_title(title: str) -> bool:
    """True if ``title`` is empty or reduces to a known placeholder word.

    A trailing media extension is dropped, then the title is lowercased and
    stripped to letters before comparison, so only a *strict* whole-title match
    counts. Longer titles that merely contain a placeholder word ("The Unknown
    Assassin", "Untitled Goose Game") are not placeholders and are left alone.
    """
    stripped = _EXTENSION_RE.sub("", (title or "").strip())
    reduced = _reduce_to_letters(stripped)
    return reduced == "" or reduced in _PLACEHOLDER_TITLES


# Additional filename cruft: resolutions, source/codec/release tags. Stripped
# before building a search query so the query is the actual work name, not
# "clip 1080p x265 webrip".
_BRACKETS_RE = re.compile(r"[\[(][^\])]*[\])]")
_SEPARATORS_RE = re.compile(r"[._]+")
_WHITESPACE_RE = re.compile(r"\s+")
_CRUFT_TOKENS = {
    # resolutions / quality
    "480p", "576p", "720p", "1080p", "1440p", "2160p", "4k", "8k", "uhd", "hd", "sd",
    # video codecs
    "x264", "x265", "h264", "h265", "hevc", "avc", "xvid", "divx", "av1",
    # audio codecs
    "aac", "ac3", "eac3", "dts", "dd5", "mp3", "flac", "opus", "truehd",
    # sources
    "web", "webrip", "webdl", "bluray", "bdrip", "brrip", "hdtv", "dvdrip", "hdrip",
    "cam", "hdcam", "amzn", "nf", "dsnp", "hmax",
    # release flags
    "proper", "repack", "remux", "internal", "extended", "uncut", "unrated",
}


def clean_search_query(title: str) -> str:
    """Strip filename cruft from ``title`` to make a cleaner search query.

    Removes a trailing media extension, bracketed groups (release tags), and
    common resolution/codec/source tokens; normalizes ``.``/``_`` separators to
    spaces and collapses whitespace. If cleaning would empty the string, the
    original (trimmed) title is returned so the caller always has *something* to
    search on.
    """
    raw = (title or "").strip()
    if not raw:
        return raw
    work = _EXTENSION_RE.sub("", raw)
    work = _BRACKETS_RE.sub(" ", work)
    work = _SEPARATORS_RE.sub(" ", work)
    tokens = [t for t in _WHITESPACE_RE.split(work) if t]
    kept = [t for t in tokens if t.lower() not in _CRUFT_TOKENS]
    cleaned = " ".join(kept).strip()
    return cleaned or raw
