from __future__ import annotations

import re

from tunabrain.api.models import Channel, ChannelMapping, MediaItem


# A lightweight, deterministic heuristic for matching media to channels based on
# overlapping themes. These keywords capture common programming buckets and a
# handful of synonyms so we can map a title even when channel descriptions vary.
KEYWORD_MAP: dict[str, set[str]] = {
    "action": {"action", "adventure", "hero", "spy"},
    "animation": {"animation", "animated", "cartoon", "toon"},
    "classic": {"classic", "retro", "vintage", "black-and-white", "anthology"},
    "comedy": {"comedy", "sitcom", "humor", "funny"},
    "crime": {"crime", "detective", "police", "noir"},
    "documentary": {"documentary", "docuseries", "nonfiction", "true story"},
    "drama": {"drama", "dramatic"},
    "fantasy": {"fantasy", "magic", "myth"},
    "family": {"family", "kids", "children", "all ages"},
    "horror": {"horror", "scary"},
    "mystery": {"mystery", "thriller", "suspense"},
    "reality": {"reality", "competition", "unscripted"},
    "romance": {"romance", "love story"},
    "sci-fi": {"sci-fi", "science fiction", "scifi", "space", "futuristic"},
    "sports": {"sports", "athletics"},
}


def _normalize_text(value: str) -> str:
    return value.lower()


def _extract_keywords(text: str) -> set[str]:
    normalized = _normalize_text(text)
    matches: set[str] = set()
    for canonical, synonyms in KEYWORD_MAP.items():
        if any(syn in normalized for syn in synonyms):
            matches.add(canonical)
    return matches


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_text(text)))


def _media_keywords(media: MediaItem) -> set[str]:
    keywords: set[str] = set()
    for genre in media.genres:
        keywords.update(_extract_keywords(genre))
    keywords.update(_extract_keywords(media.title))
    if media.description:
        keywords.update(_extract_keywords(media.description))
    return keywords


def _channel_keywords(channel: Channel) -> set[str]:
    keywords = _extract_keywords(channel.name)
    if channel.description:
        keywords.update(_extract_keywords(channel.description))
    return keywords


async def map_media_to_channels(
    media: MediaItem, channels: list[Channel], *, debug: bool = False
) -> list[ChannelMapping]:
    """Map a media item to the best-fit channels based on media synopsis & channel description.
    """

    if not channels:
        return []

    media_kw = _media_keywords(media)
    media_tokens = _tokenize(" ".join([media.title, media.description or ""] + media.genres))

    scored_channels: list[tuple[int, Channel, int, int, set[str], set[str]]] = []
    for idx, channel in enumerate(channels):
        channel_kw = _channel_keywords(channel)
        keyword_overlap = media_kw & channel_kw
        keyword_score = len(keyword_overlap)

        channel_tokens = _tokenize(" ".join([channel.name, channel.description or ""]))
        token_overlap = media_tokens & channel_tokens
        token_score = len(token_overlap)

        scored_channels.append(
            (idx, channel, keyword_score, token_score, keyword_overlap, channel_tokens)
        )

    # Sort by keyword overlap first, then by token overlap, then original order for stability.
    scored_channels.sort(key=lambda item: (-item[2], -item[3], item[0]))

    preferred = [
        item
        for item in scored_channels
        if item[2] > 0 or item[3] > 0  # keyword or token overlap present
    ]

    candidates = preferred if preferred else scored_channels

    # Ensure we always select at least one channel even when there is no overlap.
    if not preferred:
        selected = candidates[:1]
    else:
        selected = []
        for item in candidates:
            if len(selected) < 2:
                selected.append(item)
            elif len(selected) < 3 and (item[2] > 1 or item[3] > 1):
                selected.append(item)
            if len(selected) == 3:
                break

    mappings: list[ChannelMapping] = []
    for _, channel, keyword_score, token_score, overlap, channel_tokens in selected:
        reasons: list[str] = []
        if overlap:
            for kw in sorted(overlap):
                reasons.append(
                    f"Matches {kw} programming based on media metadata and channel focus."
                )
        elif keyword_score == 0 and token_score > 0:
            common_terms = ", ".join(sorted(term for term in channel_tokens if term in media_tokens))
            reasons.append(
                f"Selected due to shared terms ({common_terms}) between the media details and channel description."
            )
        else:
            reasons.append(
                "Selected as the closest available channel when no thematic overlap was found."
            )

        mappings.append(ChannelMapping(channel_name=channel.name, reasons=reasons))

    return mappings

