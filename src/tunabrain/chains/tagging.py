from __future__ import annotations

from tunabrain.api.models import MediaItem


async def generate_tags(media: MediaItem) -> list[str]:
    """Generate scheduling-friendly tags for the provided media item.

    This function should orchestrate LangChain components to build a set of
    concise tags that help place the media into thematic schedules.
    """

    raise NotImplementedError("Tagging chain is not implemented yet")

