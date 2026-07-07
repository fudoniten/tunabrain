"""Keyframe captioning for long-form enrichment.

Captions extracted keyframes with a vision-capable LLM so their visual content
can join the transcript as grounding context. If the configured model has no
vision support, captioning degrades to a no-op with a warning.
"""

from tunabrain.keyframes.caption import caption_keyframe, caption_keyframes

__all__ = ["caption_keyframe", "caption_keyframes"]
