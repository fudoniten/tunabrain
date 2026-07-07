"""Caption keyframes with a vision-capable LLM.

For v1 we reuse the configured ``TUNABRAIN_LLM_MODEL`` (via ``get_chat_model``)
with a vision prompt. Images are inlined as base64 ``data:`` URLs in a LangChain
multimodal ``HumanMessage``. If the model doesn't support vision, the call
raises and the caller degrades gracefully (keyframe analysis is skipped).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from tunabrain.llm import get_chat_model

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe what is visible in this single video frame in one concise sentence. "
    "Focus on concrete, schedulable signals: setting, people, on-screen text, and "
    "the apparent genre or format of the content. Do not speculate beyond the frame."
)


async def caption_keyframe(image_path: Path, *, llm=None, mime_type: str = "image/jpeg") -> str:
    """Return a one-sentence caption for a single keyframe image."""
    llm = llm or get_chat_model()
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": _CAPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            },
        ]
    )
    response = await llm.ainvoke([message])
    content = response.content
    if isinstance(content, list):
        # Some providers return content as a list of blocks; join their text.
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content).strip()


async def caption_keyframes(image_paths: list[Path], *, llm=None) -> list[str]:
    """Caption a list of keyframes in temporal order.

    A single failing frame is skipped with a warning rather than aborting the
    whole batch; a caption backend that fails on the first frame (e.g. no vision
    support) will simply produce an empty list.
    """
    llm = llm or get_chat_model()
    captions: list[str] = []
    for path in image_paths:
        try:
            caption = await caption_keyframe(path, llm=llm)
        except Exception as exc:
            logger.warning("Keyframe captioning failed for %s: %s", path, exc)
            continue
        if caption:
            captions.append(caption)
    return captions
