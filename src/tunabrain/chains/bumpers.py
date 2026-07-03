from __future__ import annotations

import base64
import logging

import httpx
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tunabrain.api.models import Bumper, Channel
from tunabrain.config import get_settings, is_debug_enabled
from tunabrain.llm import LLMTask, get_chat_model


logger = logging.getLogger(__name__)

_OPENROUTER_IMAGE_URL = "https://openrouter.ai/api/v1/images"
_DEFAULT_IMAGE_MODEL = "google/gemini-2.5-flash-image"


class ImagePromptResult(BaseModel):
    """LLM output for a bumper image prompt."""

    prompt: str = Field(
        ...,
        description="A vivid, detailed image-generation prompt for the bumper"
    )
    title: str = Field(
        ...,
        description="Short title for the bumper"
    )


async def _generate_image_prompt(
    channel: Channel,
    schedule_overview: str,
    duration_seconds: int,
    focus_window: str | None,
    theme: str | None,
    debug: bool = False,
) -> ImagePromptResult:
    """Use an LLM to craft a channel-appropriate image prompt."""

    debug_enabled = is_debug_enabled(debug)
    llm = get_chat_model(task=LLMTask.BUMPERS)

    system_msg = (
        "You are a creative director for a TV channel. "
        "Generate a vivid, detailed image prompt for a short channel bumper "
        "(5-15 seconds). The prompt should describe a visually striking scene "
        "that captures the channel's identity. Keep it to 1-2 sentences. "
        "Also provide a short title."
    )

    user_msg = (
        f"Channel: {channel.name}\n"
        f"Description: {channel.description or 'No description'}\n"
        f"Schedule overview: {schedule_overview}\n"
        f"Duration: {duration_seconds}s\n"
        f"Focus: {focus_window or 'general'}\n"
        f"Theme: {theme or 'general'}\n\n"
        "Generate an image prompt and title for the bumper."
    )

    parser = PydanticOutputParser(pydantic_object=ImagePromptResult)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{user_msg}\n{format_instructions}"),
    ])

    messages = prompt_template.format_messages(
        user_msg=user_msg,
        format_instructions=parser.get_format_instructions(),
    )

    response = await llm.ainvoke(messages)
    result = parser.parse(str(response.content))

    if debug_enabled:
        logger.debug(
            "Generated bumper prompt for '%s': title='%s' prompt='%s...'",
            channel.name,
            result.title,
            result.prompt[:80],
        )

    return result


async def _generate_image(
    prompt: str,
    model: str | None = None,
) -> str:
    """Call the OpenRouter Image API and return a base64-encoded PNG.

    Args:
        prompt: The image-generation prompt.
        model:  Model slug (default: google/gemini-2.5-flash-image).

    Returns:
        Base64-encoded image data.

    Raises:
        RuntimeError: If the API call fails or no API key is configured.
    """

    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is not configured. "
            "Set OPENROUTER_API_KEY to enable image generation."
        )

    request_payload = {
        "model": model or _DEFAULT_IMAGE_MODEL,
        "prompt": prompt,
        "n": 1,
    }

    logger.info(
        "Requesting image from OpenRouter (model=%s)",
        request_payload["model"],
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            _OPENROUTER_IMAGE_URL,
            json=request_payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter image generation failed: HTTP {response.status_code} - {response.text}"
        )

    data = response.json()
    image_data = data.get("data", [{}])[0].get("b64_json")
    if not image_data:
        raise RuntimeError(
            f"OpenRouter returned no image data: {data}"
        )

    logger.info("Generated image (%s chars base64)", len(image_data))
    return image_data


async def generate_bumpers(
    *,
    channel: Channel,
    schedule_overview: str,
    duration_seconds: int,
    focus_window: str | None,
    theme: str | None = None,
    debug: bool = False,
) -> list[Bumper]:
    """Generate a single bumper with an AI-generated image.

    The pipeline:
    1. Use an LLM to craft a channel-appropriate image prompt.
    2. Call the OpenRouter Image API to generate the visual.
    3. Return a Bumper record containing the image (base64).

    Args:
        channel:          Channel identity (name + description).
        schedule_overview: Schedule context for thematic coherence.
        duration_seconds: Target bumper length.
        focus_window:     Optional temporal focus string.
        theme:            Optional creative theme override.
        debug:            Enable verbose logging.

    Returns:
        A list containing one Bumper with the generated image.

    Raises:
        RuntimeError: If the OpenRouter API key is missing or the image call fails.
    """

    logger.info(
        "Bumper generation for channel='%s' (duration=%ss, theme=%s)",
        channel.name,
        duration_seconds,
        theme or "auto",
    )

    # Step 1 — LLM prompt generation
    prompt_result = await _generate_image_prompt(
        channel=channel,
        schedule_overview=schedule_overview,
        duration_seconds=duration_seconds,
        focus_window=focus_window,
        theme=theme,
        debug=debug,
    )

    # Step 2 — Image generation via OpenRouter
    image_b64 = await _generate_image(prompt_result.prompt)

    # Step 3 — Assemble bumper
    bumper = Bumper(
        title=prompt_result.title,
        script=prompt_result.prompt,
        duration_seconds=duration_seconds,
        image_base64=image_b64,
    )

    logger.info(
        "Bumper ready: title='%s' (%ss, image=%s chars)",
        bumper.title,
        bumper.duration_seconds,
        len(bumper.image_base64 or ""),
    )

    return [bumper]
