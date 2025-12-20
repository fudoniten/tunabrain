from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tunabrain.api.models import TagDecision, TagSample
from tunabrain.config import is_debug_enabled
from tunabrain.llm import get_chat_model

logger = logging.getLogger(__name__)


class TagBatchReview(BaseModel):
    """Structured review output for a set of tags."""

    decisions: list[TagDecision] = Field(
        default_factory=list, description="Per-tag governance recommendations"
    )


async def triage_tags(
    tags: Iterable[TagSample], *, target_limit: int | None = None, debug: bool = False
) -> list[TagDecision]:
    """Vet tags for scheduling usefulness and consolidation.

    The LLM is asked to keep, drop, merge, or rename tags with short rationales
    oriented around programming needs. To keep context manageable, tags are
    reviewed in batches.
    """

    tag_list = list(tags)
    if not tag_list:
        return []

    debug_enabled = is_debug_enabled(debug)
    llm = get_chat_model()

    parser = PydanticOutputParser(pydantic_object=TagBatchReview)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are cleaning a media-tag taxonomy for scheduling runs and marathons. "
                "For each tag, choose one action: keep (good scheduling hook), drop (too "
                "vague or noisy), merge (map to an existing broader tag), or rename "
                "(reword to a clearer, audience-facing synonym). Prefer concise, "
                "schedulable language and limit the total unique tags if a target is "
                "provided.",
            ),
            (
                "human",
                "Target tag count (if provided): {target_limit}.\n"
                "Review the following tags with usage and examples. Return structured JSON "
                "only, per the format instructions.\n\n{tag_table}\n\n"
                "Rules:\n"
                "- Keep tags that describe genres, tone, audience, events, seasons, or other "
                "clear scheduling hooks.\n"
                "- Drop ultra-specific, ideological, or unclear tags.\n"
                "- Merge narrow variants into their broader parent tag.\n"
                "- Rename when a clearer synonym improves scheduling clarity.\n"
                "- Provide a short rationale for each decision.\n"
                "{format_instructions}",
            ),
        ]
    )

    async def evaluate_batch(batch: list[TagSample]) -> list[TagDecision]:
        examples = []
        for sample in batch:
            example_text = ", ".join(sample.example_titles) if sample.example_titles else "None"
            examples.append(
                f"- {sample.tag} (usage={sample.usage_count}; examples: {example_text})"
            )

        inputs = {
            "target_limit": target_limit or "not provided",
            "tag_table": "\n".join(examples),
            "format_instructions": f"\n\n{parser.get_format_instructions()}",
        }

        if debug_enabled:
            logger.debug("LLM request (tag governance batch): %s", inputs)

        messages = prompt.format_messages(**inputs)
        response = await llm.ainvoke(messages)
        if debug_enabled:
            logger.debug("LLM raw response (tag governance batch): %s", response)

        try:
            result: TagBatchReview = await parser.ainvoke(response)
        except OutputParserException as exc:
            logger.error(
                "Failed to parse tag governance batch. llm_output=%s",
                getattr(exc, "llm_output", "<missing>"),
            )
            raise

        if debug_enabled:
            logger.debug("LLM parsed response (tag governance batch): %s", result.model_dump())

        return result.decisions

    decisions: list[TagDecision] = []
    batch_size = 75
    for idx in range(0, len(tag_list), batch_size):
        batch = tag_list[idx : idx + batch_size]
        for decision in await evaluate_batch(batch):
            # Preserve first recommendation per tag to avoid churn across batches.
            if not any(existing.tag == decision.tag for existing in decisions):
                decisions.append(decision)

    logger.info("Generated governance recommendations for %s tags", len(decisions))
    return decisions
