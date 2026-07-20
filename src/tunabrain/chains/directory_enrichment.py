"""Directory / tag-group profiling: derive one shared profile per group.

Grout's bulk library is organised one directory per channel/creator, with
well-structured filenames. Rather than pay for a per-file ``/enrich/short-form``
call across ~200k items, this chain derives a single shared profile
(dimensions + tags) for a whole group from the group's name and a small sample
of its filenames. Every child item then inherits the profile downstream.

It is a deliberately small, single-LLM-call chain (mirrors :mod:`describe`):
one prompt, structured output, graceful degradation. The only content signal is
the filenames, and the prompt is explicit that the model must not invent facts
the names do not support. On any LLM/parse failure the chain returns an empty
profile with a warning rather than raising — the caller (Grout's worker) treats
that as a soft failure and retries with backoff.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field

from tunabrain.api.models import (
    CategoryDefinition,
    CategoryValue,
    CostEstimate,
    EnrichProfileRequest,
    EnrichProfileResponse,
    GroupContext,
)
from tunabrain.chains.validation import partition_values
from tunabrain.config import get_settings, is_debug_enabled
from tunabrain.llm import get_chat_model
from tunabrain.scheduling.cost import calculate_cost

logger = logging.getLogger(__name__)

# The controlled dimension keys the model may populate when the caller does
# not supply `categories` (back-compat with callers on older Grout builds).
# Kept in sync with Grout's config (:dimension-descriptions) and Tunarr
# Scheduler's catalog. When `categories` is supplied, the caller's dimension
# names and candidate values are the source of truth instead — see
# `_SYSTEM_PROMPT_WITH_CATEGORIES` and `_validate_and_fill_categories`.
_DIMENSION_KEYS = ("channel", "audience", "freshness", "season", "time-slot")

_SYSTEM_PROMPT = (
    "You are a media librarian classifying a GROUP of related media items that "
    "share one organizing concept (a channel, creator, series, or directory). "
    "You are given the group's name and a sample of its filenames — nothing "
    "else. Derive a profile that describes the group as a whole.\n\n"
    "Rules:\n"
    "- Base your analysis ONLY on the concept name and the filenames. Do NOT "
    "hallucinate facts about the actual video content.\n"
    "- If operator-provided context is given below, treat it as ground truth "
    "about the group's actual content — it overrides any assumption you'd "
    "otherwise make from the concept name or filenames alone.\n"
    "- dimensions: a JSON object mapping dimension names to a list of one or "
    "more values. Allowed dimension keys: channel, audience, freshness, season, "
    "time-slot.\n"
    "- For 'channel', derive a single best value from the concept name (e.g. "
    "'Adam Neely Music' -> a music channel; 'Tom Scott' -> a general/variety "
    "channel). Do not get creative.\n"
    "- For 'audience', infer from the filename patterns (educational -> adult, "
    "cartoon -> family, etc.). When unclear, omit the dimension entirely.\n"
    "- For 'freshness', 'season', 'time-slot': set them only on a clear signal. "
    "When in doubt, omit the dimension.\n"
    "- Prefer fewer, more confident dimensions over many speculative ones.\n"
    "- tags: 3-7 short lowercase tags describing the group's typical content "
    "(e.g. 'music', 'music-theory', 'educational', 'jazz'). Lowercase, "
    "hyphenated, no spaces or special characters."
)

# Used instead of `_SYSTEM_PROMPT` when the caller supplies `categories`: the
# model is handed a closed vocabulary per dimension (fetched by Grout from
# Tunarr Scheduler) rather than proposing values freely, and every listed
# dimension is mandatory rather than omit-if-unsure.
_SYSTEM_PROMPT_WITH_CATEGORIES = (
    "You are a media librarian classifying a GROUP of related media items that "
    "share one organizing concept (a channel, creator, series, or directory). "
    "You are given the group's name, a sample of its filenames, and a "
    "controlled vocabulary of dimensions with their candidate values — "
    "nothing else. Derive a profile that describes the group as a whole.\n\n"
    "Rules:\n"
    "- Base your analysis ONLY on the concept name and the filenames. Do NOT "
    "hallucinate facts about the actual video content.\n"
    "- If operator-provided context is given below, treat it as ground truth "
    "about the group's actual content — it overrides any assumption you'd "
    "otherwise make from the concept name or filenames alone.\n"
    "- dimensions: a JSON object mapping each dimension name below to a list "
    "of one or more values, chosen ONLY from that dimension's candidate "
    "values. Never invent a value outside the candidates.\n"
    "- Every dimension listed below MUST get at least one value. If no "
    "candidate is a confident fit, pick the single closest candidate rather "
    "than omitting the dimension.\n"
    "- For 'channel' (when listed), derive a single best value from the "
    "concept name (e.g. 'Adam Neely Music' -> a music channel; 'Tom Scott' -> "
    "a general/variety channel).\n"
    "- Prefer fewer, more confident values per dimension over many "
    "speculative ones.\n"
    "- tags: 3-7 short lowercase tags describing the group's typical content "
    "(e.g. 'music', 'music-theory', 'educational', 'jazz'). Lowercase, "
    "hyphenated, no spaces or special characters."
)


class ProfileResult(BaseModel):
    """Structured LLM output for a group profile."""

    dimensions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Dimension name -> list of values. Only confident dimensions.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="3-7 short lowercase-hyphenated free-form tags.",
    )


def _estimate_cost(llm_calls: int) -> CostEstimate:
    """Rough CostEstimate for ``llm_calls`` profiling call(s).

    Mirrors the estimation style in :mod:`chains.describe`: a fixed per-call
    token budget priced against the configured model. One small prompt in, one
    small JSON object out.
    """
    model = get_settings().llm_model
    prompt_tokens = 700 * llm_calls
    completion_tokens = 150 * llm_calls
    cost_usd = calculate_cost(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return CostEstimate(
        estimated_cost_usd=cost_usd,
        llm_calls_used=llm_calls,
        estimated_tokens=f"~{prompt_tokens + completion_tokens:,}",
        model=model,
    )


def _sanitize_tag(tag: str) -> str:
    """Normalize a tag to lowercase, hyphenated, alphanumeric-only.

    The model is asked for clean tags but is not trusted to always comply, so
    we enforce the shape the caller (and its downstream tag namespace) expects:
    lowercase, non-alphanumeric runs collapsed to a single hyphen, edges
    stripped. Returns an empty string for a tag that is all punctuation.
    """
    return re.sub(r"[^a-z0-9]+", "-", tag.strip().lower()).strip("-")


def _clean_dimensions(
    dimensions: dict[str, list[str]], allowed_keys: Iterable[str]
) -> dict[str, list[str]]:
    """Keep only known dimension keys with at least one non-blank value."""
    allowed = set(allowed_keys)
    cleaned: dict[str, list[str]] = {}
    for key, values in dimensions.items():
        if key not in allowed:
            continue
        vals = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        if vals:
            cleaned[key] = vals
    return cleaned


def _normalize_category_values(
    values: list[str] | list[CategoryValue],
) -> list[tuple[str, str | None]]:
    """Normalize category values to a list of (value, description) tuples.

    Mirrors :func:`chains.categorization._normalize_category_values` — kept as
    a separate copy rather than a shared import since each chain's usage is a
    small, single call site.
    """
    result: list[tuple[str, str | None]] = []
    for v in values:
        if isinstance(v, CategoryValue):
            result.append((v.value, v.description))
        elif isinstance(v, dict):
            result.append((v.get("value", str(v)), v.get("description")))
        else:
            result.append((str(v), None))
    return result


def _format_operator_context(context: GroupContext | None) -> str:
    """Render the caller-supplied `GroupContext` as a prompt section, or `""`
    when there's nothing to say.

    Unlike `MediaContext` (per-item chains), links here are echoed as plain
    text, never fetched or summarized — this chain has no web-grounding step
    at all, only filenames and whatever the operator wrote by hand.
    """
    if context is None:
        return ""
    lines: list[str] = []
    if context.text and context.text.strip():
        lines.append(context.text.strip())
    if context.links:
        lines.append("Reference links (context only, not fetched): " + ", ".join(context.links))
    if not lines:
        return ""
    return "Operator-provided context:\n" + "\n".join(lines) + "\n\n"


def _format_categories_block(categories: dict[str, CategoryDefinition]) -> str:
    """Render the caller-supplied dimensions as a candidate-value prompt block.

    Mirrors :func:`chains.categorization._categorize_single`'s per-value
    formatting so the model sees the same style of controlled vocabulary,
    just for every dimension in one call instead of one call per dimension.
    """
    lines: list[str] = []
    for dim_name, definition in categories.items():
        lines.append(f"- {dim_name}: {definition.description}")
        for value, description in _normalize_category_values(definition.values):
            if description:
                lines.append(f"    - {value}: {description}")
            else:
                lines.append(f"    - {value}")
    return "\n".join(lines)


def _validate_and_fill_categories(
    dimensions: dict[str, list[str]],
    categories: dict[str, CategoryDefinition],
) -> dict[str, list[str]]:
    """Enforce the controlled vocabulary for every requested dimension.

    Filters out hallucinated values (not present in the dimension's candidate
    list, via :func:`chains.validation.partition_values`) and guarantees each
    requested dimension ends up with at least one value — falling back to its
    first candidate when the model omitted the dimension or every value it
    proposed was invalid. A dimension with no configured candidates is passed
    through unfiltered since there is nothing to validate against.
    """
    result: dict[str, list[str]] = {}
    for dim_name, definition in categories.items():
        allowed = [value for value, _ in _normalize_category_values(definition.values)]
        if not allowed:
            existing = dimensions.get(dim_name)
            if existing:
                result[dim_name] = existing
            continue

        valid, invalid = partition_values(dimensions.get(dim_name, []), allowed)
        if invalid:
            logger.warning(
                "Dropping hallucinated value(s) for dimension '%s': %s (valid options: %s)",
                dim_name,
                invalid,
                allowed,
            )
        if not valid:
            logger.warning(
                "Dimension '%s' had no valid value; falling back to '%s'",
                dim_name,
                allowed[0],
            )
            valid = [allowed[0]]
        result[dim_name] = valid
    return result


def _clean_tags(tags: list[str]) -> list[str]:
    """Sanitize, drop blanks, and de-duplicate (preserving first-seen order)."""
    seen: dict[str, None] = {}
    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean = _sanitize_tag(tag)
        if clean:
            seen.setdefault(clean, None)
    return list(seen.keys())


async def enrich_profile(
    request: EnrichProfileRequest,
    *,
    llm: RunnableSerializable | None = None,
) -> EnrichProfileResponse:
    """Derive a shared dimensions+tags profile for a media group.

    A single LLM call grounded on the concept name + sampled filenames (plus
    `request.categories`, when supplied, as a controlled vocabulary). Output
    is sanitized (known dimension keys only; lowercase-hyphenated tags) and
    degrades gracefully: an LLM or parse failure yields an empty profile plus a
    warning, never a raised exception.

    When `request.categories` is supplied, every listed dimension is
    guaranteed at least one value (via `_validate_and_fill_categories`) and
    any hallucinated value outside its candidate list is dropped — this is
    the caller-provided controlled vocabulary path (see
    `grout.tunarr_scheduler/fetch-value-descriptions!`). Without it, the
    model proposes values freely across the fixed `_DIMENSION_KEYS`, matching
    pre-v1.1 behavior.

    `request.context` (a `GroupContext`), when supplied, is rendered into the
    prompt verbatim as operator-authored ground truth (see
    `_format_operator_context`) — e.g. correcting a directory of retro
    video-game ads that was misclassified onto a vintage-film channel. Unlike
    per-item `MediaContext`, links are never fetched/summarized here.
    """
    debug = is_debug_enabled(request.debug)
    logger.info(
        "Profiling group '%s' from %s sample filename(s)",
        request.concept_name,
        len(request.sample_filenames),
    )

    categories = request.categories or {}
    warnings: list[str] = []
    llm_instance = llm or get_chat_model()

    parser = PydanticOutputParser(pydantic_object=ProfileResult)

    if categories:
        system_prompt = _SYSTEM_PROMPT_WITH_CATEGORIES
        categories_section = (
            "Dimensions and their candidate values:\n"
            f"{_format_categories_block(categories)}\n\n"
        )
        allowed_keys: Iterable[str] = categories.keys()
    else:
        system_prompt = _SYSTEM_PROMPT
        categories_section = ""
        allowed_keys = _DIMENSION_KEYS

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Group concept name: {concept_name}\n\n"
                "Sample filenames from this group:\n{filenames_block}\n\n"
                "{operator_context_section}"
                "{categories_section}"
                "Return only the JSON dictated by the format instructions."
                "{format_instructions}",
            ),
        ]
    )
    filenames_block = "\n".join(f"  - {name}" for name in request.sample_filenames)
    inputs = {
        "concept_name": request.concept_name,
        "filenames_block": filenames_block,
        "operator_context_section": _format_operator_context(request.context),
        "categories_section": categories_section,
        "format_instructions": f"\n\n{parser.get_format_instructions()}",
    }
    if debug:
        logger.debug("LLM request (enrich-profile): %s", inputs)

    dimensions: dict[str, list[str]] = {}
    tags: list[str] = []
    try:
        messages = prompt.format_messages(**inputs)
        response = await llm_instance.ainvoke(messages)
        if debug:
            logger.debug("LLM raw response (enrich-profile): %s", response)
        result = await parser.ainvoke(response)
        dimensions = _clean_dimensions(result.dimensions, allowed_keys)
        if categories:
            dimensions = _validate_and_fill_categories(dimensions, categories)
        tags = _clean_tags(result.tags)
        if not dimensions and not tags:
            warnings.append("model returned no usable dimensions or tags")
    except OutputParserException as exc:
        logger.error(
            "Failed to parse profile response for '%s'. llm_output=%s",
            request.concept_name,
            getattr(exc, "llm_output", "<missing>"),
        )
        warnings.append(f"profile parse failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive catch for external service
        logger.warning("Profile LLM call failed for '%s': %s", request.concept_name, exc)
        warnings.append(f"profile failed: {exc}")

    response = EnrichProfileResponse(
        concept_name=request.concept_name,
        dimensions=dimensions,
        tags=tags,
        grounding_source="filename-pattern",
        cost_estimate=_estimate_cost(1),
        warnings=warnings,
    )
    logger.info(
        "Profile complete for '%s': %s dimensions, %s tags, %s warnings",
        request.concept_name,
        len(dimensions),
        len(tags),
        len(warnings),
    )
    return response
