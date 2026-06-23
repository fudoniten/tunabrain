"""Quarterly strategy generation - orchestrate LLM to develop strategic programming themes."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tunabrain.api.models import QuarterlyStrategyRequest
from tunabrain.llm import get_chat_model

if TYPE_CHECKING:
    from tunabrain.api.models import QuarterlyStrategy

logger = logging.getLogger(__name__)


def build_quarterly_strategy_prompt(request: QuarterlyStrategyRequest) -> list[dict]:
    """Construct LLM prompt for quarterly strategy generation.
    
    Args:
        request: QuarterlyStrategyRequest with quarter, channels, media summary, etc.
    
    Returns:
        List of message dicts for LLM (system + user)
    """
    
    # Channels list
    channels_str = "\n".join([
        f"  - {c.name}: {c.description}"
        for c in request.channels
    ])
    
    # Tag availability
    tags_str = "\n".join([
        f"  - {tag}: ~{count} episodes"
        for tag, count in sorted(request.media_candidates.tag_availability.items(), 
                                  key=lambda x: x[1], reverse=True)[:15]  # Top 15 tags
    ])
    
    # Media preview
    media_preview = ""
    if request.media_candidates.preview_sample:
        media_preview = "Example media:\n"
        for item in request.media_candidates.preview_sample[:3]:
            media_preview += f"  - {item.title} ({item.genres})\n"
    
    system_prompt = """You are a TV programming strategist. Your task is to generate a strategic quarterly programming overview for a television network.

You must respond in valid JSON matching this exact schema:
{
  "quarter": "string (e.g., 'Q4 2026')",
  "overall_theme": "string (2-3 sentences describing the main quarterly theme)",
  "reasoning": "string (2-3 sentences explaining WHY this theme - industry trends, audience behavior, calendar factors)",
  "key_decisions": ["string", ...],  // 5-10 key strategic decisions made
  "channel_strategies": [
    {
      "channel": "string (channel name)",
      "theme": "string (1-2 sentences about this channel's Q focus)",
      "rationale": "string (why this theme fits this channel)",
      "recommended_mix": {"genre_or_category": "percentage", ...},  // e.g., {"drama": "40%", "comedy": "30%"}
      "special_focus": ["string", ...]  // areas of emphasis
    },
    ...  // one per channel
  ],
  "special_events": [
    {
      "date": "string (date or date range, e.g., 'Dec 25', 'Nov 24-28')",
      "event_name": "string (event name)",
      "recommendation": "string (how to schedule around this event)"
    },
    ...
  ],
  "implied_monthly_themes": {"YYYY-MM": "string theme", ...}  // Suggested monthly themes derived from quarterly strategy
}

IMPORTANT:
- Return ONLY valid JSON, no markdown or explanations
- All arrays and objects must be non-empty or use defaults
- channel_strategies must include ALL provided channels
- Reasoning should reference calendar/cultural factors specific to the quarter
- Monthly themes must cover all 3 months of the quarter"""
    
    user_prompt = f"""Generate a strategic programming overview for Q{request.quarter[1]} {request.year}.

CHANNELS ({len(request.channels)}):
{channels_str}

AVAILABLE MEDIA ({request.media_candidates.available_count} items):
Overall: {request.media_candidates.summary}

Most Available Tags:
{tags_str}

{media_preview}

{f"STRATEGIC GUIDANCE: {request.strategic_guidance}" if request.strategic_guidance else ""}

Generate a cohesive Q{request.quarter[1]} strategy that:
1. Identifies the quarter's key cultural, seasonal, or industry moments
2. Recommends per-channel theme + content mix for each channel listed above
3. Notes special events (holidays, industry moments, etc.) affecting programming
4. Suggests monthly sub-themes to guide detailed monthly planning

Make decisions that demonstrate understanding of TV industry, audience behavior, and the available media breakdown."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def generate_quarterly_strategy(request: QuarterlyStrategyRequest) -> QuarterlyStrategy:
    """Generate quarterly strategy using LLM.
    
    Args:
        request: QuarterlyStrategyRequest
    
    Returns:
        QuarterlyStrategy object (validated)
    
    Raises:
        ValueError: If LLM response is invalid JSON or doesn't match schema
        RuntimeError: If LLM request fails
    """
    
    logger.info(f"Generating quarterly strategy for Q{request.quarter} {request.year}")
    logger.debug(f"Channels: {len(request.channels)}, Media available: {request.media_candidates.available_count}")
    
    # Build prompt
    messages = build_quarterly_strategy_prompt(request)
    
    logger.debug(f"Prompt constructed: {len(messages)} messages")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"System prompt length: {len(messages[0]['content'])} chars")
        logger.debug(f"User prompt length: {len(messages[1]['content'])} chars")
    
    # Get LLM
    llm = get_chat_model()
    logger.debug(f"Using LLM: {llm.__class__.__name__}")
    
    # Invoke LLM with JSON response format constraint
    try:
        response = llm.invoke(
            messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4096
        )
        logger.debug(f"LLM response received")
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise RuntimeError(f"Failed to invoke LLM: {e}") from e
    
    # Parse response
    response_text = response.content
    logger.debug(f"Response text length: {len(response_text)} chars")
    
    try:
        strategy_json = json.loads(response_text)
        logger.debug(f"JSON parsed successfully")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Response text: {response_text[:500]}")
        raise ValueError(f"LLM did not return valid JSON: {e}") from e
    
    # Validate against schema
    try:
        from tunabrain.api.models import QuarterlyStrategy
        strategy = QuarterlyStrategy(**strategy_json)
        logger.debug(f"Strategy validated successfully")
        logger.debug(f"  - {len(strategy.channel_strategies)} channel strategies")
        logger.debug(f"  - {len(strategy.special_events)} special events")
        logger.debug(f"  - {len(strategy.implied_monthly_themes)} monthly themes")
    except Exception as e:
        logger.error(f"Strategy validation failed: {e}")
        logger.error(f"JSON was: {json.dumps(strategy_json, indent=2)[:1000]}")
        raise ValueError(f"LLM response doesn't match schema: {e}") from e
    
    logger.info(f"Quarterly strategy generated successfully")
    return strategy
