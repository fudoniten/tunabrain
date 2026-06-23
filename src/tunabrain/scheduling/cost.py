"""Cost calculation utilities for LLM operations."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


# OpenRouter pricing (as of June 2026) - per 1M tokens
# Using OpenAI-compatible endpoints
PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input": 150, "output": 600},  # $0.15 / $0.60 per 1M tokens
    "gpt-4o": {"input": 5000, "output": 15000},  # $5.00 / $15.00 per 1M tokens
    "claude-3-5-sonnet": {"input": 3000, "output": 15000},  # $3.00 / $15.00 per 1M tokens
    "claude-3-opus": {"input": 15000, "output": 75000},  # $15.00 / $75.00 per 1M tokens
    "llama-2-70b": {"input": 700, "output": 900},  # $0.70 / $0.90 per 1M tokens
}

# Model selection by tier
MODELS_BY_TIER = {
    "economy": "llama-2-70b",
    "balanced": "gpt-4o-mini",
    "premium": "gpt-4o",
}


def get_model_for_tier(tier: Literal["economy", "balanced", "premium"]) -> str:
    """Get recommended model for cost tier.
    
    Args:
        tier: Cost tier
    
    Returns:
        Model name
    """
    return MODELS_BY_TIER.get(tier, "gpt-4o-mini")


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """Calculate cost for LLM call.
    
    Args:
        model: Model name
        prompt_tokens: Input token count
        completion_tokens: Output token count
    
    Returns:
        Cost in USD
    """
    
    if model not in PRICING_PER_1M_TOKENS:
        logger.warning(
            f"Unknown model '{model}', using default pricing. Available: {list(PRICING_PER_1M_TOKENS.keys())}"
        )
        # Conservative default
        pricing = {"input": 1000, "output": 2000}
    else:
        pricing = PRICING_PER_1M_TOKENS[model]
    
    # Calculate (pricing is per 1M tokens, so divide by 1,000,000)
    input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (completion_tokens * pricing["output"]) / 1_000_000
    
    total = input_cost + output_cost
    
    logger.debug(
        f"Cost calculation: {prompt_tokens} input + {completion_tokens} output tokens "
        f"({model}) = ${total:.6f}"
    )
    
    return total


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens needed for text.
    
    Using rough heuristic: ~1 token per 4 characters for English.
    LLMs are more sophisticated but this is good enough for pre-flight estimation.
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    """
    # Very rough: ~1 token per 4 chars for English
    return max(1, len(text) // 4)


def format_cost(cents: float) -> str:
    """Format cost in USD with appropriate precision.
    
    Args:
        cents: Cost in USD
    
    Returns:
        Formatted cost string
    """
    if cents < 0.01:
        return f"<$0.01"
    elif cents < 0.10:
        return f"${cents:.4f}"
    else:
        return f"${cents:.2f}"
