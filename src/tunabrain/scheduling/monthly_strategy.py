"""Monthly strategy generation - agent-based iterative refinement."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tunabrain.api.models import (
    MonthlyStrategyRequest,
    MonthlyTheme,
    MonthlyStrategyAgentIteration,
)
from tunabrain.llm import get_chat_model

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Convergence thresholds
CONVERGENCE_THRESHOLD = 0.85
MIN_IMPROVEMENT_FOR_ITERATION = 0.05


def build_monthly_strategy_initial_prompt(request: MonthlyStrategyRequest) -> list[dict]:
    """Build the initial LLM prompt for monthly strategy generation.
    
    Args:
        request: MonthlyStrategyRequest
    
    Returns:
        List of message dicts (system + user)
    """
    
    channels_str = "\n".join([
        f"  - {c.name}: {c.description}"
        for c in request.channels
    ])
    
    # Quarterly context if provided
    quarterly_context_str = ""
    if request.quarterly_context:
        quarterly_context_str = f"""
QUARTERLY CONTEXT (Q{request.quarterly_context.quarter[-1]} {request.quarterly_context.quarter.split()[0]}):
- Overall Theme: {request.quarterly_context.overall_theme}
- Key Decisions: {', '.join(request.quarterly_context.key_decisions[:3])}
- Special Events This Quarter: {len(request.quarterly_context.special_events)} events planned
"""
    
    system_prompt = """You are a TV programming strategist specializing in monthly content planning. Your task is to generate a detailed monthly programming strategy.

You must respond in valid JSON matching this exact schema:
{
  "month": "string (YYYY-MM format)",
  "theme_name": "string (1-2 sentences, catchy theme name)",
  "theme_description": "string (3-5 sentences detailed description)",
  "key_focus_areas": ["string", ...],  // 3-5 strategic focus areas for the month
  "time_block_recommendations": [
    {
      "time_block": "string (early_morning|morning|afternoon|prime|late_night)",
      "time_range": "string (e.g., '06:00-09:00' or 'Mon-Fri 09:00-12:00')",
      "recommended_content": "string (type of content for this block)",
      "content_mix": {"genre": "percentage", ...},  // e.g., {"comedy": "60%", "sitcom": "40%"}
      "rationale": "string (why this mix works for this time block)"
    },
    ...  // one per time block (typically 4-5)
  ],
  "opening_tagline": "string (short promotional tagline)",
  "special_notes": "string (any special considerations)"
}

IMPORTANT:
- Return ONLY valid JSON, no markdown or explanations
- theme_name should be memorable and evoke the month's focus
- Ensure 4-5 time block recommendations covering full broadcast day
- content_mix percentages should sum to 100%
- rationale should be specific and justify the programming choices"""
    
    user_prompt = f"""Generate a detailed monthly programming strategy for {request.month}.

CHANNELS ({len(request.channels)}):
{channels_str}

AVAILABLE MEDIA ({request.media_candidates.available_count} items):
{request.media_candidates.summary}

Top Available Tags:
{', '.join([f"{tag}: ~{count} items" for tag, count in sorted(request.media_candidates.tag_availability.items(), key=lambda x: x[1], reverse=True)[:10]])}

{quarterly_context_str}

{f"STRATEGIC GUIDANCE: {request.strategic_guidance}" if request.strategic_guidance else ""}

Generate a cohesive monthly strategy that:
1. Defines a compelling month-long theme
2. Recommends content for each time block (early morning, morning, afternoon, prime, late night)
3. Provides rationale for content mix in each block
4. Offers a promotional tagline for the month
5. Notes any special considerations

Make strategic decisions based on available media, the quarterly context (if provided), and TV programming best practices."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def build_monthly_strategy_refinement_prompt(
    request: MonthlyStrategyRequest,
    current_strategy: MonthlyTheme,
    feedback: str,
    iteration_number: int
) -> list[dict]:
    """Build a refinement prompt for subsequent iterations.
    
    Args:
        request: Original request
        current_strategy: Current strategy from previous iteration
        feedback: Feedback on what to improve
        iteration_number: Current iteration number
    
    Returns:
        List of message dicts (system + user)
    """
    
    system_prompt = """You are refining a monthly TV programming strategy based on feedback. Generate an improved version.

Respond in the same JSON schema as before:
{
  "month": "string",
  "theme_name": "string",
  "theme_description": "string",
  "key_focus_areas": ["string", ...],
  "time_block_recommendations": [...],
  "opening_tagline": "string",
  "special_notes": "string"
}

IMPORTANT:
- Address all feedback points in your refinement
- Maintain strategic coherence across all time blocks
- Return ONLY valid JSON"""
    
    user_prompt = f"""Iteration {iteration_number}: Refine the monthly strategy based on this feedback:

{feedback}

CURRENT STRATEGY:
Theme: {current_strategy.theme_name}
Description: {current_strategy.theme_description}

Time Blocks:
{json.dumps([tb.model_dump() for tb in current_strategy.time_block_recommendations], indent=2)}

TASK:
Improve the strategy to address the feedback. Focus on:
1. Strengthening theme coherence
2. Refining time block content recommendations
3. Ensuring percentages sum to 100%
4. Making the strategy more distinctive and defensible"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def validate_monthly_strategy(strategy_json: dict) -> tuple[MonthlyTheme, float, str]:
    """Validate and score a monthly strategy.
    
    Returns:
        (validated_strategy, score, feedback_for_improvement)
    
    Raises:
        ValueError: If JSON structure is invalid
    """
    
    try:
        strategy = MonthlyTheme(**strategy_json)
    except Exception as e:
        raise ValueError(f"Strategy validation failed: {e}") from e
    
    score = 1.0
    issues = []
    
    # Check theme coherence
    if not strategy.theme_name or len(strategy.theme_name) < 3:
        score -= 0.1
        issues.append("Theme name too short or missing")
    
    if not strategy.theme_description or len(strategy.theme_description) < 20:
        score -= 0.1
        issues.append("Theme description too brief")
    
    # Check time block coverage
    if len(strategy.time_block_recommendations) < 4:
        score -= 0.15
        issues.append(f"Only {len(strategy.time_block_recommendations)} time blocks (need 4-5)")
    
    # Validate content mix percentages
    for tb in strategy.time_block_recommendations:
        if tb.content_mix:
            total = sum(
                float(v.rstrip("%")) 
                for v in tb.content_mix.values() 
                if v and isinstance(v, str)
            )
            if total < 95 or total > 105:  # Allow small rounding variance
                score -= 0.05
                issues.append(f"Time block '{tb.time_block}' percentages sum to {total}% (should be ~100%)")
    
    # Check key focus areas
    if len(strategy.key_focus_areas) < 2:
        score -= 0.05
        issues.append(f"Only {len(strategy.key_focus_areas)} focus areas (need 3-5)")
    
    # Check tagline
    if not strategy.opening_tagline or len(strategy.opening_tagline) < 5:
        score -= 0.05
        issues.append("Opening tagline missing or too short")
    
    # Clamp score
    score = max(0.0, min(1.0, score))
    
    # Generate feedback
    feedback = ""
    if issues:
        feedback = "Issues to address in next iteration:\n- " + "\n- ".join(issues)
    else:
        feedback = "Strategy is well-formed. Consider enhancing distinctive positioning or refining content mix specificity."
    
    return strategy, score, feedback


async def generate_monthly_strategy_agent_loop(
    request: MonthlyStrategyRequest,
) -> tuple[MonthlyTheme, list[MonthlyStrategyAgentIteration], int, float]:
    """Run agent loop to iteratively refine monthly strategy.
    
    Args:
        request: MonthlyStrategyRequest
    
    Returns:
        (final_strategy, iterations_history, total_iterations, final_score)
    
    Raises:
        RuntimeError: If LLM fails after max retries
        ValueError: If strategy response is invalid JSON
    """
    
    logger.info(f"Starting monthly strategy agent loop for {request.month}")
    logger.debug(f"Max iterations: {request.max_iterations}, convergence threshold: {CONVERGENCE_THRESHOLD}")
    
    llm = get_chat_model()
    iterations_history = []
    current_strategy = None
    current_score = 0.0
    
    for iteration_num in range(1, request.max_iterations + 1):
        logger.debug(f"Iteration {iteration_num}/{request.max_iterations}")
        
        # Build prompt
        if iteration_num == 1:
            messages = build_monthly_strategy_initial_prompt(request)
        else:
            messages = build_monthly_strategy_refinement_prompt(
                request, current_strategy, feedback, iteration_num
            )
        
        # Invokve LLM
        try:
            response = llm.invoke(
                messages,
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temp for refinement
                max_tokens=3000
            )
        except Exception as e:
            logger.error(f"LLM invocation failed at iteration {iteration_num}: {e}")
            raise RuntimeError(f"LLM failed at iteration {iteration_num}: {e}") from e
        
        # Parse response
        response_text = response.content
        try:
            strategy_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error at iteration {iteration_num}: {e}")
            raise ValueError(f"LLM returned invalid JSON at iteration {iteration_num}: {e}") from e
        
        # Validate + score
        try:
            current_strategy, current_score, feedback = validate_monthly_strategy(strategy_json)
        except ValueError as e:
            logger.error(f"Validation failed at iteration {iteration_num}: {e}")
            raise
        
        logger.debug(f"Iteration {iteration_num}: score={current_score:.2f}, feedback_len={len(feedback)}")
        
        # Record iteration
        is_converged = current_score >= CONVERGENCE_THRESHOLD
        iteration_record = MonthlyStrategyAgentIteration(
            iteration_number=iteration_num,
            strategy=current_strategy,
            validation_score=current_score,
            feedback=feedback,
            is_converged=is_converged
        )
        iterations_history.append(iteration_record)
        
        # Check convergence
        if is_converged:
            logger.info(f"Converged at iteration {iteration_num} with score {current_score:.2f}")
            break
        
        # Check improvement threshold
        if iteration_num > 1 and len(iterations_history) > 1:
            prev_score = iterations_history[-2].validation_score
            improvement = current_score - prev_score
            if improvement < MIN_IMPROVEMENT_FOR_ITERATION and iteration_num > 3:
                logger.info(f"Minimal improvement ({improvement:.2f}) at iteration {iteration_num}, stopping")
                break
    
    logger.info(
        f"Agent loop complete: {len(iterations_history)} iterations, "
        f"final score {current_score:.2f}, converged={current_score >= CONVERGENCE_THRESHOLD}"
    )
    
    return current_strategy, iterations_history, len(iterations_history), current_score
