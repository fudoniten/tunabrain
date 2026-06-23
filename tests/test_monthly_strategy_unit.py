"""Test monthly strategy endpoint with agent loop."""

import json
from tunabrain.api.models import (
    MonthlyStrategyRequest,
    MediaCandidateSummary,
    ChannelContext,
    MediaItem,
    QuarterlyStrategy,
    ChannelStrategyAdjustment,
)


def create_sample_quarterly_context() -> QuarterlyStrategy:
    """Create a sample quarterly strategy for context."""
    return QuarterlyStrategy(
        quarter="Q4 2026",
        overall_theme="Awards Season → Holiday",
        reasoning="October prestige launches, November awards momentum, December holidays",
        key_decisions=[
            "Prioritize prestige launches in October",
            "Horror content mid-October for Halloween",
            "Family content Thanksgiving week",
        ],
        channel_strategies=[
            ChannelStrategyAdjustment(
                channel="Prime",
                theme="Prestige Launch Drama → Holiday Movies",
                rationale="Prime audience seeks quality content",
                recommended_mix={"drama": "40%", "comedy": "30%"},
                special_focus=["awards-season", "prestige-launches"],
            )
        ],
        special_events=[],
        implied_monthly_themes={
            "2026-10": "Prestige Launch + Halloween",
            "2026-11": "Awards + Thanksgiving",
            "2026-12": "Holiday + Year-End",
        },
    )


def create_sample_monthly_request() -> MonthlyStrategyRequest:
    """Create a sample monthly strategy request for testing."""
    
    return MonthlyStrategyRequest(
        month="2026-10",
        channels=[
            ChannelContext(
                name="Prime",
                description="Evening 20:00-23:00, diverse audience",
            ),
            ChannelContext(
                name="Comedy",
                description="24/7 comedy, sitcoms + standup",
            ),
        ],
        quarterly_context=create_sample_quarterly_context(),
        media_candidates=MediaCandidateSummary(
            available_count=1200,
            summary="250 comedies, 180 dramas, 120 movies, 650 specialty",
            preview_sample=[
                MediaItem(
                    id="ep_001",
                    title="Prestige Drama - Pilot",
                    genres=["drama"],
                    duration_minutes=45,
                    audience_rating=8.9,
                ),
                MediaItem(
                    id="mov_001",
                    title="Horror Movie",
                    genres=["horror", "thriller"],
                    duration_minutes=120,
                    audience_rating=7.8,
                ),
            ],
            tag_availability={
                "prestige": 45,
                "halloween": 67,
                "comedy": 180,
                "drama": 200,
                "family-friendly": 120,
            },
        ),
        strategic_guidance="October emphasizes prestige launches and Halloween content",
        max_iterations=8,
        cost_tier="balanced",
    )


def test_request_validation():
    """Test that monthly strategy request validates."""
    req = create_sample_monthly_request()
    
    assert req.month == "2026-10"
    assert len(req.channels) == 2
    assert req.media_candidates.available_count == 1200
    assert req.quarterly_context is not None
    assert req.max_iterations == 8
    
    print("✅ Monthly request validation passed")


def test_request_json_serialization():
    """Test that request can be serialized to JSON."""
    req = create_sample_monthly_request()
    json_str = req.model_dump_json()
    
    parsed = json.loads(json_str)
    assert parsed["month"] == "2026-10"
    assert len(parsed["channels"]) == 2
    assert "quarterly_context" in parsed
    assert parsed["quarterly_context"]["quarter"] == "Q4 2026"
    
    print("✅ Monthly JSON serialization passed")


def test_prompt_construction():
    """Test monthly strategy prompt construction."""
    from tunabrain.scheduling.monthly_strategy import build_monthly_strategy_initial_prompt
    
    req = create_sample_monthly_request()
    messages = build_monthly_strategy_initial_prompt(req)
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    
    assert "JSON" in system_content
    assert "time_block" in system_content.lower()
    assert "2026-10" in user_content
    assert "Prime" in user_content
    assert "QUARTERLY CONTEXT" in user_content
    
    print(f"✅ Monthly prompt construction passed")
    print(f"   System prompt: {len(system_content)} chars")
    print(f"   User prompt: {len(user_content)} chars")


def test_strategy_validation():
    """Test monthly strategy validation and scoring."""
    from tunabrain.scheduling.monthly_strategy import validate_monthly_strategy
    from tunabrain.api.models import MonthlyTheme, TimeBlockRecommendation
    
    # Create a valid strategy
    strategy_data = {
        "month": "2026-10",
        "theme_name": "Prestige Launch + Halloween Thrills",
        "theme_description": (
            "October blends prestige drama launches with Halloween horror content. "
            "Prime time focuses on acclaimed new series, while late night embraces thrills and scares. "
            "Weekend daytime offers family-friendly alternatives before evening transitions to adult content."
        ),
        "key_focus_areas": [
            "Prestige drama launches",
            "Halloween-themed horror content",
            "Awards season momentum",
        ],
        "time_block_recommendations": [
            {
                "time_block": "morning",
                "time_range": "Mon-Fri 09:00-12:00",
                "recommended_content": "Family-friendly sitcoms and comedies",
                "content_mix": {"sitcom": "60%", "comedy": "40%"},
                "rationale": "Safe morning content for younger viewers",
            },
            {
                "time_block": "afternoon",
                "time_range": "14:00-18:00",
                "recommended_content": "Prestige dramas and thrillers",
                "content_mix": {"drama": "70%", "thriller": "30%"},
                "rationale": "Build momentum toward evening prestige block",
            },
            {
                "time_block": "prime",
                "time_range": "20:00-23:00",
                "recommended_content": "Prestige launches and award contenders",
                "content_mix": {"prestige_drama": "50%", "quality_drama": "50%"},
                "rationale": "Prime time for flagship prestige content",
            },
            {
                "time_block": "late_night",
                "time_range": "23:00-02:00",
                "recommended_content": "Horror and thriller content",
                "content_mix": {"horror": "60%", "thriller": "40%"},
                "rationale": "Halloween theme with darker, mature content",
            },
        ],
        "opening_tagline": "Where Prestige Meets Thrills — October's Greatest Hits",
        "special_notes": "Halloween week (Oct 24-31) emphasizes horror content in all time blocks",
    }
    
    strategy, score, feedback = validate_monthly_strategy(strategy_data)
    
    assert strategy.month == "2026-10"
    assert len(strategy.time_block_recommendations) == 4
    assert score > 0.7  # Should score well for complete strategy
    assert strategy.opening_tagline == "Where Prestige Meets Thrills — October's Greatest Hits"
    
    print(f"✅ Monthly strategy validation passed")
    print(f"   Score: {score:.2f}")
    print(f"   Feedback preview: {feedback[:100]}...")


def test_refinement_prompt():
    """Test refinement prompt construction."""
    from tunabrain.scheduling.monthly_strategy import build_monthly_strategy_refinement_prompt
    from tunabrain.api.models import MonthlyTheme, TimeBlockRecommendation
    
    req = create_sample_monthly_request()
    
    # Create a strategy to refine
    strategy = MonthlyTheme(
        month="2026-10",
        theme_name="October Prestige",
        theme_description="October focuses on prestige content.",
        key_focus_areas=["prestige"],
        time_block_recommendations=[
            TimeBlockRecommendation(
                time_block="prime",
                time_range="20:00-23:00",
                recommended_content="Prestige drama",
                content_mix={"drama": "100%"},
                rationale="Quality content",
            )
        ],
        opening_tagline="October Quality",
        special_notes="",
    )
    
    feedback = "Add more time blocks (need 4-5, currently have 1). Ensure percentages sum to 100%."
    
    messages = build_monthly_strategy_refinement_prompt(
        req, strategy, feedback, iteration_number=2
    )
    
    assert len(messages) == 2
    assert "Iteration 2" in messages[1]["content"]
    assert "time blocks" in messages[1]["content"].lower()
    
    print("✅ Monthly refinement prompt construction passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MONTHLY STRATEGY ENDPOINT - UNIT TESTS")
    print("="*60 + "\n")
    
    test_request_validation()
    test_request_json_serialization()
    test_prompt_construction()
    test_strategy_validation()
    test_refinement_prompt()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60 + "\n")
