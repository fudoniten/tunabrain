"""Test quarterly strategy endpoint."""

import json
from tunabrain.api.models import (
    QuarterlyStrategyRequest,
    MediaCandidateSummary,
    ChannelContext,
    MediaItem,
)


def create_sample_request() -> QuarterlyStrategyRequest:
    """Create a sample quarterly strategy request for testing."""
    
    return QuarterlyStrategyRequest(
        quarter="Q4",
        year=2026,
        channels=[
            ChannelContext(
                name="Prime",
                description="Evening primetime 20:00-23:00, diverse audience",
            ),
            ChannelContext(
                name="Comedy",
                description="24/7 comedy block, sitcoms + standup",
            ),
            ChannelContext(
                name="Mystery",
                description="1-hour mysteries, noir, detective shows",
            ),
        ],
        media_candidates=MediaCandidateSummary(
            available_count=2847,
            summary="450 comedies, 380 dramas, 320 movies, 1697 specialty content",
            preview_sample=[
                MediaItem(
                    id="ep_001",
                    title="The Office - S7E1",
                    genres=["comedy", "workplace"],
                    duration_minutes=22,
                    audience_rating=8.5,
                ),
                MediaItem(
                    id="ep_002",
                    title="Breaking Bad - S5E15",
                    genres=["drama", "crime"],
                    duration_minutes=47,
                    audience_rating=9.2,
                ),
                MediaItem(
                    id="mov_001",
                    title="Knives Out",
                    genres=["mystery", "thriller"],
                    duration_minutes=130,
                    audience_rating=8.4,
                ),
            ],
            tag_availability={
                "family-friendly": 312,
                "prestige": 89,
                "holiday": 156,
                "halloween": 104,
                "evergreen": 1420,
                "drama": 445,
                "comedy": 312,
                "sitcom": 289,
            },
        ),
        strategic_guidance=(
            "Q4 is awards season. Emphasize prestige launches in October. "
            "Halloween content early-to-mid October. Thanksgiving family content Nov 24-28. "
            "Holiday specials Dec 1-24. New Year content Dec 25-31."
        ),
        cost_tier="balanced",
    )


def test_request_validation():
    """Test that sample request validates."""
    req = create_sample_request()
    
    assert req.quarter == "Q4"
    assert req.year == 2026
    assert len(req.channels) == 3
    assert req.media_candidates.available_count == 2847
    assert len(req.media_candidates.preview_sample) == 3
    assert req.cost_tier == "balanced"
    
    print("✅ Request validation passed")


def test_request_json_serialization():
    """Test that request can be serialized to JSON."""
    req = create_sample_request()
    json_str = req.model_dump_json()
    
    parsed = json.loads(json_str)
    assert parsed["quarter"] == "Q4"
    assert parsed["year"] == 2026
    assert len(parsed["channels"]) == 3
    assert len(parsed["media_candidates"]["preview_sample"]) == 3
    
    print("✅ JSON serialization passed")


def test_cost_calculation():
    """Test cost calculation module."""
    from tunabrain.scheduling.cost import calculate_cost, get_model_for_tier
    
    # Economy tier should be cheapest
    cost_eco = calculate_cost("llama-2-70b", 2000, 1500)
    cost_balanced = calculate_cost("gpt-4o-mini", 2000, 1500)
    cost_premium = calculate_cost("gpt-4o", 2000, 1500)
    
    assert cost_eco > 0
    assert cost_balanced > cost_eco
    assert cost_premium > cost_balanced
    
    # Model selection
    assert get_model_for_tier("economy") == "llama-2-70b"
    assert get_model_for_tier("balanced") == "gpt-4o-mini"
    assert get_model_for_tier("premium") == "gpt-4o"
    
    print(f"✅ Cost calculation passed")
    print(f"   Economy ($): ${cost_eco:.4f}")
    print(f"   Balanced ($): ${cost_balanced:.4f}")
    print(f"   Premium ($): ${cost_premium:.4f}")


def test_prompt_construction():
    """Test prompt construction."""
    from tunabrain.scheduling.quarterly_strategy import build_quarterly_strategy_prompt
    
    req = create_sample_request()
    messages = build_quarterly_strategy_prompt(req)
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    
    assert "JSON" in system_content
    assert "quarterly" in user_content.lower()
    assert "Q4" in user_content
    assert "2026" in user_content
    assert "Prime" in user_content
    assert "prestige" in user_content.lower()
    
    print(f"✅ Prompt construction passed")
    print(f"   System prompt: {len(system_content)} chars")
    print(f"   User prompt: {len(user_content)} chars")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUARTERLY STRATEGY ENDPOINT - UNIT TESTS")
    print("="*60 + "\n")
    
    test_request_validation()
    test_request_json_serialization()
    test_cost_calculation()
    test_prompt_construction()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60 + "\n")
