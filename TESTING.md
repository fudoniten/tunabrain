# Testing Guide for TunaBrain Scheduling Agent

This guide explains how to test the autonomous scheduling agent, submit sample queries, configure different LLM backends, and verify the results.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuring LLM Backends](#configuring-llm-backends)
3. [Manual Testing](#manual-testing)
4. [Running Integration Tests](#running-integration-tests)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **Python Environment**: Make sure you're in the Nix development shell or have Python 3.11+ with dependencies installed:
   ```bash
   nix develop  # If using Nix
   # OR
   pip install -e .  # If using pip
   ```

2. **API Key**: Set your OpenAI API key (required for GPT models):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

### Run the Manual Test Script

The easiest way to test the agent is with the included manual test script:

```bash
python test_scheduling_manual.py
```

This will run three test scenarios:
1. **Basic Schedule**: Morning sitcom block
2. **Gap Filling**: Schedule around pre-filled content
3. **Multi-Day**: 3-day evening programming

---

## Configuring LLM Backends

TunaBrain supports multiple LLM providers via environment variables.

### OpenAI (Default)

```bash
export OPENAI_API_KEY=sk-...
export TUNABRAIN_LLM_PROVIDER=openai
export TUNABRAIN_LLM_MODEL=gpt-4o-mini  # Or gpt-4o for premium
```

**Cost tier mapping**:
- `economy`: Uses local models (Ollama) when available, falls back to `gpt-4o-mini`
- `balanced`: Uses `gpt-4o-mini` (default, recommended)
- `premium`: Uses `gpt-4o` for critical decisions

### Ollama (Local Models)

For free local inference:

```bash
# Start Ollama server (in separate terminal)
ollama serve

# Pull a model
ollama pull deepseek-r1:8b

# Configure TunaBrain
export TUNABRAIN_LLM_PROVIDER=ollama
export TUNABRAIN_LLM_MODEL=deepseek-r1:8b
```

**Note**: The `economy` cost tier will automatically try to use Ollama if available.

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export TUNABRAIN_LLM_PROVIDER=anthropic
export TUNABRAIN_LLM_MODEL=claude-3-5-sonnet-20241022
```

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export TUNABRAIN_LLM_PROVIDER=azure_openai
export TUNABRAIN_LLM_MODEL=gpt-4o-mini
```

### Debug Logging

Enable detailed logging to see LLM prompts and responses:

```bash
export TUNABRAIN_DEBUG=1
```

---

## Manual Testing

### Using the Test Script

The `test_scheduling_manual.py` script demonstrates three common scenarios.

**Example: Test just the basic schedule**

Edit the script and comment out the tests you don't want:

```python
async def main():
    # Test 1: Basic scheduling
    await test_basic_schedule()
    
    # # Test 2: Gap filling (commented out)
    # await test_gap_filling()
    
    # # Test 3: Multi-day (commented out)
    # await test_multi_day_schedule()
```

### Custom Test Scenarios

Create your own test by copying this template:

```python
import asyncio
from datetime import datetime
from tunabrain.agents.scheduling_agent import build_schedule_with_agent
from tunabrain.api.models import Channel, MediaItem, ScheduleRequest

async def my_custom_test():
    # Define your channel
    channel = Channel(
        name="My Test Channel",
        description="Testing custom scenarios"
    )
    
    # Define your media library
    media = [
        MediaItem(
            id="show-1",
            title="My Show",
            genres=["comedy"],
            duration_minutes=30,
        ),
        # ... add more shows
    ]
    
    # Create scheduling request
    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 1, 18, 0),
        scheduling_window_days=1,
        user_instructions="Your custom instructions here",
        preferred_slots=["18:00", "19:00", "20:00"],
        cost_tier="balanced",
        max_iterations=20,
    )
    
    # Run the agent
    response = await build_schedule_with_agent(request)
    
    # Inspect results
    print(f"Overview: {response.overview}")
    print(f"Iterations: {response.reasoning_summary.total_iterations}")
    print(f"Slots scheduled: {len(response.daily_slots)}")
    
    for slot in response.daily_slots:
        print(f"  {slot.start_time} - {slot.end_time}: {slot.media_id}")

# Run it
asyncio.run(my_custom_test())
```

### Testing via API

You can also test by running the FastAPI server and sending HTTP requests.

**1. Start the server:**

```bash
python -m tunabrain
```

**2. Send a request:**

```bash
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "channel": {
      "name": "Test Channel",
      "description": "Testing via API"
    },
    "media": [
      {
        "id": "show1",
        "title": "Test Show",
        "genres": ["comedy"],
        "duration_minutes": 30
      }
    ],
    "start_date": "2026-02-01T18:00:00",
    "scheduling_window_days": 1,
    "user_instructions": "Fill evening slots with comedy",
    "preferred_slots": ["18:00", "19:00", "20:00"],
    "max_iterations": 15
  }'
```

---

## Running Integration Tests

### Run All Tests

```bash
pytest tests/test_integration.py -v
```

### Run Specific Tests

```bash
# Run only multi-day tests
pytest tests/test_integration.py -k "multi_day" -v

# Run only gap-filling tests
pytest tests/test_integration.py -k "gap_filling" -v
```

### Test with Different Cost Tiers

```bash
# Test with economy tier (local models)
TUNABRAIN_LLM_MODEL=deepseek-r1:8b pytest tests/test_integration.py

# Test with premium tier
TUNABRAIN_LLM_MODEL=gpt-4o pytest tests/test_integration.py
```

### Skip Slow Tests

Mark tests that take a long time:

```bash
pytest tests/test_integration.py -m "not slow"
```

---

## Understanding Results

### ScheduleResponse Structure

When you run the agent, you get back a `ScheduleResponse` with these fields:

```python
{
  "overview": "High-level summary of what was scheduled",
  
  "reasoning_summary": {
    "total_iterations": 12,              # How many agent loops
    "completion_status": "complete",     # complete | partial | failed
    "quality_score": 0.85,               # 0.0 - 1.0
    "unfilled_slots_count": 2,           # Gaps remaining
    "key_decisions": [                   # Important choices made
      "Scheduled Friends in morning slot",
      "Filled gap at 10 AM with Seinfeld",
      ...
    ],
    "constraints_applied": [             # Parsed from instructions
      "Morning slots (8-12): sitcoms only"
    ],
    "cost_estimate": {                   # Estimated API cost
      "total_tool_calls": 24,
      "estimated_cost_usd": 0.015,
      ...
    }
  },
  
  "weekly_plan": [                       # Day-by-day summary (TBD)
    "Monday: Morning comedy block",
    ...
  ],
  
  "daily_slots": [                       # Actual schedule
    {
      "start_time": "2026-02-01T08:00:00",
      "end_time": "2026-02-01T09:00:00",
      "media_id": "friends-s01",
      "media_selection_strategy": "random",
      "category_filters": ["comedy", "sitcom"],
      "notes": []
    },
    ...
  ]
}
```

### Interpreting Completion Status

- **`complete`**: All time slots filled, quality threshold met
- **`partial`**: Some slots filled, but gaps remain
- **`failed`**: Unable to make progress (e.g., insufficient content)

### Quality Score

The quality score (0.0-1.0) considers:
- **Coverage**: Percentage of time slots filled
- **Constraint adherence**: Whether instructions were followed
- **Variety**: Content diversity (reduces repetition)
- **Flow**: Logical progression of content

### Cost Estimates

Cost estimates are rough approximations based on:
- Number of tool calls
- Estimated tokens per call
- Model pricing (as of Jan 2026)

**Typical costs** (as of Phase 2):
- Economy tier: $0.01 - $0.02 per schedule
- Balanced tier: $0.05 - $0.10 per schedule
- Premium tier: $0.20 - $0.50 per schedule

---

## Troubleshooting

### "No slots were scheduled"

**Possible causes:**
1. **Agent didn't understand instructions**: Try simpler, more explicit instructions
2. **Iteration limit too low**: Increase `max_iterations`
3. **Media library too small**: Add more content or relax constraints
4. **LLM struggled with task**: Try `cost_tier="premium"` or different model

**Debug steps:**
```bash
# Enable debug logging
export TUNABRAIN_DEBUG=1

# Increase iterations
max_iterations=30

# Check reasoning summary for clues
print(response.reasoning_summary.key_decisions)
```

### "Max iterations reached"

The agent hit the iteration limit before completing the schedule.

**Solutions:**
- Increase `max_iterations` (e.g., from 20 to 50)
- Reduce `scheduling_window_days` (fewer days = faster)
- Lower `quality_threshold` (e.g., from 0.7 to 0.6)
- Use `cost_tier="balanced"` or `"premium"` for better LLM performance

### "API key not set" errors

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Ollama (no key needed, but server must be running)
ollama serve

# Check if Ollama is responding
curl http://localhost:11434/api/tags
```

### Slow performance

**Optimize for speed:**

1. **Reduce iteration count**:
   ```python
   max_iterations=15  # Down from 50
   ```

2. **Use economy tier** (local models):
   ```bash
   export TUNABRAIN_LLM_PROVIDER=ollama
   export TUNABRAIN_LLM_MODEL=deepseek-r1:8b
   ```

3. **Smaller scheduling windows**:
   ```python
   scheduling_window_days=1  # Just one day
   ```

4. **Fewer preferred slots**:
   ```python
   preferred_slots=["18:00", "20:00"]  # Fewer boundaries
   ```

### Tests failing

**Common issues:**

1. **Async test errors**: Make sure tests are marked with `@pytest.mark.asyncio`

2. **Import errors**: Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. **LLM rate limits**: Add delays between tests:
   ```python
   import asyncio
   await asyncio.sleep(1)  # Between API calls
   ```

4. **Flaky tests**: Agent behavior can vary. Consider:
   - Relaxing assertions (e.g., `>= 0` instead of `> 3`)
   - Testing patterns, not exact values
   - Using `max_iterations` to ensure termination

### Getting help

If you're stuck:

1. **Check the logs**: Set `TUNABRAIN_DEBUG=1` and look for error messages
2. **Inspect the state**: Print intermediate agent state
3. **Try a simpler scenario**: Start with 1 day, 3 shows, no constraints
4. **Check the PLAN.md**: Current implementation status and known limitations

---

## Next Steps

Once you're comfortable with testing:

1. **Phase 3 (Next)**: Implement the remaining 5 tools (constraint parsing, media suggestion, quality evaluation)
2. **Phase 4**: Full API integration and production testing
3. **Phase 5**: Cost optimization with model tiering

For implementation details, see [PLAN.md](PLAN.md).

---

## Example Session

Here's what a successful test session looks like:

```bash
$ export OPENAI_API_KEY=sk-...
$ python test_scheduling_manual.py

╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              TunaBrain Scheduling Agent Test Suite                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

======================================================================
  Test 1: Basic Schedule - Morning Sitcoms
======================================================================

Request Configuration:
  Channel: Comedy Central Test
  Media items: 5
  Start date: 2026-02-01 08:00:00
  Duration: 1 day(s)
  Instructions: Fill morning time slots (8 AM - 1 PM) with sitcoms...
  Max iterations: 15
  Cost tier: balanced

🤖 Running scheduling agent...

======================================================================
  Results
======================================================================

📋 Overview:
Scheduled 6 time slots across 1 days for Comedy Central Test.

🔍 Reasoning Summary:
  Total iterations: 8
  Completion status: partial
  Quality score: 0.70
  Unfilled slots: 2

  Key decisions:
    1. Agent ran with basic tools
    2. Filled 6 time slots

  Cost estimate:
    {
      "total_tool_calls": 16,
      "note": "Cost estimation coming in Phase 5"
    }

📺 Schedule (sorted by time):
  Sat 02/01 08:00 - 09:00: friends-s01
  Sat 02/01 09:00 - 10:00: seinfeld-s01
  Sat 02/01 10:00 - 11:00: office-us-s01
  Sat 02/01 11:00 - 12:00: parks-rec-s01
  Sat 02/01 12:00 - 13:00: 30-rock-s01
  Sat 02/01 13:00 - 14:00: friends-s01

✅ All scenarios executed successfully!
```
