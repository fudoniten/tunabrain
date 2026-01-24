# Quick Start: Testing the Scheduling Agent

## 🚀 Fastest Way to Test (3 minutes)

### 1. Set up your API key

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### 2. Run the manual test script

```bash
python test_scheduling_manual.py
```

That's it! You'll see three test scenarios run with visual output showing what the agent does.

---

## 📚 What I've Added

### 1. **`test_scheduling_manual.py`** - Interactive Testing Script

A standalone Python script that demonstrates:
- **Test 1**: Basic morning sitcom schedule
- **Test 2**: Gap filling around pre-scheduled content  
- **Test 3**: Multi-day scheduling (3 days)

**Features**:
- Colorful output with progress indicators
- Shows agent iterations, decisions, and results
- Easy to modify for your own scenarios

**Run it**:
```bash
python test_scheduling_manual.py
```

### 2. **`tests/test_integration.py`** - Automated Integration Tests

10 comprehensive pytest tests covering:
- Single-day scheduling
- Multi-day scheduling
- Gap filling with immutable slots
- Constraint interpretation
- Iteration limits
- Cost tiers
- Reasoning summary validation
- Schedule sorting

**Run them**:
```bash
# All tests
pytest tests/test_integration.py -v

# Specific test
pytest tests/test_integration.py -k "gap_filling" -v
```

### 3. **`TESTING.md`** - Complete Testing Guide

Full documentation covering:
- How to configure different LLM backends (OpenAI, Ollama, Claude, Azure)
- Manual testing approaches
- How to read and interpret results
- Troubleshooting common issues
- Example test sessions

---

## 🎯 Common Testing Scenarios

### Test with Different Models

**GPT-4o-mini (default, recommended)**:
```bash
export TUNABRAIN_LLM_MODEL=gpt-4o-mini
python test_scheduling_manual.py
```

**GPT-4o (premium)**:
```bash
export TUNABRAIN_LLM_MODEL=gpt-4o
python test_scheduling_manual.py
```

**Local Ollama (free)**:
```bash
# Start Ollama in another terminal
ollama serve

# Pull a model
ollama pull deepseek-r1:8b

# Run test
export TUNABRAIN_LLM_PROVIDER=ollama
export TUNABRAIN_LLM_MODEL=deepseek-r1:8b
python test_scheduling_manual.py
```

### Test with Debug Logging

See exactly what the agent is thinking:

```bash
export TUNABRAIN_DEBUG=1
python test_scheduling_manual.py
```

### Test via HTTP API

```bash
# Terminal 1: Start server
python -m tunabrain

# Terminal 2: Send request
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "channel": {"name": "Test Channel"},
  "media": [
    {"id": "show1", "title": "Show", "genres": ["comedy"], "duration_minutes": 30}
  ],
  "start_date": "2026-02-01T18:00:00",
  "scheduling_window_days": 1,
  "user_instructions": "Fill evening with comedy",
  "max_iterations": 15
}
EOF
```

---

## 🔍 Understanding the Output

When you run a test, you'll see:

### Request Configuration
```
Channel: Comedy Central Test
Media items: 5
Start date: 2026-02-01 08:00:00
Duration: 1 day(s)
Instructions: Fill morning time slots...
Max iterations: 15
Cost tier: balanced
```

### Agent Progress
```
🤖 Running scheduling agent...
```

### Results Summary
```
📋 Overview:
Scheduled 6 time slots across 1 days for Comedy Central Test.

🔍 Reasoning Summary:
  Total iterations: 8
  Completion status: partial
  Quality score: 0.70
  Unfilled slots: 2
```

### Scheduled Content
```
📺 Schedule (sorted by time):
  Sat 02/01 08:00 - 09:00: friends-s01
  Sat 02/01 09:00 - 10:00: seinfeld-s01
  Sat 02/01 10:00 - 11:00: office-us-s01
```

---

## 🎨 Customize Your Tests

### Create a Custom Test

Copy this template to `my_test.py`:

```python
import asyncio
from datetime import datetime
from tunabrain.agents.scheduling_agent import build_schedule_with_agent
from tunabrain.api.models import Channel, MediaItem, ScheduleRequest

async def main():
    channel = Channel(name="My Channel")
    media = [
        MediaItem(id="show1", title="My Show", 
                  genres=["comedy"], duration_minutes=30),
    ]
    
    request = ScheduleRequest(
        channel=channel,
        media=media,
        start_date=datetime(2026, 2, 1, 18, 0),
        scheduling_window_days=1,
        user_instructions="Your custom instructions",
        preferred_slots=["18:00", "20:00"],
        max_iterations=20,
    )
    
    response = await build_schedule_with_agent(request)
    print(f"Scheduled {len(response.daily_slots)} slots")
    for slot in response.daily_slots:
        print(f"  {slot.start_time}: {slot.media_id}")

asyncio.run(main())
```

Run it:
```bash
python my_test.py
```

---

## 💡 Key Parameters to Experiment With

### `user_instructions`
Natural language constraints:
```python
"Weekday mornings: sitcoms. Evenings: dramas. No horror before 10 PM."
```

### `preferred_slots`
Guide slot boundaries:
```python
preferred_slots=["08:00", "12:00", "18:00", "22:00"]
```

### `max_iterations`
How many agent loops (more = potentially better):
```python
max_iterations=30  # Good for complex schedules
```

### `cost_tier`
Model quality/cost tradeoff:
```python
cost_tier="economy"   # Local models, cheapest
cost_tier="balanced"  # GPT-4o-mini, recommended
cost_tier="premium"   # GPT-4o, highest quality
```

### `quality_threshold`
When to stop (higher = more iterations):
```python
quality_threshold=0.6  # Lenient
quality_threshold=0.8  # Strict
```

---

## 🐛 Troubleshooting Quick Fixes

**"No slots scheduled"**
→ Try: Increase `max_iterations`, simplify `user_instructions`, or add more media

**"Max iterations reached"**
→ Try: Increase `max_iterations` or lower `quality_threshold`

**"API key error"**
→ Check: `echo $OPENAI_API_KEY` is set correctly

**Slow performance**
→ Try: Reduce `scheduling_window_days`, use fewer `preferred_slots`, or switch to Ollama

---

## 📖 Next Steps

1. **Read TESTING.md** for comprehensive documentation
2. **Modify test_scheduling_manual.py** with your own media/channels
3. **Run integration tests** with `pytest tests/test_integration.py`
4. **Check PLAN.md** to see what's coming in Phase 3-5

---

## Current Implementation Status (Phase 2 Complete)

✅ **What works now**:
- Basic agent loop with planner and tools
- Gap identification (finds empty time slots)
- Slot filling (adds content to schedule)
- Multi-iteration reasoning
- Pre-scheduled slot preservation
- Cost tracking

🔄 **Coming in Phase 3** (next):
- Constraint parsing (LLM-powered natural language → structured rules)
- Media analysis (content feasibility checks)
- Media suggestion (smart content selection)
- Constraint validation (rule checking)
- Quality evaluation (subjective assessment)

---

## Questions?

Check the detailed guides:
- **[TESTING.md](TESTING.md)** - Full testing documentation
- **[PLAN.md](PLAN.md)** - Implementation roadmap and architecture
- **[README.md](README.md)** - Project overview

Happy testing! 🎉
