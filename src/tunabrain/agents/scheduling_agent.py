"""Autonomous scheduling agent using LangGraph."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tunabrain.agents.scheduling_state import SchedulingState
from tunabrain.agents.scheduling_tools import fill_time_slot, identify_schedule_gaps
from tunabrain.api.models import Channel, DailySlot, ScheduleRequest, ScheduleResponse
from tunabrain.llm import get_chat_model

logger = logging.getLogger(__name__)


# System prompt template
SCHEDULING_SYSTEM_PROMPT = """You are an expert TV programming scheduler for the channel "{channel_name}".

YOUR GOAL:
Build a complete {num_days}-day schedule that fills all available time slots with appropriate content.

AVAILABLE TOOLS:
- identify_schedule_gaps: Find unfilled time slots in the schedule
- fill_time_slot: Add a show/movie to a specific time slot

YOUR PROCESS:
1. Use identify_schedule_gaps to see what time slots need filling
2. For each gap, decide what content would be appropriate
3. Use fill_time_slot to add content to the schedule
4. Continue until all gaps are filled or you've done your best

CONSTRAINTS:
{user_instructions}

CURRENT STATUS:
- Iteration: {iteration}/{max_iterations}
- Slots filled so far: {filled_slots}
- Total slots needed: ~{total_slots_estimate}

IMPORTANT:
- Work through gaps systematically (day by day, or time by time)
- When you have no more gaps to fill, simply respond without calling any tools
- If you get stuck or reach max iterations, that's okay - return what you have

Available media library has {media_count} items to choose from.
"""


def build_scheduling_system_prompt(state: SchedulingState) -> str:
    """Build the system prompt with current state context."""
    # Count filled slots
    filled_slots = sum(len(slots) for slots in state["current_schedule"].values())

    # Estimate total slots needed (rough: 20 hours/day * num_days / 1 hour per slot)
    total_slots_estimate = state["scheduling_window_days"] * 20

    user_instructions = state["user_instructions"] or "No specific constraints provided."

    return SCHEDULING_SYSTEM_PROMPT.format(
        channel_name=state["channel"].name,
        num_days=state["scheduling_window_days"],
        user_instructions=user_instructions,
        iteration=state["iterations"],
        max_iterations=state["max_iterations"],
        filled_slots=filled_slots,
        total_slots_estimate=total_slots_estimate,
        media_count=len(state["media_library"]),
    )


def initialize_state(request: ScheduleRequest) -> SchedulingState:
    """Create initial state from API request."""
    # Calculate end_date if not provided
    end_date = request.end_date or (
        request.start_date + timedelta(days=request.scheduling_window_days)
    )

    # Extract pre-scheduled slots and mark as immutable
    immutable_slots = set()
    current_schedule = defaultdict(list)

    for slot in request.daily_slots:
        day_key = slot.start_time.strftime("%Y-%m-%d")
        current_schedule[day_key].append(
            {
                "start_time": slot.start_time.isoformat(),
                "end_time": slot.end_time.isoformat(),
                "media_id": slot.media_id,
                "media_selection_strategy": slot.media_selection_strategy,
                "category_filters": slot.category_filters,
                "notes": slot.notes,
            }
        )
        immutable_slots.add(f"{day_key}:{slot.start_time.isoformat()}")

    # Create initial prompt message
    initial_prompt = (
        f"Please create a {request.scheduling_window_days}-day schedule "
        f"for {request.channel.name} starting from {request.start_date.strftime('%Y-%m-%d')}."
    )

    if request.user_instructions:
        initial_prompt += f"\n\nInstructions: {request.user_instructions}"

    if request.daily_slots:
        initial_prompt += (
            f"\n\nNote: {len(request.daily_slots)} slots are already scheduled "
            "and should not be modified."
        )

    return {
        "messages": [HumanMessage(content=initial_prompt)],
        "channel": request.channel,
        "media_library": request.media,
        "user_instructions": request.user_instructions,
        "scheduling_window_days": request.scheduling_window_days,
        "start_date": request.start_date,
        "end_date": end_date,
        "preferred_slots": request.preferred_slots,
        "cost_tier": request.cost_tier,
        "max_iterations": request.max_iterations,
        "quality_threshold": request.quality_threshold,
        "constraints": None,
        "current_schedule": dict(current_schedule),
        "immutable_slots": immutable_slots,
        "media_analysis": None,
        "gap_analysis": None,
        "iterations": 0,
        "confidence_score": 0.0,
        "completion_status": "in_progress",
        "key_decisions": [],
        "tool_calls_made": [],
    }


def agent_planner(state: SchedulingState) -> dict:
    """Main reasoning node - LLM decides what to do next."""
    # Increment iteration counter
    iterations = state["iterations"] + 1

    logger.info(f"Agent planner iteration {iterations}/{state['max_iterations']}")

    # Get appropriate LLM based on cost tier
    # For now, always use default (we'll implement tiering in Phase 5)
    llm = get_chat_model()

    # Bind tools to LLM
    tools = [identify_schedule_gaps, fill_time_slot]
    llm_with_tools = llm.bind_tools(tools)

    # Build system prompt with current state context
    system_prompt = build_scheduling_system_prompt(state)

    # Invoke LLM
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    logger.debug(f"Invoking LLM with {len(messages)} messages")
    response = llm_with_tools.invoke(messages)

    # Track tool calls for cost estimation
    tool_calls_made = list(state["tool_calls_made"])
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_calls_made.append(
                {
                    "name": tool_call.get("name", "unknown"),
                    "iteration": iterations,
                }
            )
        logger.info(f"Agent wants to call {len(response.tool_calls)} tool(s)")
    else:
        logger.info("Agent finished - no more tool calls")

    return {
        "messages": [response],
        "iterations": iterations,
        "tool_calls_made": tool_calls_made,
    }


def should_continue(state: SchedulingState) -> Literal["tools", "end"]:
    """Routing logic - decide next node."""
    # Safety: max iterations reached
    if state["iterations"] >= state["max_iterations"]:
        logger.warning(f"Max iterations ({state['max_iterations']}) reached")
        return "end"

    last_message = state["messages"][-1]

    # Check if agent wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Agent decided to finish
    logger.info("Agent chose to finish")
    return "end"


def create_scheduling_agent() -> StateGraph:
    """Construct the scheduling agent graph."""
    logger.info("Creating scheduling agent graph")

    # Create tools
    tools = [identify_schedule_gaps, fill_time_slot]
    tool_node = ToolNode(tools)

    # Build the graph
    workflow = StateGraph(SchedulingState)

    # Add nodes
    workflow.add_node("planner", agent_planner)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "planner")  # Loop back after tool execution

    logger.info("Scheduling agent graph created successfully")
    return workflow.compile()


async def build_schedule_with_agent(request: ScheduleRequest) -> ScheduleResponse:
    """Build a schedule using the autonomous agent.

    This is the main entry point that will be called from the API.
    """
    logger.info(
        f"Starting autonomous scheduling for channel '{request.channel.name}' "
        f"({request.scheduling_window_days} days)"
    )

    # Initialize state
    state = initialize_state(request)

    # Create agent
    agent = create_scheduling_agent()

    # Run agent
    logger.info("Invoking agent...")
    # Set recursion limit based on max_iterations to avoid hitting LangGraph's default limit
    # Each iteration can involve planner + tools, so we need at least 2x + buffer
    config = {"recursion_limit": request.max_iterations * 2 + 20}
    final_state = await agent.ainvoke(state, config=config)

    logger.info(
        f"Agent completed after {final_state['iterations']} iterations. "
        f"Status: {final_state['completion_status']}"
    )

    # Format response (simplified for now - we'll add full formatting in Phase 4)
    filled_slots = sum(len(slots) for slots in final_state["current_schedule"].values())

    overview = (
        f"Scheduled {filled_slots} time slots across {request.scheduling_window_days} days "
        f"for {request.channel.name}."
    )

    # Convert schedule dict to DailySlot list
    daily_slots = []
    for day_slots in final_state["current_schedule"].values():
        for slot_dict in day_slots:
            daily_slots.append(
                DailySlot(
                    start_time=datetime.fromisoformat(slot_dict["start_time"]),
                    end_time=datetime.fromisoformat(slot_dict["end_time"]),
                    media_id=slot_dict["media_id"],
                    media_selection_strategy=slot_dict["media_selection_strategy"],
                    category_filters=slot_dict["category_filters"],
                    notes=slot_dict["notes"],
                )
            )

    # Sort by start time
    daily_slots.sort(key=lambda s: s.start_time)

    # Build reasoning summary (simplified for now)
    from tunabrain.api.models import ReasoningSummary

    reasoning_summary = ReasoningSummary(
        total_iterations=final_state["iterations"],
        key_decisions=final_state["key_decisions"][-10:]
        if final_state["key_decisions"]
        else [
            "Agent ran with basic tools",
            f"Filled {filled_slots} time slots",
        ],
        constraints_applied=[request.user_instructions] if request.user_instructions else [],
        completion_status="partial",  # Will implement proper detection later
        unfilled_slots_count=0,  # Will calculate properly later
        quality_score=0.5,  # Placeholder
        cost_estimate={
            "total_tool_calls": len(final_state["tool_calls_made"]),
            "note": "Cost estimation coming in Phase 5",
        },
    )

    return ScheduleResponse(
        overview=overview,
        reasoning_summary=reasoning_summary,
        weekly_plan=[],  # Will implement in Phase 4
        daily_slots=daily_slots,
    )
