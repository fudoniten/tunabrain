"""Monthly override proposal (Phase 6) - sparse deltas over a frozen grid.

The monthly layer does NOT author a schedule. Given the channel's *frozen* grid
as context, it emits only the exceptions for the month - "Sat the 10th: Cheers
marathon", "Fridays this month: evening movie". A month with no special plans
yields an empty list, and every week then materializes identically to the grid.

Output is small and bounded by construction (a handful of overrides), so there
is no "big bang" to truncate. Recurring scopes ("Fridays this month") are bounded
deterministically to the month here, rather than trusting the LLM to get the
date math right.

This module is stateless: it receives the grid + profile, returns overrides.
"""

from __future__ import annotations

import calendar
import json
import logging
from datetime import date

from tunabrain.api.models import MonthlyOverridesRequest
from tunabrain.llm import LLMTask, get_chat_model
from tunabrain.scheduling.grid import Content, Grid, Override, OverrideScope
from tunabrain.scheduling.quarterly_grid import summarize_catalog_profile

logger = logging.getLogger(__name__)


def month_bounds(month: str) -> tuple[date, date]:
    """Return (first_day, last_day) for a 'YYYY-MM' month string."""
    year, mon = (int(p) for p in month.split("-"))
    last = calendar.monthrange(year, mon)[1]
    return date(year, mon, 1), date(year, mon, last)


def summarize_grid_for_prompt(grid: Grid) -> str:
    """Render the frozen grid compactly so the LLM proposes only deltas."""
    if not grid.strips:
        return "(empty grid)"
    lines = []
    for s in grid.strips:
        label = f" [{s.content.label}]" if s.content.label else ""
        lines.append(f"  - {s.days} {s.start}-{s.end}: {s.content.media_id}{label}")
    if grid.default_content:
        lines.append(f"  - (default fill): {grid.default_content.media_id}")
    return "\n".join(lines)


def build_monthly_overrides_prompt(request: MonthlyOverridesRequest) -> list[dict]:
    """Build the prompt asking for sparse monthly overrides."""
    events = ""
    if request.planned_events:
        events = "\nPLANNED EVENTS / OPERATOR REQUESTS:\n" + "\n".join(
            f"  - {e}" for e in request.planned_events
        )
    theme = f"\nMONTHLY THEME: {request.monthly_theme}" if request.monthly_theme else ""
    guidance = (
        f"\nSTRATEGIC GUIDANCE: {request.strategic_guidance}"
        if request.strategic_guidance
        else ""
    )

    system_prompt = """You are planning the MONTHLY EXCEPTIONS for a TV channel whose weekly grid is already frozen.

You do NOT rewrite the schedule. You only propose sparse OVERRIDES - special events that replace the normal grid in a specific time window. If nothing special is planned, return an empty list.

Respond in valid JSON ONLY:
{
  "overrides": [
    {
      "scope": {"date": "YYYY-MM-DD"}  // a single day
              | {"days": ["fri", ...]}  // recurring within this month
              | {"days": "weekends"},
      "start": "HH:MM",
      "end": "HH:MM (end <= start wraps past midnight)",
      "media_id": "series:<id> | movie:<id> | random:<category>",
      "strategy": "sequential | random | specific",
      "marathon": true | false,
      "category_filters": ["string", ...],
      "label": "string (short, for the GUI)",
      "note": "string (why this override exists)"
    }
  ]
}

RULES:
- Be SPARSE. Only override for genuine special programming; never restate the normal grid.
- An override replaces the grid ONLY within its [start, end) window; the rest of the day keeps the normal grid.
- Use 'date' for one-off events (a marathon on the 10th); use 'days' for a recurring change this month (every Friday).
- Choose media that exists in the catalog.
- Return ONLY JSON, no markdown. Empty list is a valid, common answer."""

    user_prompt = f"""Channel: "{request.channel.name}" - {request.channel.description}
Month: {request.month}{theme}{guidance}{events}

FROZEN WEEKLY GRID (do not restate - only propose exceptions to it):
{summarize_grid_for_prompt(request.grid)}

AVAILABLE MEDIA (shape only):
{summarize_catalog_profile(request.catalog_profile)}

Propose the sparse set of overrides for {request.month}. If nothing special is warranted, return an empty list."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_overrides(
    channel: str, month: str, payload: dict
) -> tuple[list[Override], list[str]]:
    """Parse override JSON, generating ids and bounding recurring scopes to the month."""
    first_day, last_day = month_bounds(month)
    overrides: list[Override] = []
    warnings: list[str] = []

    for i, raw in enumerate(payload.get("overrides", [])):
        raw_scope = raw.get("scope", {})
        if raw_scope.get("date"):
            try:
                d = date.fromisoformat(raw_scope["date"])
            except ValueError:
                warnings.append(f"Override {i} has an unparseable date '{raw_scope['date']}'; skipped")
                continue
            if not (first_day <= d <= last_day):
                warnings.append(
                    f"Override {i} date {d} is outside {month}; kept but verify intent"
                )
            scope = OverrideScope(date=raw_scope["date"])
        elif raw_scope.get("days"):
            # Bound recurring scopes to the month deterministically ("this month").
            scope = OverrideScope(
                days=raw_scope["days"],
                effective_start=first_day.isoformat(),
                effective_end=last_day.isoformat(),
            )
        else:
            warnings.append(f"Override {i} has no valid scope (date or days); skipped")
            continue

        content = Content(
            media_id=raw["media_id"],
            strategy=raw.get("strategy", "sequential"),
            marathon=bool(raw.get("marathon", False)),
            category_filters=raw.get("category_filters", []) or [],
            label=raw.get("label"),
        )
        overrides.append(
            Override(
                override_id=f"{channel}-{month}-ovr-{i}".lower().replace(" ", "_"),
                scope=scope,
                start=raw["start"],
                end=raw["end"],
                content=content,
                mode="replace",
                note=raw.get("note"),
            )
        )

    return overrides, warnings


async def propose_monthly_overrides(
    request: MonthlyOverridesRequest,
) -> tuple[list[Override], list[str], int]:
    """Propose sparse overrides for one channel-month over its frozen grid.

    Returns:
        (overrides, warnings, llm_calls)
    """
    logger.info(
        "Proposing monthly overrides for channel='%s' month=%s (%s strips in grid)",
        request.channel.name,
        request.month,
        len(request.grid.strips),
    )

    llm = get_chat_model(LLMTask.SCHEDULING)
    response = llm.invoke(
        build_monthly_overrides_prompt(request),
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=2000,
    )
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as e:
        logger.error("Monthly overrides returned invalid JSON: %s; body: %s", e, response.content[:500])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    overrides, warnings = _parse_overrides(request.channel.name, request.month, payload)
    logger.info("Proposed %s overrides (%s warnings)", len(overrides), len(warnings))
    return overrides, warnings, 1
