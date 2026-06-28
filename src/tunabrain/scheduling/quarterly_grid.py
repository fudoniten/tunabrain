"""Quarterly grid proposal (Phase 4) - two-pass dayparting + strip fill.

Replaces the old "big bang" approach. A channel's frozen grid is authored
top-down so no single LLM call ever has to emit the whole quarter:

- **Pass A (dayparting):** one small call proposes a ``DaypartSkeleton`` -
  4-5 coarse blocks with assigned programming roles. The coherence-bearing frame.
- **Pass B (strip fill):** one small call *per daypart* fills concrete recurring
  ``GridStrip``s within that block's bounds, against the ``CatalogProfile`` and
  seeded with the strips already chosen for earlier dayparts (for consistency).

Capacity math is never asked of the LLM here - that is the deterministic
feasibility checker's job (run by Tunarr Scheduler), whose findings come back
through :func:`repair_quarterly_grid`.

This module is stateless: it receives a profile, returns a proposal.
"""

from __future__ import annotations

import json
import logging

from openai import LengthFinishReasonError

from tunabrain.api.models import (
    QuarterlyGridRepairRequest,
    QuarterlyGridRequest,
)
from tunabrain.config import get_settings
from tunabrain.llm import LLMTask, get_chat_model
from tunabrain.scheduling.grid import (
    CatalogProfile,
    Content,
    DaypartBlock,
    DaypartSkeleton,
    Grid,
    GridStrip,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def summarize_catalog_profile(
    profile: CatalogProfile,
    max_shows: int = 40,
    *,
    min_available_episodes: int = 1,
) -> str:
    """Render the catalog *shape* compactly for a prompt (never raw media).

    Shows with fewer than ``min_available_episodes`` available episodes are
    omitted from the per-show list: they cannot be stripped into a schedule, so
    listing them only burns prompt budget and tempts the model to plan blocks it
    can't fill. Genres with no available episodes are pruned for the same reason.
    The remaining schedulable shows are listed highest-volume first up to
    ``max_shows``; any further tail is acknowledged with a count for honesty.
    """
    lines: list[str] = []

    if profile.genres:
        genre_bits = [
            f"{g.genre} ({g.show_count} shows / {g.episode_count} eps)"
            for g in sorted(profile.genres, key=lambda g: g.episode_count, reverse=True)
            if g.episode_count > 0
        ]
        if genre_bits:
            lines.append("Genres: " + ", ".join(genre_bits))

    if profile.runtime_histogram:
        rt_bits = [f"{b.label}: {b.item_count}" for b in profile.runtime_histogram]
        lines.append("Runtimes: " + ", ".join(rt_bits))

    if profile.movie_count:
        lines.append(f"Movies available: {profile.movie_count}")

    schedulable = sorted(
        (s for s in profile.shows if s.available_episode_count >= min_available_episodes),
        key=lambda s: s.available_episode_count,
        reverse=True,
    )
    dropped_unschedulable = len(profile.shows) - len(schedulable)

    lines.append("")
    lines.append("Shows (media_id | available eps | avg runtime | genres):")
    for s in schedulable[:max_shows]:
        runtime = f"~{round(s.avg_runtime_minutes)}min" if s.avg_runtime_minutes else "?min"
        genres = ",".join(s.genres) if s.genres else "-"
        lines.append(
            f"  - {s.title} ({s.media_id}) | {s.available_episode_count} eps | {runtime} | {genres}"
        )
    if len(schedulable) > max_shows:
        lines.append(f"  ... and {len(schedulable) - max_shows} more schedulable shows")
    if dropped_unschedulable > 0:
        lines.append(
            f"  ({dropped_unschedulable} further shows have no available episodes and are omitted)"
        )

    return "\n".join(lines)


def build_daypart_skeleton_prompt(request: QuarterlyGridRequest) -> list[dict]:
    """Pass A: propose the coarse dayparting for the channel."""
    theme = (
        f"\nQUARTERLY THEME (for coherence): {request.quarterly_theme}"
        if request.quarterly_theme
        else ""
    )
    guidance = (
        f"\nSTRATEGIC GUIDANCE: {request.strategic_guidance}"
        if request.strategic_guidance
        else ""
    )

    system_prompt = """You are a TV programming strategist designing the DAYPARTING for a single channel.

Divide the broadcast day into 4-5 coarse blocks, each with a clear programming role. This is a high-level frame; specific shows are chosen later.

Respond in valid JSON ONLY, matching this schema:
{
  "blocks": [
    {
      "name": "string (e.g. 'early_morning', 'daytime', 'prime', 'late_night')",
      "start": "HH:MM (24h)",
      "end": "HH:MM (24h); use end <= start for blocks that cross midnight",
      "role": "string (programming intent, e.g. 'marquee sitcoms')",
      "genre_focus": ["string", ...],
      "rationale": "string (1 sentence)"
    }
  ]
}

RULES:
- Blocks must tile the full broadcast day contiguously (each block's end == next block's start), wrapping around midnight back to the broadcast day start.
- Return ONLY JSON, no markdown."""

    user_prompt = f"""Design the dayparting for channel "{request.channel.name}".
Channel purpose: {request.channel.description}
Broadcast day starts at {request.broadcast_day_start}.{theme}{guidance}

AVAILABLE MEDIA (shape only):
{summarize_catalog_profile(request.catalog_profile, max_shows=get_settings().schedule_max_shows)}

Produce 4-5 contiguous dayparts covering the whole broadcast day."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_strip_fill_prompt(
    request: QuarterlyGridRequest,
    block: DaypartBlock,
    prior_strips: list[GridStrip],
) -> list[dict]:
    """Pass B: fill concrete recurring strips within one daypart block."""
    prior = ""
    if prior_strips:
        prior_lines = [
            f"  - {s.days} {s.start}-{s.end}: {s.content.media_id}" for s in prior_strips
        ]
        prior = (
            "\nALREADY SCHEDULED IN OTHER DAYPARTS (keep the channel coherent, "
            "avoid clashing choices):\n" + "\n".join(prior_lines)
        )

    system_prompt = """You are filling concrete recurring STRIPS within ONE daypart of a frozen weekly grid.

A strip is a recurring rule, not a dated slot: "weekdays 17:00-18:00 -> Seinfeld" covers every matching weekday all quarter.

Respond in valid JSON ONLY:
{
  "strips": [
    {
      "days": "daily" | "weekdays" | "weekends" | ["mon","wed","fri", ...],
      "start": "HH:MM",
      "end": "HH:MM (end <= start wraps past midnight)",
      "media_id": "series:<id> | movie:<id> | random:<category> (use ids from the catalog)",
      "strategy": "sequential | random | specific",
      "category_filters": ["string", ...],
      "label": "string (short, for the GUI)"
    }
  ]
}

RULES:
- Every strip must lie WITHIN this daypart's time bounds.
- Strips within the daypart must not overlap each other.
- Prefer 'sequential' for a single series stripped across days; 'random' for a rotating pool.
- Choose shows that plausibly have enough episodes for the strip's weekly frequency (do not do precise math; a downstream checker validates capacity).
- Return ONLY JSON, no markdown."""

    user_prompt = f"""Channel: "{request.channel.name}" - {request.channel.description}

DAYPART TO FILL:
  name: {block.name}
  bounds: {block.start}-{block.end}
  role: {block.role}
  genre focus: {", ".join(block.genre_focus) if block.genre_focus else "(none specified)"}
{prior}

AVAILABLE MEDIA (shape only):
{summarize_catalog_profile(request.catalog_profile, max_shows=get_settings().schedule_max_shows)}

Fill this daypart with recurring strips that realize its role."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_grid_repair_prompt(request: QuarterlyGridRepairRequest) -> list[dict]:
    """Repair: revise only the strips flagged by the feasibility checker."""
    findings = []
    for f in request.feasibility_report.strip_findings:
        if f.status != "ok":
            findings.append(
                f"  - strip '{f.rule_id}' ({f.media_id}): needs {f.slots_required} airings, "
                f"only {f.episodes_available} episodes available [{f.status}]. {f.message}"
            )
    overlaps = [f"  - overlap: {o}" for o in request.feasibility_report.overlaps]
    uncovered = [f"  - uncovered: {u}" for u in request.feasibility_report.uncovered_intervals]
    problem_block = "\n".join(findings + overlaps + uncovered) or "  (no specific findings)"

    current_strips = json.dumps(
        [
            {
                "strip_id": s.strip_id,
                "days": s.days,
                "start": s.start,
                "end": s.end,
                "media_id": s.content.media_id,
                "strategy": s.content.strategy,
            }
            for s in request.current_grid.strips
        ],
        indent=2,
    )

    system_prompt = """You are repairing a frozen weekly TV grid to resolve feasibility problems found by a deterministic checker.

Respond in valid JSON ONLY, returning the COMPLETE corrected strip list:
{
  "strips": [
    {"strip_id": "string (keep existing ids; only invent ids for genuinely new strips)",
     "days": ..., "start": "HH:MM", "end": "HH:MM",
     "media_id": "...", "strategy": "...", "category_filters": [...], "label": "..."}
  ],
  "changes": ["string (one line per change you made)"]
}

RULES:
- Change ONLY the strips named in the findings (capacity shortfalls, overlaps, gaps). Leave every other strip byte-identical.
- To fix a shortfall: reduce the strip's weekly frequency, swap to a show with more episodes, or pool into a 'random' rotation.
- Return ONLY JSON."""

    user_prompt = f"""Channel: "{request.channel.name}"

CURRENT STRIPS:
{current_strips}

FEASIBILITY FINDINGS TO FIX:
{problem_block}

AVAILABLE MEDIA (shape only):
{summarize_catalog_profile(request.catalog_profile, max_shows=get_settings().schedule_max_shows)}

Return the full corrected strip list, changing only what the findings require."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------


def _invoke_json(messages: list[dict], *, max_tokens: int, temperature: float) -> dict:
    """Invoke the scheduling LLM and parse a JSON object response.

    Reasoning-capable models (e.g. ``deepseek-v4-flash``) spend part of the
    completion budget on hidden reasoning tokens *before* emitting any JSON. If
    the budget runs out first the response is truncated mid-object and the
    OpenAI structured-output path raises ``LengthFinishReasonError`` rather than
    returning partial content. Catch it and surface an actionable message
    instead of letting it bubble up as an opaque 500.
    """
    llm = get_chat_model(LLMTask.SCHEDULING)
    try:
        response = llm.invoke(
            messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LengthFinishReasonError as e:
        logger.error(
            "LLM response truncated at the %d-token completion limit before valid "
            "JSON was produced (reasoning models consume part of this budget on "
            "hidden reasoning tokens).",
            max_tokens,
        )
        raise ValueError(
            f"LLM response hit the {max_tokens}-token completion budget before "
            "returning valid JSON. Raise the scheduling token budget or select a "
            "model that spends fewer reasoning tokens."
        ) from e
    try:
        return json.loads(response.content)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s; body: %s", e, response.content[:500])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


def _parse_skeleton(channel: str, payload: dict) -> DaypartSkeleton:
    blocks = [DaypartBlock(**b) for b in payload.get("blocks", [])]
    if not blocks:
        raise ValueError("Dayparting returned no blocks")
    return DaypartSkeleton(channel=channel, blocks=blocks)


def _parse_strips(channel: str, daypart: str, payload: dict, start_index: int) -> list[GridStrip]:
    strips: list[GridStrip] = []
    for i, raw in enumerate(payload.get("strips", [])):
        content = Content(
            media_id=raw["media_id"],
            strategy=raw.get("strategy", "sequential"),
            category_filters=raw.get("category_filters", []) or [],
            label=raw.get("label"),
        )
        strips.append(
            GridStrip(
                strip_id=f"{channel}-{daypart}-{start_index + i}".lower().replace(" ", "_"),
                days=raw["days"],
                start=raw["start"],
                end=raw["end"],
                content=content,
                daypart=daypart,
            )
        )
    return strips


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def propose_quarterly_grid(
    request: QuarterlyGridRequest,
) -> tuple[Grid, DaypartSkeleton, list[str], int]:
    """Run Pass A (dayparting) then Pass B (strip fill per daypart).

    Returns:
        (grid, skeleton, warnings, llm_calls)
    """
    logger.info(
        "Proposing quarterly grid for channel '%s' (%s shows in profile)",
        request.channel.name,
        len(request.catalog_profile.shows),
    )
    warnings: list[str] = []

    # Pass A - dayparting skeleton (one small call). The budget must leave room
    # for reasoning models to "think" before emitting JSON, hence well above the
    # few hundred tokens the skeleton itself needs.
    skeleton_payload = _invoke_json(
        build_daypart_skeleton_prompt(request), max_tokens=4096, temperature=0.3
    )
    skeleton = _parse_skeleton(request.channel.name, skeleton_payload)
    logger.info("Dayparting produced %s blocks", len(skeleton.blocks))
    llm_calls = 1

    # Pass B - fill strips per daypart (one small call each), seeded with prior strips
    all_strips: list[GridStrip] = []
    for block in skeleton.blocks:
        payload = _invoke_json(
            build_strip_fill_prompt(request, block, all_strips),
            max_tokens=4096,
            temperature=0.4,
        )
        llm_calls += 1
        block_strips = _parse_strips(
            request.channel.name, block.name, payload, len(all_strips)
        )
        if not block_strips:
            warnings.append(f"Daypart '{block.name}' returned no strips")
        all_strips.extend(block_strips)

    default_content = (
        Content(media_id=request.default_media_id, strategy="random")
        if request.default_media_id
        else None
    )

    grid = Grid(
        channel=request.channel.name,
        broadcast_day_start=request.broadcast_day_start,
        skeleton=skeleton,
        strips=all_strips,
        default_content=default_content,
    )
    logger.info("Grid proposed: %s strips across %s dayparts", len(all_strips), len(skeleton.blocks))
    return grid, skeleton, warnings, llm_calls


async def repair_quarterly_grid(
    request: QuarterlyGridRepairRequest,
) -> tuple[Grid, list[str], int]:
    """Targeted repair pass driven by deterministic feasibility findings.

    Returns:
        (revised_grid, changes, llm_calls)
    """
    logger.info(
        "Repairing grid for channel '%s' (%s findings)",
        request.channel.name,
        len(request.feasibility_report.strip_findings),
    )
    payload = _invoke_json(
        build_grid_repair_prompt(request), max_tokens=4096, temperature=0.2
    )

    revised_strips = _parse_strips_preserving_ids(
        request.channel.name, payload.get("strips", [])
    )
    changes = payload.get("changes", []) or []

    revised = request.current_grid.model_copy(update={"strips": revised_strips})
    logger.info("Grid repaired: %s strips, %s changes", len(revised_strips), len(changes))
    return revised, changes, 1


def _parse_strips_preserving_ids(channel: str, raw_strips: list[dict]) -> list[GridStrip]:
    """Parse repaired strips, keeping caller-supplied strip_ids where present."""
    strips: list[GridStrip] = []
    for i, raw in enumerate(raw_strips):
        content = Content(
            media_id=raw["media_id"],
            strategy=raw.get("strategy", "sequential"),
            category_filters=raw.get("category_filters", []) or [],
            label=raw.get("label"),
        )
        strip_id = raw.get("strip_id") or f"{channel}-repair-{i}".lower().replace(" ", "_")
        strips.append(
            GridStrip(
                strip_id=strip_id,
                days=raw["days"],
                start=raw["start"],
                end=raw["end"],
                content=content,
                daypart=raw.get("daypart"),
            )
        )
    return strips
