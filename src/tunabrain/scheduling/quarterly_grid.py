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
import random

from openai import LengthFinishReasonError

from tunabrain.api.models import (
    DaypartSkeletonRequest,
    QuarterlyGridRepairRequest,
    QuarterlyGridRequest,
    StripFillRequest,
)
from tunabrain.config import get_settings
from tunabrain.llm import LLMTask, get_chat_model
from tunabrain.scheduling.grid import (
    CatalogProfile,
    Content,
    DaypartBlock,
    DaypartCandidate,
    DaypartSkeleton,
    Grid,
    GridStrip,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _random_pool_categories(profile: CatalogProfile) -> list[tuple[str, int, int]]:
    """The bare category names a ``random:<category>`` strip may legally use,
    derived from the dimension-model ``tag_aggregates``.

    This is the *same* vocabulary Tunarr Scheduler's feasibility checker
    validates a ``random:<category>`` against (``feasibility/tag-matches?``
    accepts a category bare or ``genre:``-prefixed) and — because
    ``tag_aggregates`` is already sliced to this channel's own pool — every
    category here actually resolves to media at playout. Previously the prompt
    advertised ``profile.genres`` (a deprecated, differently-cased view) and
    each show's comma-joined ``genres`` list instead, so the model would emit
    categories the checker had never heard of (``sci-fi-and-fantasy``) or the
    comma-joined show label as one tag (``animation,family``) — dead pools that
    resolve to empty collections. Only genre-dimension (and bare, dimensionless)
    tags are eligible pools; ``channel:``/other-dimension tags are not
    categories. Sorted by episode volume, empty pools dropped.
    """
    out: list[tuple[str, int, int]] = []
    for ta in profile.tag_aggregates:
        tag = ta.tag or ""
        if tag.startswith("genre:"):
            category = tag[len("genre:") :]
        elif ":" in tag:
            continue  # a non-genre dimension (channel:, decade:, …) — not a pool
        else:
            category = tag
        if category and ta.episode_count > 0:
            out.append((category, ta.show_count, ta.episode_count))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def summarize_catalog_profile(
    profile: CatalogProfile,
    max_shows: int = 40,
    *,
    min_available_episodes: int = 1,
    rng: random.Random | None = None,
) -> str:
    """Render the catalog *shape* compactly for a prompt (never raw media).

    Shows with fewer than ``min_available_episodes`` available episodes are
    omitted from the per-show list: they cannot be stripped into a schedule, so
    listing them only burns prompt budget and tempts the model to plan blocks it
    can't fill. Genres with no available episodes are pruned for the same reason.

    When more schedulable shows remain than ``max_shows``, a strict top-N by
    episode count would hide the entire long tail *every* run, leaving those
    shows effectively dead. Instead the highest-volume shows are kept as fixed
    anchors (the best strip candidates) and the rest of the budget is sampled
    randomly from the tail, so lower-volume shows rotate into view across runs.
    Pass ``rng`` (a seeded ``random.Random``) for deterministic output in tests.
    """
    lines: list[str] = []

    # The authoritative random:<category> vocabulary, taken from the
    # channel-scoped tag_aggregates so it matches what the downstream checker
    # accepts and what actually has media. Falls back to the deprecated
    # profile.genres view only when no dimension tags are present (older
    # profiles), never showing both — two conflicting lists is exactly what let
    # the model reach for a name that doesn't resolve.
    pools = _random_pool_categories(profile)
    if pools:
        pool_bits = [f"{c} ({sc} shows / {ec} eps)" for c, sc, ec in pools]
        lines.append(
            "Random pools — the ONLY valid random:<category> values, copy a name VERBATIM: "
            + ", ".join(pool_bits)
        )
    elif profile.genres:
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

    if len(schedulable) > max_shows:
        picker = rng or random
        anchor_count = max(1, max_shows // 2)
        tail = schedulable[anchor_count:]
        sampled = picker.sample(tail, k=min(max_shows - anchor_count, len(tail)))
        selected = sorted(
            schedulable[:anchor_count] + sampled,
            key=lambda s: s.available_episode_count,
            reverse=True,
        )
    else:
        selected = schedulable

    lines.append("")
    lines.append("Shows (media_id | available eps | avg runtime | genres):")
    for s in selected:
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


def build_daypart_skeleton_prompt(
    request: QuarterlyGridRequest | DaypartSkeletonRequest,
) -> list[dict]:
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


def render_candidate_menu(candidates: list[DaypartCandidate]) -> str:
    """Render a precomputed slot-tiling menu for the strip-fill prompt.

    Each candidate is a duration-feasible way to tile the daypart, built by
    Tunarr Scheduler's scheduling/candidates.clj from the catalog's real
    per-tag runtime histogram against this exact block's bounds — never
    invented by the LLM. Guiding the model toward these lengths (rather than
    letting it invent arbitrary ones) is the whole point of splitting
    proposal into two round trips; see DURATION_AWARE_SCHEDULING.md §4.2-4.4.

    Empty/absent input renders nothing, so a call with no candidates degrades
    to exactly today's unconstrained behavior."""
    if not candidates:
        return ""
    lines = [
        "\nDURATION-FEASIBLE SLOT MENU (built from real inventory for this exact "
        "daypart — prefer these lengths over inventing your own; a length not "
        "listed here may have no matching-length content in that category):"
    ]
    for c in candidates:
        slot_bits = ", ".join(
            f"{s.duration_minutes}min {s.category} (x{s.available_count} available)"
            for s in c.slots
        )
        lines.append(f"  - {c.layout_id}: {slot_bits}")
    return "\n".join(lines)


def build_strip_fill_prompt(
    request: QuarterlyGridRequest | StripFillRequest,
    block: DaypartBlock,
    prior_strips: list[GridStrip],
    *,
    candidates: list[DaypartCandidate] | None = None,
) -> list[dict]:
    """Pass B: fill concrete recurring strips within one daypart block.

    `candidates`, when supplied (the split-round-trip path — see
    `propose_strip_fill`), is rendered as a duration-feasible menu the model
    is instructed to prefer. Omitted or empty is unconstrained, identical to
    the original single-call behavior."""
    prior = ""
    if prior_strips:
        prior_lines = [
            f"  - {s.days} {s.start}-{s.end}: {s.content.media_id}" for s in prior_strips
        ]
        prior = (
            "\nALREADY SCHEDULED IN OTHER DAYPARTS (keep the channel coherent, "
            "avoid clashing choices):\n" + "\n".join(prior_lines)
        )

    menu = render_candidate_menu(candidates or [])
    menu_rule = (
        "\n- A DURATION-FEASIBLE SLOT MENU is provided below; prefer strip lengths "
        "from it over inventing your own."
        if menu
        else ""
    )

    system_prompt = f"""You are filling concrete recurring STRIPS within ONE daypart of a frozen weekly grid.

A strip is a recurring rule, not a dated slot: "weekdays 17:00-18:00 -> Seinfeld" covers every matching weekday all quarter.

Respond in valid JSON ONLY:
{{
  "strips": [
    {{
      "days": "daily" | "weekdays" | "weekends" | ["mon","wed","fri", ...],
      "start": "HH:MM",
      "end": "HH:MM (end <= start wraps past midnight)",
      "media_id": "series:<id> | movie:<id> | random:<category> (category MUST be copied VERBATIM from the 'Random pools' list in the catalog profile above — exactly one pool name. NEVER invent a category, NEVER join two with a comma (no 'animation,family'), NEVER slugify or reuse a show's comma-joined genre label, and NEVER use 'series', 'movie', 'show', or 'episode')",
      "strategy": "sequential | random | specific",
      "category_filters": ["string", ...],
      "label": "string (short, for the GUI)"
    }}
  ]
}}

RULES:
- Every strip must lie WITHIN this daypart's time bounds.
- Strips within the daypart must not overlap each other.
- Prefer 'sequential' for a single series stripped across days; 'random' for a rotating pool.
- A random:<category> is only valid if <category> is one of the 'Random pools' names above verbatim. There is no pool for a genre that isn't listed, and no compound/comma pool — if the block wants a mix, use a listed pool or strip in named series instead.
- Choose shows that plausibly have enough episodes for the strip's weekly frequency (do not do precise math; a downstream checker validates capacity).
- SERIES-FIRST: this is a real programming grid, not a genre wheel. For an anchor/marquee/prime-style daypart (its role names a flagship slot, e.g. "prime", "marquee sitcoms", "appointment viewing"), strip in SPECIFIC named shows from the AVAILABLE MEDIA list below (`series:<media_id>`, 'sequential') — the higher its available-episode count, the better an anchor it makes. Reserve `random:<genre>` pools for daytime filler, overnight rotation, or genuinely miscellaneous blocks where no single show should dominate. A daypart described as a flagship block that resolves entirely to `random:<genre>` strips is a bad answer even if it technically fits the role.{menu_rule}
- Return ONLY JSON, no markdown."""

    user_prompt = f"""Channel: "{request.channel.name}" - {request.channel.description}

DAYPART TO FILL:
  name: {block.name}
  bounds: {block.start}-{block.end}
  role: {block.role}
  genre focus: {", ".join(block.genre_focus) if block.genre_focus else "(none specified)"}
{prior}
{menu}

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


# A malformed-JSON response is usually transient at temperature > 0 (a markdown
# fence, a stray token, a dropped brace), so a re-roll typically succeeds. Cap
# the attempts so a model that *consistently* fails still surfaces an error
# rather than looping forever.
_MAX_JSON_ATTEMPTS = 3


def _strip_code_fences(content: str) -> str:
    """Unwrap a Markdown code fence the model may have added around its JSON.

    Despite ``response_format={"type": "json_object"}``, some models routed via
    OpenRouter (e.g. ``minimax-m3``) return the object inside a ```` ```json ````
    fence. That leading backtick makes ``json.loads`` fail at "line 1 column 1",
    so peel the fence off before parsing. Text without a fence is returned
    unchanged.
    """
    text = content.strip()
    if not text.startswith("```"):
        return content
    lines = text.splitlines()
    # Drop the opening fence line (``` or ```json) ...
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # ... and the closing fence if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _invoke_json(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    task: LLMTask = LLMTask.SCHEDULING,
) -> dict:
    """Invoke a scheduling LLM and parse a JSON object response.

    ``task`` selects which model routes the call (defaults to the schedule-
    authoring model; the review loop passes ``LLMTask.SCHEDULE_REVIEW`` to use
    its own, possibly sharper, reviewer model). The reasoning-model and
    malformed-JSON handling below is task-agnostic, so it is shared verbatim.

    Reasoning-capable models (e.g. ``deepseek-v4-flash``) spend part of the
    completion budget on hidden reasoning tokens *before* emitting any JSON. If
    the budget runs out first the response is truncated mid-object and the
    OpenAI structured-output path raises ``LengthFinishReasonError`` rather than
    returning partial content. Catch it and surface an actionable message
    instead of letting it bubble up as an opaque 500.

    Malformed-but-complete JSON (markdown fences, stray tokens) is retried up to
    ``_MAX_JSON_ATTEMPTS`` times - re-rolling at temperature > 0 usually yields
    parseable output. A budget truncation is *not* retried, since a re-roll hits
    the same ceiling.
    """
    llm = get_chat_model(task)
    last_error: json.JSONDecodeError | None = None
    for attempt in range(1, _MAX_JSON_ATTEMPTS + 1):
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
            return json.loads(_strip_code_fences(response.content))
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                "LLM returned invalid JSON (attempt %d/%d): %s; body: %s",
                attempt,
                _MAX_JSON_ATTEMPTS,
                e,
                response.content[:500],
            )

    logger.error(
        "LLM returned invalid JSON after %d attempts: %s", _MAX_JSON_ATTEMPTS, last_error
    )
    raise ValueError(
        f"LLM returned invalid JSON after {_MAX_JSON_ATTEMPTS} attempts: {last_error}"
    )


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


async def propose_daypart_skeleton(
    request: QuarterlyGridRequest | DaypartSkeletonRequest,
) -> tuple[DaypartSkeleton, int]:
    """Pass A only: propose the coarse dayparting for a channel.

    Split out of the original propose_quarterly_grid so Tunarr Scheduler can
    see real daypart bounds *before* Pass B runs, and compute a duration-
    feasible candidate menu from them (scheduling/candidates.clj) to hand
    into `propose_strip_fill` per block — see DURATION_AWARE_SCHEDULING.md
    §4.3 (Option A: two round trips). `propose_quarterly_grid` below calls
    this internally and is otherwise unchanged.

    Returns:
        (skeleton, llm_calls)
    """
    logger.info(
        "Proposing daypart skeleton for channel '%s' (%s shows in profile)",
        request.channel.name,
        len(request.catalog_profile.shows),
    )
    # The budget must leave room for reasoning models to "think" before
    # emitting JSON, hence well above the few hundred tokens the skeleton
    # itself needs.
    skeleton_payload = _invoke_json(
        build_daypart_skeleton_prompt(request), max_tokens=10000, temperature=0.3
    )
    skeleton = _parse_skeleton(request.channel.name, skeleton_payload)
    logger.info("Dayparting produced %s blocks", len(skeleton.blocks))
    return skeleton, 1


async def propose_strip_fill(
    request: QuarterlyGridRequest | StripFillRequest,
    block: DaypartBlock,
    prior_strips: list[GridStrip],
    *,
    candidates: list[DaypartCandidate] | None = None,
) -> tuple[list[GridStrip], int]:
    """Pass B for ONE daypart block.

    `candidates`, when supplied, is the precomputed duration-feasible slot
    menu for this exact block (see `render_candidate_menu`); omitted or empty
    is unconstrained, identical to `propose_quarterly_grid`'s original
    per-block behavior.

    Returns:
        (strips, llm_calls)
    """
    payload = _invoke_json(
        build_strip_fill_prompt(request, block, prior_strips, candidates=candidates),
        max_tokens=10000,
        temperature=0.4,
    )
    block_strips = _parse_strips(request.channel.name, block.name, payload, len(prior_strips))
    return block_strips, 1


async def propose_quarterly_grid(
    request: QuarterlyGridRequest,
) -> tuple[Grid, DaypartSkeleton, list[str], int]:
    """Run Pass A (dayparting) then Pass B (strip fill per daypart) as a
    single call, composed from `propose_daypart_skeleton` + `propose_strip_fill`
    with no candidate menu (unconstrained strip-fill, exactly the original
    behavior). Callers that want the candidate-menu path call those two
    functions directly across two round trips instead — see
    DURATION_AWARE_SCHEDULING.md §4.3.

    Returns:
        (grid, skeleton, warnings, llm_calls)
    """
    warnings: list[str] = []

    skeleton, llm_calls = await propose_daypart_skeleton(request)

    all_strips: list[GridStrip] = []
    for block in skeleton.blocks:
        block_strips, calls = await propose_strip_fill(request, block, all_strips)
        llm_calls += calls
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
        build_grid_repair_prompt(request), max_tokens=10000, temperature=0.2
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
