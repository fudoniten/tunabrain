"""Schedule review / critique loop (Phase 7).

The propose -> feasibility -> repair loop guarantees a grid is *structurally*
sound (enough episodes, no overlaps, no uncovered air). It says nothing about
whether the result is any *good*: a grid can pass every feasibility check and
still be a wall of ``random:<genre>`` pools with no recurring identity, repeat
the same show three times a day, or ignore the very daypart roles it was filled
from. Those are taste judgements, and they only become visible on a CONCRETE
week -- real show titles in real slots -- which is exactly the artifact a cheap
LLM can critique well (this is the whole reason the earlier hand-off to a small
local model "just worked": it was looking at explicit media, not a catalog
summary).

Two functions, mirroring the propose/repair split:

- :func:`review_grid` -- critique a realized sample week against the channel's
  daypart plan, returning a verdict + actionable findings. Uses the dedicated
  ``LLMTask.SCHEDULE_REVIEW`` model.
- :func:`revise_grid_from_review` -- apply a failed review's findings to the
  grid, returning a revised grid (same shape as a feasibility repair). Uses the
  schedule-authoring model, since it is authoring strips.

Tunarr Scheduler drives the loop: expand a sample week, review, and -- while the
review fails and a bounded revise budget remains -- revise and re-review. This
module is stateless: it receives the grid + sample week, returns a critique or a
revision.
"""

from __future__ import annotations

import logging

from tunabrain.api.models import (
    ReviewReviseRequest,
    ScheduleReview,
    ScheduleReviewRequest,
)
from tunabrain.llm import LLMTask
from tunabrain.scheduling.grid import Grid, GridStrip
from tunabrain.scheduling.quarterly_grid import (
    _invoke_json,
    _parse_strips_preserving_ids,
    summarize_catalog_profile,
)

logger = logging.getLogger(__name__)

_WEEKDAY_ORDER = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_daypart_plan(request: ScheduleReviewRequest) -> str:
    """Render the daypart skeleton (the *intent* the week should honour)."""
    if not request.skeleton or not request.skeleton.blocks:
        return "(no explicit daypart plan supplied)"
    lines = []
    for b in request.skeleton.blocks:
        focus = f" [focus: {', '.join(b.genre_focus)}]" if b.genre_focus else ""
        lines.append(f"  - {b.name} {b.start}-{b.end}: {b.role}{focus}")
    return "\n".join(lines)


def render_sample_week(request: ScheduleReviewRequest) -> str:
    """Render the concrete realized week, grouped by weekday in time order."""
    if not request.sample_week:
        return "(empty week)"
    by_day: dict[str, list] = {d: [] for d in _WEEKDAY_ORDER}
    for slot in request.sample_week:
        by_day.setdefault(slot.day, []).append(slot)
    lines = []
    for day in _WEEKDAY_ORDER:
        slots = sorted(by_day.get(day, []), key=lambda s: s.start)
        if not slots:
            continue
        lines.append(f"{day.upper()}:")
        for s in slots:
            strat = "" if s.strategy == "sequential" else f" ({s.strategy})"
            lines.append(f"  {s.start}-{s.end}  {s.label}{strat}")
    return "\n".join(lines)


def build_review_prompt(request: ScheduleReviewRequest) -> list[dict]:
    """Ask the reviewer to judge the realized week against its daypart plan."""
    profile_block = ""
    if request.catalog_profile:
        profile_block = (
            "\n\nAVAILABLE MEDIA (shape only — use this to judge whether named series are "
            "being under-used in favour of generic pools):\n"
            + summarize_catalog_profile(request.catalog_profile)
        )

    system_prompt = """You are a critical TV programming reviewer. You are shown a channel's daypart PLAN and a concrete SAMPLE WEEK realized from that plan (real shows in real slots). Judge the week as a viewer and a programmer would — not its structure (that's already validated), but its TASTE.

Assess these dimensions:
- variety: does the same show/pool repeat too often, or within a day? Is there enough range?
- daypart-fit: does each block honour its stated role (e.g. a "marquee sitcoms" prime block actually featuring marquee sitcoms, not a generic pool)?
- series-usage: are specific recurring series anchoring the week where they should, or is it mostly `random:<genre>` when named series are available?
- pacing: sensible block lengths and transitions; no jarring whiplash.
- coherence: does the week read as a channel with an identity, or as noise?

Respond in valid JSON ONLY:
{
  "verdict": "pass" | "fail",
  "score": 0.0-1.0,           // overall taste quality
  "summary": "one paragraph overall assessment",
  "findings": [
    {
      "aspect": "variety" | "daypart-fit" | "genericness" | "series-usage" | "pacing" | "coherence" | "other",
      "severity": "minor" | "major",
      "message": "what's wrong, concretely and actionably",
      "target": "daypart name or strip_id this is about (optional)"
    }
  ]
}

RULES:
- verdict is "fail" if and ONLY if there is at least one "major" finding. Otherwise "pass".
- Be specific and actionable — "prime block is all random:comedy though Seinfeld/Cheers/Frasier are available" beats "needs more variety".
- A genuinely good week has an empty or minor-only findings list and a high score. Do not invent major problems to look thorough.
- Return ONLY JSON, no markdown."""

    user_prompt = f"""Channel: "{request.channel.name}" — {request.channel.description}

DAYPART PLAN (the intent this week should realize):
{render_daypart_plan(request)}

SAMPLE WEEK (concrete, one representative week):
{render_sample_week(request)}{profile_block}

Review this week. Return the JSON verdict."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _render_review_for_revision(review: ScheduleReview) -> str:
    """Render a failed review's findings as a fix list for the reviser."""
    if not review.findings:
        return "  (no specific findings — improve overall variety and daypart fit)"
    lines = []
    for f in review.findings:
        tgt = f" @{f.target}" if f.target else ""
        lines.append(f"  - [{f.severity}] {f.aspect}{tgt}: {f.message}")
    return "\n".join(lines)


def build_revise_prompt(request: ReviewReviseRequest) -> list[dict]:
    """Ask the schedule author to revise the grid to address a failed review."""
    import json

    current_strips = json.dumps(
        [
            {
                "strip_id": s.strip_id,
                "days": s.days,
                "start": s.start,
                "end": s.end,
                "media_id": s.content.media_id,
                "strategy": s.content.strategy,
                "daypart": s.daypart,
            }
            for s in request.current_grid.strips
        ],
        indent=2,
    )

    system_prompt = """You are revising a frozen weekly TV grid to address a reviewer's TASTE critique (not a structural error — the grid is already feasible). Change as little as needed to resolve the findings; leave strips the review didn't complain about byte-identical.

Respond in valid JSON ONLY, returning the COMPLETE corrected strip list:
{
  "strips": [
    {"strip_id": "keep existing ids; only invent ids for genuinely new strips",
     "days": ..., "start": "HH:MM", "end": "HH:MM",
     "media_id": "series:<id> | movie:<id> | random:<genre>",
     "strategy": "sequential | random | specific",
     "category_filters": [...], "label": "..."}
  ],
  "changes": ["one line per change you made, referencing the finding it addresses"]
}

RULES:
- To fix genericness / series-usage: replace a `random:<genre>` strip in an anchor/prime daypart with a specific `series:<id>` from the catalog profile, 'sequential'.
- To fix variety: rotate in a different show, or split a repeated block across shows.
- Keep every strip WITHIN its daypart's time bounds and non-overlapping.
- Do not reintroduce a capacity problem: prefer well-stocked shows for high-frequency strips.
- Return ONLY JSON."""

    user_prompt = f"""Channel: "{request.channel.name}" — {request.channel.description}

REVIEWER VERDICT: {request.review.verdict} (score {request.review.score:.2f})
{request.review.summary}

FINDINGS TO ADDRESS:
{_render_review_for_revision(request.review)}

CURRENT STRIPS:
{current_strips}

AVAILABLE MEDIA (shape only):
{summarize_catalog_profile(request.catalog_profile)}

Return the full corrected strip list, changing only what the findings require."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def review_grid(request: ScheduleReviewRequest) -> tuple[ScheduleReview, int]:
    """Critique a realized sample week. Returns ``(review, llm_calls)``.

    The verdict from the model is trusted for its findings and prose, but the
    pass/fail is recomputed deterministically from finding severities so the
    contract ("fail iff a major finding exists") holds regardless of whether the
    model set ``verdict`` consistently with its own findings.
    """
    logger.info(
        "Reviewing schedule for channel='%s' (%s strips, %s sample slots)",
        request.channel.name,
        len(request.grid.strips),
        len(request.sample_week),
    )
    payload = _invoke_json(
        build_review_prompt(request),
        max_tokens=4000,
        temperature=0.3,
        task=LLMTask.SCHEDULE_REVIEW,
    )
    review = ScheduleReview(**payload)

    # Recompute verdict from severities — the model's own `verdict` is advisory.
    has_major = any(f.severity == "major" for f in review.findings)
    derived = "fail" if has_major else "pass"
    if derived != review.verdict:
        logger.info(
            "Overriding model verdict '%s' -> '%s' from finding severities",
            review.verdict,
            derived,
        )
        review = review.model_copy(update={"verdict": derived})

    logger.info(
        "Review verdict=%s score=%.2f (%s findings)",
        review.verdict,
        review.score,
        len(review.findings),
    )
    return review, 1


async def revise_grid_from_review(
    request: ReviewReviseRequest,
) -> tuple[Grid, list[str], int]:
    """Revise the grid to address a failed review. Returns ``(grid, changes, llm_calls)``."""
    logger.info(
        "Revising grid for channel='%s' against %s review findings",
        request.channel.name,
        len(request.review.findings),
    )
    payload = _invoke_json(
        build_revise_prompt(request),
        max_tokens=10000,
        temperature=0.3,
    )
    revised_strips: list[GridStrip] = _parse_strips_preserving_ids(
        request.channel.name, payload.get("strips", [])
    )
    changes = payload.get("changes", []) or []
    revised = request.current_grid.model_copy(update={"strips": revised_strips})
    logger.info("Grid revised: %s strips, %s changes", len(revised_strips), len(changes))
    return revised, changes, 1
