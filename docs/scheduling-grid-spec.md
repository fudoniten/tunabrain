# Grid-Based Scheduling — Phase 0 Contract & Expansion Spec

This document defines the integration contracts and the deterministic expansion
algorithm for the layered scheduling design. It is the single source of truth
for both the Python reference implementation in this repo and the production
implementation in **Tunarr Scheduler (Clojure)**.

## 1. Motivation

The previous monthly endpoint tried to author an entire period's schedule in one
"big bang" LLM call, re-deriving everything each run. That is both fragile (one
truncated JSON response fails the whole thing) and wrong (the schedule should
*not* change week to week).

The fix is a layered model where the schedule is **authored once and projected
many times**:

| Layer | Artifact | Author | Cadence |
|-------|----------|--------|---------|
| Quarterly | recurring **grid** (rules) | LLM (creative) | once / quarter |
| Monthly | sparse **overrides** (deltas) | LLM (creative, given the grid) | once / month |
| Weekly | concrete **slots** | deterministic expander | on demand, no LLM |

### The core invariant

> A week's structure is a pure, deterministic function of
> **(frozen grid, sparse overrides, dates)**. The same inputs always produce the
> same slots.

With no overrides, every week materializes identically. The *structure* (what
airs when) is frozen; only the *fill* (which episode of Seinfeld) may rotate, and
that happens downstream in Pseudovision via `media_selection_strategy` — it is
not a schedule change.

## 2. System responsibilities

```
Pseudovision (media server / playout)
  ├─ owns the raw library, runtimes, watched status
  ├─ exposes catalog aggregates  ──►  CatalogProfile (built by Tunarr)
  ├─ resolves media_id ("series:seinfeld") ──► concrete episode at air time
  └─ consumes the expanded DailySlot stream and streams the channels

Tunarr Scheduler (Clojure control plane, stateful)
  ├─ stores the frozen Grid + the Override list (system of record)
  ├─ runs the deterministic expander  (this spec)
  ├─ runs the feasibility checker  ──►  FeasibilityReport
  ├─ drives the propose → check → repair loop
  └─ hosts the GUI checkpoints (approve skeleton, approve grid)

Tunabrain (this service, stateless)
  └─ proposes only: dayparting skeleton, grid strips, monthly overrides.
     Receives a CatalogProfile (+ FeasibilityReport for repairs); returns a
     proposal. Never persists, never sees raw media, never expands.
```

## 3. The four contracts

Defined as Pydantic models in `src/tunabrain/scheduling/grid.py`; mirror these in
Clojure.

1. **`CatalogProfile`** — Pseudovision → Tunarr → Tunabrain. The *shape* of the
   library (per-show / per-genre counts, runtime histogram). Sized the same
   regardless of library size. The only catalog view the LLM ever sees.
2. **`Grid`** (`GridStrip` + optional `DaypartSkeleton`) — Tunabrain → Tunarr,
   then stored frozen. The recurring base layer.
3. **`Override`** (`OverrideScope`) — Tunabrain → Tunarr, then stored. Sparse,
   higher-precedence exceptions.
4. **`FeasibilityReport`** (`StripFeasibility`) — Tunarr → Tunabrain. Real
   capacity/overlap arithmetic, used as the *true* feedback for the repair pass
   (replacing the old heuristic length-checks).

## 4. Time & calendar conventions

- Times are 24h wall-clock `"HH:MM"` strings.
- A strip/override occupies the half-open interval `[start, end)`.
- **Cross-midnight:** when `end <= start`, the interval ends on the *next*
  calendar day (e.g. `22:00 → 10:00` is an 12-hour overnight block). A recurring
  pattern matches the calendar date of the **start**, so "weekdays 22:00-10:00"
  airs Mon-night through Sat-morning.
- `broadcast_day_start` (default `06:00`) bounds where `default_content` fill
  applies; it does **not** affect explicit strip placement.
- Day patterns: an explicit list of `mon..sun`, or a named group `daily` /
  `weekdays` (mon-fri) / `weekends` (sat-sun).

## 5. Precedence cascade

When intervals overlap, the winner is chosen by this tuple, compared
lexicographically (higher wins):

```
(layer_rank, scope_specificity, priority, definition_order)
```

| Field | Meaning | Values |
|-------|---------|--------|
| `layer_rank` | override beats base grid | base = 0, override = 1 |
| `scope_specificity` | more specific scope wins | specific date = 3, explicit weekday list = 2, named group = 1, daily = 0 |
| `priority` | explicit author tiebreak | integer, higher wins |
| `definition_order` | final stable tiebreak | materialization index, later wins |

This is a CSS-specificity-style cascade: a dated override (`2026-01-09`) beats a
recurring "every Friday" override in the same window, which beats the base grid.

**Partial override:** precedence is resolved *per elementary time interval*, so a
Saturday `10:00-22:00` marathon replaces only the daytime grid — the overnight
`22:00-10:00` strip survives on either side.

## 6. Expansion algorithm

Pure function `expand(grid, overrides, range_start, range_end) → DailySlot[]`.
Reference implementation: `src/tunabrain/scheduling/expander.py`.

1. **Materialize.** For each strip and override, for each matching calendar date
   in `[range_start - 1 day, range_end)` (the extra leading day lets an overnight
   strip from the prior day cover the early hours of `range_start`), compute the
   absolute `[start, end)` interval (adding a day when it wraps) and its
   precedence tuple.
2. **Sweep.** Collect all interval boundary points within the output window.
   For each elementary interval between consecutive boundaries, the
   highest-precedence candidate that fully covers it wins. If nothing covers it,
   fill with `grid.default_content` (or leave a genuine gap if there is none).
3. **Merge.** Coalesce adjacent elementary intervals won by the same rule into a
   single slot.
4. **Emit.** Build `DailySlot`s (the existing downstream shape) sorted by start
   time, clipped to `[range_start, range_end)`.

No randomness, no I/O, no LLM. The Clojure port must produce identical slots for
identical inputs; `tests/test_grid_expander.py` pins the behaviors that matter
(determinism, weekly identity, partial override, specificity cascade,
cross-midnight, default fill).

## 7. How the layers are authored (Phases 4–6, not yet built)

- **Quarterly (Pass A → Pass B → check → repair).** Pass A proposes the
  `DaypartSkeleton` (coarse, one screen). Pass B fills `GridStrip`s per daypart
  against the `CatalogProfile`. Tunarr runs the feasibility checker and feeds any
  `FeasibilityReport` shortfalls back to Tunabrain for a targeted repair. The LLM
  never does capacity math and never sees raw media.
- **Monthly.** Given the frozen grid as context, Tunabrain emits only the
  `Override` deltas for the month — usually a short list, often empty.
- **Weekly.** Tunarr calls `expand(...)`. No Tunabrain call.

## 8. Open items deferred past Phase 0

- `Override.mode` currently only supports `"replace"`; `"insert"` / `"preempt"`
  semantics are future work.
- Richer override scopes ("2nd Tuesday", "last Friday") beyond specific-date and
  weekday-pattern.
- The `query_media_count` pull tool (Phase 7) for feasibility questions the
  `CatalogProfile` did not pre-answer.
