# Handoff Spec — Tunarr Scheduler & Pseudovision

This document specifies the work needed in **Tunarr Scheduler** (Clojure control
plane) and **Pseudovision** (media server / playout) to complete the layered
"grid" scheduling design. The **Tunabrain** side is already implemented on
branch `claude/elegant-bohr-fkf53n` (PR #36); this is the cross-system
counterpart.

A fresh session can execute from this document alone. Where it says "see
`<path>`", that path is in the **tunabrain** repo and is the authoritative
contract.

---

## 1. The design in one paragraph

The schedule is **authored once and projected many times**. Quarterly, an LLM
proposes a *frozen weekly grid* of recurring rules ("weekdays 17:00-18:00 →
Seinfeld"). Monthly, the LLM proposes *sparse overrides* (deltas: "Sat the 10th:
Cheers marathon"). Weekly, a **deterministic, LLM-free** expander projects
(frozen grid + sparse overrides + dates) into concrete dated slots. The grid
never changes week to week; only overrides cause variation. Consistency is
structural, not something the LLM re-derives.

```
Pseudovision                Tunarr Scheduler (control plane, stateful)              Tunabrain (stateless LLM)
────────────                ──────────────────────────────────────────             ─────────────────────────
catalog aggregate  ──────►  assemble CatalogProfile  ──────────────────────────►   propose-quarterly-grid
                                     │                                              (Pass A dayparting + Pass B strips)
                            feasibility check ◄── Grid ───────────────────────────  ◄── Grid
                                     │ shortfalls
                                     └── FeasibilityReport ─────────────────────►   repair-quarterly-grid
                                     │                                              ◄── revised Grid
                            store frozen Grid
                                     │
                            (per month) ───────────────────────────────────────►   propose-monthly-overrides
                                     │                                              ◄── Override[]
                            store Overrides
                                     │
                            expand(grid, overrides, week)  ── DailySlot[] ──────►   (no Tunabrain call)
push concrete slots ◄────────────────┘
+ resolve media_id → episode
```

---

## 2. The four contracts (already defined in Tunabrain)

Authoritative source: **`src/tunabrain/scheduling/grid.py`** (Pydantic). Mirror
these as Clojure specs/schemas. Spec prose: **`docs/scheduling-grid-spec.md`**.

### 2.1 CatalogProfile  (Pseudovision → Tunarr → Tunabrain)

The *shape* of the library — never the raw items. Sized the same regardless of
library size.

```json
{
  "channel_scope": "Classic Comedy",
  "total_items": 900,
  "total_episodes": 880,
  "movie_count": 20,
  "shows": [
    {
      "media_id": "series:seinfeld",
      "title": "Seinfeld",
      "genres": ["comedy", "sitcom"],
      "episode_count": 180,
      "available_episode_count": 180,
      "avg_runtime_minutes": 22.0,
      "tags": ["classic"]
    }
  ],
  "genres": [{"genre": "comedy", "show_count": 2, "episode_count": 450}],
  "runtime_histogram": [
    {"label": "20-30min", "min_minutes": 20, "max_minutes": 30, "item_count": 450}
  ],
  "generated_at": "2026-06-24T12:00:00"
}
```

### 2.2 Grid / GridStrip / DaypartSkeleton  (Tunabrain → Tunarr, stored frozen)

```json
{
  "channel": "Classic Comedy",
  "broadcast_day_start": "06:00",
  "skeleton": {
    "channel": "Classic Comedy",
    "blocks": [
      {"name": "prime", "start": "17:00", "end": "22:00", "role": "marquee sitcoms",
       "genre_focus": ["comedy"], "rationale": "…"}
    ]
  },
  "strips": [
    {
      "strip_id": "classic_comedy-prime-0",
      "days": "weekdays",
      "start": "17:00",
      "end": "18:00",
      "content": {"media_id": "series:seinfeld", "strategy": "sequential",
                  "marathon": false, "category_filters": [], "label": "Seinfeld at Five",
                  "notes": []},
      "priority": 0,
      "daypart": "prime"
    }
  ],
  "default_content": {"media_id": "random:sitcom", "strategy": "random",
                      "marathon": false, "category_filters": [], "label": null, "notes": []}
}
```

- `days` is one of `"daily"`, `"weekdays"` (mon-fri), `"weekends"` (sat-sun), or
  an explicit list like `["mon","wed","fri"]`.
- Times are 24h `"HH:MM"`. `end <= start` ⇒ the strip crosses midnight.

### 2.3 Override / OverrideScope  (Tunabrain → Tunarr, stored)

```json
{
  "override_id": "classic_comedy-2026-01-ovr-0",
  "scope": {"date": "2026-01-10"},
  "start": "10:00",
  "end": "22:00",
  "content": {"media_id": "series:cheers", "strategy": "sequential", "marathon": true,
              "category_filters": [], "label": "Cheers Marathon", "notes": []},
  "mode": "replace",
  "priority": 0,
  "note": "Operator request"
}
```

`scope` is **exactly one of**:
- `{"date": "YYYY-MM-DD"}` — a single day (most specific), or
- `{"days": <pattern>, "effective_start": "YYYY-MM-DD", "effective_end": "YYYY-MM-DD"}`
  — recurring, bounded to a window (Tunabrain clamps these to the month).

### 2.4 FeasibilityReport  (Tunarr → Tunabrain, repair feedback)

```json
{
  "horizon_start": "2026-01-01",
  "horizon_end": "2026-04-01",
  "overall_status": "blocked",
  "strip_findings": [
    {"rule_id": "classic_comedy-prime-0", "media_id": "series:seinfeld",
     "slots_required": 65, "episodes_available": 180, "headroom_ratio": 2.77,
     "status": "ok", "message": ""}
  ],
  "overlaps": ["classic_comedy-prime-0 overlaps classic_comedy-prime-1 on weekdays 17:30-18:00"],
  "uncovered_intervals": ["weekdays 02:00-06:00"],
  "notes": []
}
```

`overall_status`: `"ok"` | `"warnings"` | `"blocked"`. `status` per finding:
`"ok"` | `"tight"` | `"shortfall"`.

### 2.5 DailySlot  (expander output → Pseudovision)

Already the existing shape (`src/tunabrain/api/models.py::DailySlot`):

```json
{
  "start_time": "2026-01-10T10:00:00",
  "end_time": "2026-01-10T22:00:00",
  "media_id": "series:cheers",
  "media_selection_strategy": "sequential",
  "category_filters": [],
  "notes": []
}
```

---

## 3. Pseudovision work

### 3.1 (Phase 2) Catalog aggregate endpoint — REQUIRED

Expose the deterministic rollup that becomes `CatalogProfile`. Pseudovision owns
the library, runtimes, and watched/eligibility state, so the aggregation lives
here.

- **Endpoint:** `GET /api/catalog/aggregate` (or POST with a filter body).
- **Query/body params:**
  - `channel` (optional) — slice to media eligible for one channel.
  - `eligibility` (optional) — what counts toward `available_episode_count`
    (e.g. `unwatched`, `all`, `in-window`). Define this to match how you track
    watched state; `available_episode_count` is the number eligible to air.
- **Returns:** the `CatalogProfile` JSON in §2.1. Notes:
  - `shows[]` is per *series* (and one entry per movie is optional; movies are
    summarized by `movie_count` + the runtime histogram).
  - `avg_runtime_minutes` drives capacity math, so populate it.
  - Keep it summarized — do **not** return per-episode rows.

### 3.2 (Phase 7, DEFERRED) Media count endpoint — OPTIONAL

Backs a future Tunabrain "pull tool" for feasibility questions the profile did
not pre-answer.

- **Endpoint:** `POST /api/catalog/count` with `{ "filters": {...} }` →
  `{ "count": N }`. Read-only, fast. Skip until Phases 1-6 are working.

### 3.3 Playout ingestion — VERIFY (likely already exists)

Pseudovision must accept the expanded `DailySlot[]` stream (§2.5) for a channel +
date range and resolve `media_id` + `media_selection_strategy` into concrete
episodes at air time. This is the same `DailySlot` shape the current `/schedule`
flow already produces, so likely **no change** beyond confirming it can be fed
from Tunarr's expander output. `media_id` conventions: `series:<id>`,
`movie:<id>`, `random:<category>`.

---

## 4. Tunarr Scheduler work

### 4.1 Mirror the contracts — REQUIRED

Define Clojure specs (clojure.spec / Malli / plain maps) for: `Content`,
`CatalogProfile` (+ `ShowProfile`, `GenreProfile`, `RuntimeBucket`), `Grid` (+
`GridStrip`, `DaypartSkeleton`, `DaypartBlock`), `Override` (+ `OverrideScope`),
`FeasibilityReport` (+ `StripFeasibility`), `DailySlot`. JSON field names must
match §2 exactly (the wire format with Tunabrain).

### 4.2 (Phase 1) Deterministic expander — REQUIRED, build first

Port `expand(grid, overrides, range_start, range_end) → DailySlot[]` from
**`src/tunabrain/scheduling/expander.py`**. This is the spine; build and test it
before anything else. It is a **pure function** — no I/O, no randomness, no LLM.

Algorithm (full prose in `docs/scheduling-grid-spec.md` §6):

1. **Materialize.** For each strip and override, for each matching calendar date
   in `[range_start − 1 day, range_end)` (the extra leading day lets an overnight
   strip from the prior day cover the early hours of `range_start`), compute the
   absolute `[start, end)` datetime interval (add a day when `end <= start`) and
   a **precedence tuple**.
2. **Sweep.** Collect all interval boundary points in the output window. For each
   elementary interval between consecutive boundaries, the highest-precedence
   candidate that *fully covers* it wins. If none covers it, fill with
   `grid.default_content` (or leave a gap if there is none).
3. **Merge** adjacent elementary intervals won by the same rule.
4. **Emit** `DailySlot`s sorted by start, clipped to `[range_start, range_end)`.

**Precedence tuple** (compared lexicographically, higher wins):
`(layer_rank, scope_specificity, priority, definition_order)`
- `layer_rank`: base grid strip = 0, override = 1.
- `scope_specificity`: specific date = 3, explicit weekday list = 2, named group
  (`weekdays`/`weekends`) = 1, `daily` = 0.
- `priority`: the integer field on the rule.
- `definition_order`: materialization index (later wins).

**Conformance:** replicate every case in **`tests/test_grid_expander.py`** as
Clojure tests — determinism, week-to-week identity, partial override (Saturday
marathon leaves the overnight strip intact), specificity cascade, cross-midnight,
default fill, empty grid. That test file is the golden spec.

### 4.3 (Phase 3) Feasibility checker — REQUIRED

Pure-ish function `(grid, catalog_profile, horizon_start, horizon_end) →
FeasibilityReport`. Deterministic arithmetic — the LLM never does this.

**Per-strip capacity** (`slots_required` = number of airings over the horizon):
```
slots_required(strip) = count of dates D in [horizon_start, horizon_end)
                        where strip.days matches weekday(D)
```
Then by `media_id` kind:
- `series:<id>` with `strategy = "sequential"`: `episodes_available` = that
  show's `available_episode_count`. To avoid repeats across the horizon you want
  `available >= slots_required`.
  - `status`: `shortfall` if `available < slots_required`;
    `tight` if `available < slots_required * MARGIN` (suggest `MARGIN = 1.2`);
    else `ok`. `headroom_ratio = available / slots_required`.
- `random:<category>` (pooled rotation): repeats are acceptable; check the pool
  is non-trivial (e.g. `episode_count` for that category ≥ a small floor).
  Usually `ok`/`tight`, rarely `shortfall`.
- `movie:<id>`: single item; flag `tight` if it would air more than once unless
  intended.

**Overlap check:** within the base grid, two strips whose day patterns intersect
*and* whose time intervals overlap → add to `overlaps[]`. (The expander resolves
overlaps by precedence, but ambiguous base-grid overlap usually signals an
authoring mistake worth surfacing.)

**Coverage check:** within the broadcast day, any time not covered by a strip and
with no `default_content` → add to `uncovered_intervals[]`.

**`overall_status`:** `blocked` if any `shortfall`; else `warnings` if any
`tight`/overlap/uncovered; else `ok`.

### 4.4 CatalogProfile assembly — REQUIRED

Call Pseudovision's aggregate endpoint (§3.1) and assemble the `CatalogProfile`
to send to Tunabrain. Slice per channel.

### 4.5 Storage — REQUIRED

Persist, as the system of record:
- One frozen **Grid** per (channel, quarter). Immutable once frozen; editable
  only via an explicit re-author/version, not by the monthly/weekly steps.
- An **Override** list per (channel, month).
Recommend versioning + an audit trail (each Grid/Override set carries the
`grid_id`/`overrides_id` Tunabrain returns).

### 4.6 (Phase 5) Orchestration: propose → check → repair — REQUIRED

For each channel, per quarter:
1. `CatalogProfile` ← §4.4.
2. `Grid` ← POST Tunabrain `propose-quarterly-grid` (§5.1).
3. `FeasibilityReport` ← §4.3 over a quarter horizon (~91 days).
4. If `overall_status != "ok"`: `Grid` ← POST `repair-quarterly-grid` (§5.2)
   with the report; re-check. **Bound the loop** (e.g. max 3 repairs), then
   accept best-effort with a flagged status.
5. Freeze and store the `Grid` (§4.5).

Per month:
6. `Override[]` ← POST `propose-monthly-overrides` (§5.3) with the frozen grid +
   that month's `CatalogProfile` + any operator `planned_events`. Store.

Per week (or on demand):
7. `DailySlot[]` ← `expand(grid, overrides, week_start, week_end)` (§4.2). Push
   to Pseudovision. **No Tunabrain call.**

### 4.7 GUI checkpoints — RECOMMENDED (can follow)

Human review falls naturally on the small artifacts: approve the
`DaypartSkeleton` (one screen) and the frozen `Grid` (a list of rules) before
committing. Reviews are of *rules*, not thousands of slots.

---

## 5. Tunabrain endpoints to call

Base: the Tunabrain service. All POST, JSON in/out. Request/response models in
`src/tunabrain/api/models.py`; logic in `src/tunabrain/scheduling/`.

### 5.1 `POST /api/scheduling/propose-quarterly-grid`

Request (`QuarterlyGridRequest`):
```json
{
  "channel": {"name": "Classic Comedy", "description": "24/7 vintage sitcoms"},
  "quarter": "Q1",
  "year": 2026,
  "catalog_profile": { /* §2.1 */ },
  "quarterly_theme": "New year, classic laughs",
  "strategic_guidance": null,
  "broadcast_day_start": "06:00",
  "default_media_id": "random:sitcom",
  "cost_tier": "balanced"
}
```
Response (`QuarterlyGridResponse`): `{ grid_id, status, grid, skeleton, warnings,
cost_estimate, suggested_next_steps }`. Runs Pass A (dayparting) + Pass B (strip
fill per daypart) internally; one channel per call.

### 5.2 `POST /api/scheduling/repair-quarterly-grid`

Request (`QuarterlyGridRepairRequest`):
```json
{
  "channel": {"name": "Classic Comedy", "description": "…"},
  "catalog_profile": { /* §2.1 */ },
  "current_grid": { /* §2.2 */ },
  "feasibility_report": { /* §2.4 */ },
  "cost_tier": "balanced"
}
```
Response (`QuarterlyGridRepairResponse`): `{ grid_id, status, grid, changes,
cost_estimate }`. Changes only the strips named in the report; preserves
`strip_id`s.

### 5.3 `POST /api/scheduling/propose-monthly-overrides`

Request (`MonthlyOverridesRequest`):
```json
{
  "channel": {"name": "Classic Comedy", "description": "…"},
  "month": "2026-01",
  "grid": { /* the frozen §2.2 grid */ },
  "catalog_profile": { /* §2.1 */ },
  "monthly_theme": null,
  "planned_events": ["Cheers marathon Saturday the 10th"],
  "strategic_guidance": null,
  "cost_tier": "balanced"
}
```
Response (`MonthlyOverridesResponse`): `{ overrides_id, status, month, overrides,
warnings, cost_estimate, suggested_next_steps }`. Output is sparse; an empty
`overrides` list is normal.

---

## 6. Recommended build order

Deterministic spine first (no LLM uncertainty, fully testable, delivers a working
channel before any AI is wired in):

1. **Tunarr §4.1** contracts → **§4.2** expander (with the conformance tests).
   *Milestone:* hand-author a `Grid` JSON, expand a week, push to Pseudovision,
   watch TV. No Tunabrain, no Pseudovision aggregate yet.
2. **Pseudovision §3.1** aggregate endpoint → **Tunarr §4.4** profile assembly.
3. **Tunarr §4.3** feasibility checker (hand-authored grid can now be validated).
4. **Tunarr §4.6** orchestration: wire the three Tunabrain calls + the
   propose→check→repair loop + storage.
5. **Tunarr §4.7** GUI checkpoints.
6. **Pseudovision §3.2** pull-count endpoint (only if/when needed).

---

## 7. Key invariants to preserve

- **Expansion is pure and deterministic.** Same (grid, overrides, dates) ⇒ same
  slots, always. No randomness in *structure*; episode rotation happens in
  Pseudovision at air time via `media_selection_strategy`, which is not a
  schedule change.
- **The grid is frozen.** The monthly and weekly steps never mutate it. All
  week-to-week variation comes from overrides.
- **The LLM never sees raw media and never does capacity math.** It sees only
  `CatalogProfile`; arithmetic lives in the feasibility checker.
- **Tunabrain is stateless.** Tunarr Scheduler holds all state.
