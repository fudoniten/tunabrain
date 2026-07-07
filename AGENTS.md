# AGENTS.md — Tunabrain

> Working notes for AI agents and humans working on this repo.
> For a deeper tour, see [README.md](README.md) and the cross-system spec at
> `docs/handoff-tunarr-pseudovision.md`.

## What this is

Tunabrain is the **LLM gateway** for the Fudo stack. It is a FastAPI service
that wraps upstream language models behind a stable HTTP contract:

- Tagging (`/tags`)
- Channel mapping (`/channel-mapping`)
- Schedule authoring (`/api/scheduling/propose-quarterly-grid`,
  `/api/scheduling/repair-quarterly-grid`, `/api/scheduling/propose-monthly-overrides`)
- Bumper script + strategy (`/api/bumpers`, `/api/scheduling/get-{quarterly,monthly}-strategy`)

It uses **LangChain** under the hood and routes across providers
(`TUNABRAIN_LLM_PROVIDER` — `openai`, `openrouter`, etc.) so the rest of the
stack doesn't depend on a specific model. **Tunabrain is stateless**; all
persistent state lives in Tunarr Scheduler (control plane) and Pseudovision
(playout).

## How it fits in the ecosystem

```
                    ┌────────────────┐
                    │   Tunabrain    │  (this repo, stateless LLM gateway)
                    │   FastAPI      │
                    └────────┬───────┘
                             │ HTTP
                             ▼
                    ┌────────────────┐
                    │ Tunarr Sched.  │  caller (Tunarr Scheduler is the
                    │                │   only client in the live cluster)
                    └────────┬───────┘
                             │ HTTP
                             ▼
                    ┌────────────────┐
                    │  Pseudovision  │
                    └────────────────┘
```

- **Single client**: Tunarr Scheduler invokes Tunabrain. Pseudovision,
  Marquee, and Grout do **not** call Tunabrain directly in the live cluster
  (Grout is planned to call `/enrich` for metadata enrichment but doesn't
  yet).
- **Upstream models**: OpenAI, OpenRouter, or anything else LangChain
  supports. Tunabrain holds **no model credentials of its own** in the
  contract — the caller sets env vars; the gateway reads them.
- **Cost tier routing**: Per-task `*_LLM_MODEL` env vars let cheap models do
  high-volume tagging and a stronger model do quarterly scheduling. See
  `README.md` for the full list.

## Live endpoints (cluster)

| Service | URL | Notes |
|---|---|---|
| Public HTTPS | `https://tunabrain.kube.sea.fudo.link` | Ingress via cert-manager |
| Health | `GET /health` | Returns 200 |
| Version | `GET /api/version` | `{git-commit, git-timestamp, version-tag}` (suffix `-dirty` if uncommitted) |
| OpenAPI | `GET /openapi.json` | FastAPI-generated |
| In-cluster | `tunabrain.arr.svc.cluster.local:5546` | In the `arr` namespace, NOT `media` (the legacy sidecar was a stub with no endpoints) |

**Deployed as of 2026-07-03:** `917398a-dirty` (live cluster has uncommitted
changes; the `-dirty` suffix in `/api/version` is the tell). The local master
is at `2fe8f99` — always cross-check before debugging a "model not behaving
the same" report.

## Local development

```bash
# Nix shell (recommended — provides Python 3.11, uv, all deps)
nix develop

# Or uv-based (modern Python tooling)
uv sync

# Run
python -m tunabrain                        # default port 8000
python -m tunabrain --port 9000

# Tests
pytest                                     # all
pytest tests/test_grid_expander.py         # scheduling spec conformance
pytest -k "propose_quarterly"              # focused

# Lint / type-check (if configured in pyproject.toml)
ruff check .
mypy src/tunabrain
```

Required env vars:

```bash
# Provider + default model
export TUNABRAIN_LLM_PROVIDER=openrouter
export TUNABRAIN_LLM_MODEL=deepseek/deepseek-v4-flash

# Stronger model for quarterly scheduling (long context, correctness matters)
export TUNABRAIN_SCHEDULE_LLM_MODEL=anthropic/claude-opus-4.8

# For OpenAI / OpenRouter
export OPENAI_API_KEY=***       # or OPENROUTER_API_KEY
```

## Source layout

```
src/tunabrain/
├── __main__.py            ; `python -m tunabrain` entry
├── app.py                 ; FastAPI app factory
├── api/
│   ├── routes.py          ; HTTP endpoints (all of them — 15 in current spec)
│   ├── models.py          ; Pydantic request/response models
│   └── ...
├── chains/                ; LangChain workflows (one module per task)
├── tools/                 ; LangChain-compatible tools (e.g. wikipedia_lookup)
├── scheduling/            ; grid / override / strategy implementations
│   ├── grid.py            ; canonical Pydantic contracts (Grid, Override, DailySlot)
│   ├── quarterly_grid.py  ; propose + repair quarterly grid
│   ├── monthly_overrides.py
│   └── strategy.py
├── enrichment/            ; tag governance, bumper scripting
└── llm/                   ; provider routing, prompt templates

docs/                      ; cross-system specs (handoff, scheduling-grid-spec)
tests/                     ; pytest suites (incl. golden conformance for the expander)
```

## Public API surface

The 17 endpoints in current spec (verify live with
`curl https://tunabrain.kube.sea.fudo.link/openapi.json | jq '.paths | keys'`):

**Tagging & classification (8):**
- `POST /tags` — generate tags for a media item
- `POST /categorize` — multi-dimension categorisation
- `POST /channel-mapping` — fit-score media to channels
- `POST /tags/episode-special-flag` — special-flag detection
- `POST /tags/audit` — recommend tag deletions
- `POST /tag-governance/triage` — merge / consolidate
- `POST /bumpers` — promo text generation
- `POST /schedule` — legacy "schedule for N days" (predates `/api/scheduling/*`)

**Grout enrichment (2):**
- `POST /enrich/short-form` — one-call categorize + tags for short-form filler
  (bumpers, ads, music videos); no STT
- `POST /enrich/long-form` — fetch → STT (+ optional keyframe captions) →
  categorize + tags for long-form media. STT backend is pluggable
  (`whisper-http` / `subgen` / `auto`); shells out to `ffmpeg`. See `README.md`
  for the `TUNABRAIN_STT_*` / `TUNABRAIN_SCRATCH_DIR` / `TUNABRAIN_ENRICH_LONG_TIMEOUT`
  env vars.

**Scheduling (5) — the new control plane:**
- `POST /api/scheduling/propose-quarterly-grid` — Phase 2: LLM authors a
  frozen weekly skeleton
- `POST /api/scheduling/repair-quarterly-grid` — Phase 2 repair loop
- `POST /api/scheduling/propose-monthly-overrides` — Phase 3: sparse deltas
- `POST /api/scheduling/get-quarterly-strategy` — optional strategy phase
- `POST /api/scheduling/get-monthly-strategy` — optional strategy phase

**Health & version (2):**
- `GET /health`, `GET /api/version`

The Pydantic contracts in `src/tunabrain/scheduling/grid.py` are the
authoritative wire format. The Clojure side (`tunarr-scheduler`) mirrors
these as Malli schemas. The prose spec is in
`docs/scheduling-grid-spec.md`. **When changing a contract, change all three.**

## Common pitfalls

1. **The Pydantic contracts in `scheduling/grid.py` are the source of truth.**
   Tunarr Scheduler mirrors them as Malli, and Pseudovision's daily-slots
   handler consumes the resulting `DailySlot` JSON. Touching these contracts
   without coordinating both sides is a silent breakage (one side validates
   fine, the other rejects everything). Update the contracts first, then
   update the consumers in lockstep.
2. **LLM outputs are not clean.** `0584869` added option-set validation in
   `_categorize_single` and `map_media_to_channels` (re-prompts up to 2× on
   rejection, then filters as a final safety net). If you see a category
   value that doesn't match `resources/config.edn` `:categories`, it predates
   that commit OR came from a different code path. Add a regression test
   rather than widening the option set.
3. **`propose-quarterly-grid` calls are slow.** ~2 minutes per channel is
   normal. The scheduler's `run-quarterly!` loop is **delayed-tolerant**
   (commit `5d25df6`); if a request times out the next iteration is still
   valid. Don't tighten the timeout below 5 minutes.
4. **Catalog profiles are bounded by `TUNABRAIN_SCHEDULE_MAX_SHOWS`** (default
   300). Long-context models can take more; cost and latency scale linearly.
   Shows with no available episodes are always pruned first.
5. **Cost-tier routing is intentional, not a bug.** Tagging/flagging use
   the cheap model; quarterly scheduling uses the strong model. If a tagging
   response looks low-quality, check that `TUNABRAIN_LLM_MODEL` (the default)
   is still a budget model — accidentally pointing it at opus will burn cash
   for no quality gain.
6. **Wire format is JSON, internal model is snake_case.** The HTTP API
   accepts/returns snake_case keys (`tunabrain.scheduler.media/critic-rating`
   etc. in the scheduler's namespaced-key naming). Don't lowercase the model
   in `response_model()`; let Pydantic handle it.
7. **Don't read `TUNABRAIN_*` env vars at import time.** `python -m tunabrain`
   must be restartable for env-var changes to take effect. If you see
   stale-config bugs, check that the env is read inside the request handler
   (or memoized per-process) rather than at module load.

## Where to look next

- `README.md` — features, env vars, model selection
- `PLAN.md` — phase history and roadmap
- `docs/handoff-tunarr-pseudovision.md` — the cross-system spec (authoritative
  contract for the scheduler↔tunabrain integration)
- `docs/scheduling-grid-spec.md` — full prose for the expander
- `src/tunabrain/scheduling/grid.py` — Pydantic contracts
- `tests/test_grid_expander.py` — golden conformance suite that the
  scheduler's `expander.clj` is ported from
- `references/tunabrain-openapi-current.md` (in `pseudovision-ecosystem-development`
  skill) — endpoint inventory
