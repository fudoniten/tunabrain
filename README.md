# Tunabrain

Tunabrain is the LLM gateway for the Pseudovision ecosystem: a stateless
FastAPI service that wraps upstream models (via LangChain) behind typed,
purpose-built endpoints. Every AI call in the ecosystem goes through it, so no
other service holds model keys.

- **Tunarr Scheduler** calls it to categorize and tag media, govern tags,
  author quarterly grids and monthly overrides, review grids, and make bumper
  art.
- **Grout** calls it to enrich filler and long-form clips (`/enrich/*`).

Tunabrain keeps no state; TS and Grout own everything it produces.

Deployed at `https://tunabrain.kube.sea.fudo.link` (in-cluster:
`tunabrain.arr.svc.cluster.local:5546`).

Ongoing work lives in [PLAN.md](PLAN.md). Conventions and pitfalls for
contributors are in [AGENTS.md](AGENTS.md). The scheduling wire contracts and
the grid expansion algorithm are specified in
[docs/scheduling-grid-spec.md](docs/scheduling-grid-spec.md).

## Project layout

```
src/tunabrain/
├── app.py, __main__.py   FastAPI app and `python -m tunabrain` entry point
├── config.py, llm.py     env-var config and per-task model routing
├── api/                  routes.py (every endpoint) and models.py (Pydantic payloads)
├── chains/               one LangChain workflow per task (tagging, categorization,
│                         enrichment, tag governance, bumpers, ...)
├── scheduling/           grid.py (the scheduling contracts), quarterly_grid.py,
│                         monthly_overrides.py, review.py, expander.py (reference
│                         implementation for TS's port)
├── stt/, keyframes/      speech-to-text and keyframe captioning for /enrich/long-form
└── tools/                LangChain tools (Wikipedia lookup, title helpers)
tests/                    pytest suites, incl. the golden expander conformance tests
```

## Development

```bash
nix develop                     # Python 3.12 with all deps, pytest, ruff
python -m tunabrain             # listens on :8000 (--port to change)
curl localhost:8000/health
```

### Testing

```bash
pytest                          # full suite; no network or LLM needed
pytest tests/test_grid_expander.py
ruff check .
nix flake check                 # the same pytest run, against the locked env
```

### Configuring the LLM backend

TunaBrain reads environment variables to decide which chat model to use:

- `TUNABRAIN_LLM_PROVIDER`: LangChain provider name (default: `openai`).
- `TUNABRAIN_LLM_MODEL`: Default model identifier to load (default: `gpt-4o-mini`).
- `TUNABRAIN_DEBUG`: Set to `1`, `true`, or `yes` to force debug logging for LLM
  prompts and downstream HTTP requests, even before a request payload can be
  parsed.

Different tasks have different needs, so each can override the default model. A
task with no override falls back to `TUNABRAIN_LLM_MODEL`:

- `TUNABRAIN_SHOW_LLM_MODEL`: show tagging.
- `TUNABRAIN_EPISODE_LLM_MODEL`: episode special-flag detection.
- `TUNABRAIN_SCHEDULE_LLM_MODEL`: quarterly grid / schedule building.
- `TUNABRAIN_REVIEW_LLM_MODEL`: grid review and revision.
- `TUNABRAIN_BUMPERS_LLM_MODEL`: bumper image generation.

`TUNABRAIN_ENABLE_WIKIPEDIA_SEARCH` (default on) lets chains ground answers
on Wikipedia lookups.

Tagging and flagging are cheap, low-stakes, and run at high volume, so a small
fast model is a good fit. Quarterly scheduling is the opposite — it reasons over
the whole catalog and correctness matters — so point it at a strong long-context
model, e.g. on OpenRouter:

```bash
export TUNABRAIN_LLM_PROVIDER=openrouter
export TUNABRAIN_LLM_MODEL=deepseek/deepseek-v4-flash      # cheap default for tagging/flagging
export TUNABRAIN_SCHEDULE_LLM_MODEL=anthropic/claude-opus-4.8  # scheduling only
```

Scheduling prompts also expose a slice of the catalog to the model. Shows with no
available episodes are always pruned (they can't be scheduled); the per-show
detail list is then capped by:

- `TUNABRAIN_SCHEDULE_MAX_SHOWS`: how many schedulable shows to enumerate
  (default: `300`). Long-context models can afford a higher value; the catalog's
  aggregate shape (genres, runtimes, movie count) is always included regardless.

When using OpenAI, provide an API key via `OPENAI_API_KEY` (or rely on your shell's
existing `OPENAI_API_KEY` export). For example:

```bash
export OPENAI_API_KEY=sk-...
export TUNABRAIN_LLM_MODEL=gpt-4o-mini
python -m tunabrain
```

Every chain that invokes an LLM will pick up these settings automatically.

### Grout enrichment (STT + keyframes)

The `/enrich/long-form` endpoint transcribes media before categorizing it, using
one of the cluster's speech-to-text services. These are pluggable; a default is
chosen here and can be overridden per request via `options.stt_backend`:

- `TUNABRAIN_STT_WHISPER_URL`: whisper-http (OpenAI-compatible) base URL
  (default: `http://whisper-http.wyoming.svc.cluster.local:10301`).
- `TUNABRAIN_STT_SUBGEN_URL`: subgen base URL
  (default: `http://subgen.arr.svc.cluster.local:9000`).
- `TUNABRAIN_STT_DEFAULT_BACKEND`: `whisper-http`, `subgen`, or `auto`
  (default: `auto` — probes both and uses whichever health endpoint responds
  first, falling back to the other if the winner's transcription fails).
- `TUNABRAIN_STT_WHISPER_MODEL`: model name requested from whisper-http
  (default: `turbo` — the only model registered in the current deployment; do
  **not** set this to `large-v3`, the server will reject it).
- `TUNABRAIN_SCRATCH_DIR`: shared scratch space for fetched media and extracted
  audio/keyframes (default: `/tmp/tunabrain-scratch`). `file_id` sources are
  resolved relative to this directory.
- `TUNABRAIN_ENRICH_LONG_TIMEOUT`: hard cap in seconds for the whole long-form
  pipeline (default: `900`). On timeout the request returns a degraded response
  with a warning rather than hanging.

`/enrich/long-form` shells out to `ffmpeg` (already a Nix shell dependency) to
extract audio and keyframes.

### Endpoints

The live list is at `GET /openapi.json`.

| Endpoint | Purpose | Caller |
|---|---|---|
| `POST /categorize` | Categorize media across caller-supplied dimensions | TS |
| `POST /tags` | Free-form scheduling tags for a media item | TS |
| `POST /tags/audit`, `POST /tag-governance/triage` | Recommend tag deletions; keep/remove/rename decisions | TS |
| `POST /tags/episode-special-flag` | Flag special episodes | TS |
| `POST /bumpers` | Bumper art (base64 PNG) and theme | TS |
| `POST /enrich/short-form` | Categorize + tag filler, bumpers, ads (no STT) | Grout |
| `POST /enrich/long-form` | Fetch → STT (+ keyframe captions) → categorize + tag | nobody yet (Grout should) |
| `POST /enrich/profile` | One shared profile for a directory group | Grout |
| `POST /enrich/describe` | Name/description; used inside `/enrich/short-form` | internal |
| `POST /api/scheduling/propose-quarterly-grid` | Propose a channel's frozen grid in one call | TS |
| `POST /api/scheduling/propose-daypart-skeleton`, `propose-strip-fill` | The same proposal as two round trips, so TS can supply a duration-feasible menu | TS |
| `POST /api/scheduling/repair-quarterly-grid` | Repair a grid against a feasibility report | TS |
| `POST /api/scheduling/review-grid`, `revise-grid` | Taste review of a concrete week, and revision | TS |
| `POST /api/scheduling/propose-monthly-overrides` | Sparse monthly overrides against a frozen grid | TS |
| `POST /channel-mapping` | Deprecated; use the `channel` dimension in `/categorize` | none |
| `POST /api/scheduling/get-{quarterly,monthly}-strategy` | Strategy proposals with placeholder cost estimates | none |
| `GET /health`, `GET /api/version` | Liveness and build info | |

The endpoints with no caller are slated for removal (see [PLAN.md](PLAN.md)).
