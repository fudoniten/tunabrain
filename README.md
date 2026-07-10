# Tuna Brain

Tuna Brain is a FastAPI service that wraps upstream LLMs with LangChain to support
Tunarr Scheduler. It tags media, maps media to channels, authors layered TV
programming grids, and produces bumpers for upcoming programming blocks.

## Project layout

- `src/tunabrain/app.py`: FastAPI application factory and router wiring.
- `src/tunabrain/api/models.py`: Pydantic models for request/response payloads.
- `src/tunabrain/api/routes.py`: HTTP endpoints for tagging, channel mapping,
  layered-grid scheduling (`/api/scheduling/*`), and bumpers.
- `src/tunabrain/scheduling/`: the layered-grid scheduling pipeline —
  `quarterly_grid.py` (two-pass dayparting + strip-fill, plus repair),
  `monthly_overrides.py` (sparse overrides against a frozen grid), `grid.py`
  (the shared Pydantic contracts), `expander.py` (the golden Grid+Overrides
  → DailySlot conformance suite tunarr-scheduler ports). See
  `docs/scheduling-grid-spec.md` and `AGENTS.md` for the authoritative design
  and the live endpoint list.
- `src/tunabrain/chains/`: LangChain-powered workflow stubs (tagging, channel
  mapping, enrichment, bumpers), ready to be implemented.
- `src/tunabrain/tools/`: LangChain-compatible tools (e.g., Wikipedia lookup) available to
  chains.
- `flake.nix`: Nix flake for a reproducible development shell with Python dependencies.
- `pyproject.toml`: Project metadata and Python dependencies.

## Development

### Using Nix

```bash
nix develop
```

This provides a Python 3.11 environment with FastAPI, Uvicorn, Pydantic, and LangChain.

### Running the API

```bash
python -m tunabrain
```

The service will start on port 8000. The `/health` endpoint can be used to verify
startup.

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

- `POST /tags`: Generate scheduling-oriented tags for a media item.
- `POST /categorize`: Categorize a media item across caller-supplied dimensions.
- `POST /channel-mapping`: Associate a media item with matching channels (deprecated; use the `channel` dimension in `/categorize`).
- `POST /bumpers`: Produce bumpers tailored to a schedule.
- `POST /enrich/short-form`: One-call enrichment for short-form media (bumpers,
  fillers, ads, music videos). Orchestrates `/categorize` + `/tags`; no STT.
- `POST /enrich/long-form`: One-call enrichment for long-form media
  (documentaries, video essays, interviews). Fetches the media, extracts audio,
  transcribes it (STT), optionally captions keyframes, then runs
  `/categorize` + `/tags` grounded on the transcript.

### Layered-grid scheduling endpoints

These are the live scheduling API — see `AGENTS.md` for the full request/response
shapes and `docs/scheduling-grid-spec.md` for the authoritative design:

- `POST /api/scheduling/propose-quarterly-grid`: propose a channel's frozen
  quarterly Grid (two-pass dayparting + strip-fill).
- `POST /api/scheduling/propose-daypart-skeleton` / `propose-strip-fill`: the
  same two passes as a split round trip, so the caller can compute a
  duration-feasible candidate menu between them.
- `POST /api/scheduling/repair-quarterly-grid`: targeted repair against a
  deterministic feasibility report.
- `POST /api/scheduling/propose-monthly-overrides`: sparse monthly overrides
  against a frozen grid.

(`POST /schedule`, an earlier LangGraph-agent-based scheduler, was removed —
it implemented only 2 of a planned 7 tools and was superseded by the
endpoints above before it ever had a production caller.)

