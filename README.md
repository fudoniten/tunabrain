# Tuna Brain

Tuna Brain is a FastAPI service that wraps upstream LLMs with LangChain to support
Tunarr Scheduler. It will help tag media, map media to channels, craft schedules, and
produce bumpers for upcoming programming blocks.

## Project layout

- `src/tunabrain/app.py`: FastAPI application factory and router wiring.
- `src/tunabrain/api/models.py`: Pydantic models for request/response payloads.
- `src/tunabrain/api/routes.py`: HTTP endpoints for tagging, channel mapping, scheduling, and bumpers.
- `src/tunabrain/chains/`: LangChain-powered workflow stubs, ready to be implemented.
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

### Endpoints

- `POST /tags`: Generate scheduling-oriented tags for a media item.
- `POST /channel-mapping`: Associate a media item with matching channels.
- `POST /schedule`: Build a schedule for a channel using provided media and instructions.
- `POST /bumpers`: Produce bumpers tailored to a schedule.

Each endpoint currently delegates to LangChain workflow stubs that raise
`NotImplementedError`, ready for future implementation.

