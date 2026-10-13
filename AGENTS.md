# AGENTS.md — Tunabrain

> Working notes for AI agents and humans changing this repo. Start with
> [README.md](README.md) (what Tunabrain is, endpoints, env vars, how to run
> and test it) and [PLAN.md](PLAN.md) (what's being worked on).

## Ground rules

- **Stateless.** Tunabrain never persists anything. Tunarr Scheduler stores
  grids, overrides, and tags; Grout stores enrichment results and replays any
  grounding context it wants reused.
- **Typed endpoints only.** Each caller need gets a purpose-built endpoint with
  Pydantic request and response models. There is no generic chat-completions
  passthrough.
- **The LLM never does capacity math or sees raw media for scheduling.** It
  gets a `CatalogProfile` (and a `FeasibilityReport` or candidate menu from TS)
  and returns rules, not slots.

## Contracts

`src/tunabrain/scheduling/grid.py` is the source of truth for the scheduling
wire format. TS mirrors it as Malli schemas in `scheduling/contracts.clj`, and
the prose spec is [docs/scheduling-grid-spec.md](docs/scheduling-grid-spec.md).
When you change a contract, change all three, and keep
`tests/test_grid_expander.py` in step with TS's ported
`expander_golden_test.clj`. A contract changed on one side only fails
silently: one side validates, the other rejects everything.

The HTTP API is snake_case JSON. Don't rename fields in `response_model`; let
Pydantic serialize them.

## Common pitfalls

1. **LLM output isn't clean.** Categorization re-prompts up to twice when a
   value falls outside the caller's option set, then filters what's left. If
   an out-of-vocabulary value leaks through, add a regression test rather than
   widening the option set.
2. **Quarterly grid proposals are slow**, about two minutes per channel. Keep
   that in mind before tightening timeouts on either side.
3. **Cost-tier routing is intentional.** Tagging and flagging use the cheap
   default model (`TUNABRAIN_LLM_MODEL`); scheduling and review use stronger
   per-task models. Pointing the default at an expensive model burns money for
   no quality gain.
4. **Catalog size is capped** by `TUNABRAIN_SCHEDULE_MAX_SHOWS` (default 300).
   Shows with no available episodes are pruned first; cost and latency scale
   with the cap.
5. **Read `TUNABRAIN_*` env vars at request time** (or memoize per process),
   not at import time, so a restart picks up config changes.
6. **`whisper-http` only serves the `turbo` model** in the current deployment.
   Setting `TUNABRAIN_STT_WHISPER_MODEL=large-v3` makes every transcription
   fail.
